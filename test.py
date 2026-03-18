import os
import json
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.quantization

class TaskAwareEdgeDetector(nn.Module):
    def __init__(self, num_tasks=14):
        super(TaskAwareEdgeDetector, self).__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.vision_backbone = mobilenet.features
        self.num_channels = 1280
        
        self.task_embedding = nn.Embedding(num_embeddings=num_tasks, embedding_dim=self.num_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.regression_head = nn.Sequential(
            nn.Linear(self.num_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(256, 4)
        )

    def forward(self, image, task_id):
        vision_features = self.vision_backbone(image)
        text_features = self.task_embedding(task_id).unsqueeze(2).unsqueeze(3)
        fused_features = vision_features * torch.sigmoid(text_features)
        pooled_features = self.pool(fused_features)
        flattened = pooled_features.view(pooled_features.size(0), -1)
        raw_box_coords = self.regression_head(flattened)
        return torch.sigmoid(raw_box_coords)

def prepare_model_for_fpga(model):
    # Model must be in train mode to insert FakeQuantize nodes for QAT
    model.train() 
    model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(model, inplace=True)
    return model

class COCOTasksDataset(Dataset):
    def __init__(self, img_dir, annotation_file, task_id, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            self.task_data = json.load(f)
        self.samples = self._parse_to_flat_list(self.task_data, task_id)

    def _parse_to_flat_list(self, raw_json, task_id):
        parsed_data = []
        image_info = {img['id']: {'w': img['width'], 'h': img['height']} for img in raw_json.get('images', [])}
        images_dict = {}

        for ann in raw_json.get('annotations', []):
            if ann['category_id'] == 1:
                img_id = ann['image_id']
                orig_w, orig_h = image_info[img_id]['w'], image_info[img_id]['h']
                x_min, y_min, w, h = ann['bbox']

                norm_w, norm_h = w / orig_w, h / orig_h
                norm_cx, norm_cy = (x_min + (w / 2.0)) / orig_w, (y_min + (h / 2.0)) / orig_h

                if img_id not in images_dict:
                    images_dict[img_id] = {
                        'file_name': f"COCO_train2014_{img_id:012d}.jpg",
                        'task_id': task_id,
                        'bbox': [norm_cx, norm_cy, norm_w, norm_h]
                    }

        for img_id, data in images_dict.items():
            if os.path.exists(os.path.join(self.img_dir, data['file_name'])):
                parsed_data.append(data)
        return parsed_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(os.path.join(self.img_dir, sample['file_name'])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(sample['task_id'] - 1, dtype=torch.long), torch.tensor(sample['bbox'], dtype=torch.float32)

TASK_NAMES = [
    "step on", "sit comfortably", "place flowers", "get potatoes out of fire",
    "water plant", "get lemon out of tea", "dig hole", "open bottle of beer",
    "open parcel", "serve wine", "pour sugar", "smear butter", "extinguish fire", "pound carpet"
]

def calculate_iou(pred_boxes, target_boxes):
    px1, py1 = pred_boxes[:, 0] - (pred_boxes[:, 2] / 2), pred_boxes[:, 1] - (pred_boxes[:, 3] / 2)
    px2, py2 = pred_boxes[:, 0] + (pred_boxes[:, 2] / 2), pred_boxes[:, 1] + (pred_boxes[:, 3] / 2)
    
    tx1, ty1 = target_boxes[:, 0] - (target_boxes[:, 2] / 2), target_boxes[:, 1] - (target_boxes[:, 3] / 2)
    tx2, ty2 = target_boxes[:, 0] + (target_boxes[:, 2] / 2), target_boxes[:, 1] + (target_boxes[:, 3] / 2)

    ix1, iy1 = torch.max(px1, tx1), torch.max(py1, ty1)
    ix2, iy2 = torch.min(px2, tx2), torch.min(py2, ty2)

    inter_area = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    pred_area = (px2 - px1) * (py2 - py1)
    target_area = (tx2 - tx1) * (ty2 - ty1)

    union_area = pred_area + target_area - inter_area
    return inter_area / (union_area + 1e-6)

def visualize_prediction(image_tensor, pred_box, target_box, task_id_num, iou_score):
    pcx, pcy, pw, ph = pred_box
    real_pw, real_ph = pw * 224.0, ph * 224.0
    px_min, py_min = (pcx * 224.0) - (real_pw / 2.0), (pcy * 224.0) - (real_ph / 2.0)

    img_for_plot = image_tensor.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img_for_plot)
    
    if iou_score is not None:
        ax.set_title(f"Task: \"{TASK_NAMES[task_id_num]}\" | IoU: {iou_score:.3f}", fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"Task: \"{TASK_NAMES[task_id_num]}\"", fontsize=14, fontweight='bold')

    ax.add_patch(patches.Rectangle((px_min, py_min), real_pw, real_ph, linewidth=3, edgecolor='#FF0000', facecolor='none', label='Prediction'))

    if target_box is not None:
        tcx, tcy, tw, th = target_box
        real_tw, real_th = tw * 224.0, th * 224.0
        tx_min, ty_min = (tcx * 224.0) - (real_tw / 2.0), (tcy * 224.0) - (real_th / 2.0)
        ax.add_patch(patches.Rectangle((tx_min, ty_min), real_tw, real_th, linewidth=3, edgecolor='#00FF00', facecolor='none', label='Ground Truth'))

    plt.legend(frameon=True, facecolor='white', framealpha=1.0, fontsize=12, loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def load_and_evaluate(mode='random', model_weights_path='./dvcon_task_aware_model_v2.pth', 
                      custom_image_path=None, custom_task_id=None, custom_target_box=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model to {device}...")

    base_model = TaskAwareEdgeDetector(num_tasks=14)
    qat_model = prepare_model_for_fpga(base_model)
    
    try:
        qat_model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Warning: Could not find model weights at '{model_weights_path}'. Using uninitialized weights.")
        
    qat_model.to(device)
    qat_model.eval() 

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])

    if mode == 'random':
        random_task_id = random.randint(1, 14)
        json_path = f"./dataset/annotations/task_{random_task_id}_train.json"
        
        try:
            dataset = COCOTasksDataset('./dataset/images', json_path, task_id=random_task_id, transform=transform)
            if len(dataset) == 0:
                print("No valid images found for this task. Run again.")
                return

            rand_idx = random.randint(0, len(dataset) - 1)
            image, task_id, target_bbox = dataset[rand_idx]

            images_batch = image.unsqueeze(0).to(device)
            task_ids_batch = task_id.unsqueeze(0).to(device)
            target_bboxes_batch = target_bbox.unsqueeze(0).to(device)

            with torch.no_grad():
                pred_bboxes = qat_model(images_batch, task_ids_batch).view(-1, 4)
                iou_score = calculate_iou(pred_bboxes, target_bboxes_batch).item()

            print(f"Evaluated Task {random_task_id}! IoU: {iou_score:.4f}")
            visualize_prediction(image, pred_bboxes[0].cpu().numpy(), target_bbox.numpy(), task_id.item(), iou_score)
            
        except FileNotFoundError:
            print(f"Error: Could not find COCO dataset files at '{json_path}'. Make sure your ./dataset/ folder is structured correctly.")

    elif mode == 'custom':
        if custom_image_path is None or not os.path.exists(custom_image_path):
            print(f"Error: Could not find image at '{custom_image_path}'")
            return
            
        if custom_task_id is None or not (1 <= custom_task_id <= 14):
            print("Error: custom_task_id must be a number between 1 and 14.")
            return

        raw_image = Image.open(custom_image_path).convert('RGB')
        image_tensor = transform(raw_image)
        
        images_batch = image_tensor.unsqueeze(0).to(device)
        
        # Internal embedding layer requires 0-indexed task IDs (0-13)
        task_ids_batch = torch.tensor([custom_task_id - 1], dtype=torch.long).to(device)

        with torch.no_grad():
            pred_bboxes = qat_model(images_batch, task_ids_batch).view(-1, 4)
            
            if custom_target_box is not None:
                target_bboxes_batch = torch.tensor(custom_target_box, dtype=torch.float32).unsqueeze(0).to(device)
                iou_score = calculate_iou(pred_bboxes, target_bboxes_batch).item()
                target_box_pass = torch.tensor(custom_target_box, dtype=torch.float32)
            else:
                iou_score = None
                target_box_pass = None

        print(f"Evaluated Custom Image for Task: \"{TASK_NAMES[custom_task_id - 1]}\"")
        visualize_prediction(image_tensor, pred_bboxes[0].cpu().numpy(), target_box_pass, custom_task_id - 1, iou_score)

if __name__ == "__main__":
    TEST_MODE = 'random' 
    MY_IMAGE_PATH = "./test6.jpg" 
    
    # Task IDs (1-14):
    # 1: step on, 2: sit comfortably, 3: place flowers, 4: get potatoes out of fire
    # 5: water plant, 6: get lemon out of tea, 7: dig hole, 8: open bottle of beer
    # 9: open parcel, 10: serve wine, 11: pour sugar, 12: smear butter, 13: extinguish fire, 14: pound carpet
    MY_TASK_ID = 9
    
    # Format: [norm_cx, norm_cy, norm_w, norm_h] or None
    MY_TARGET_BOX = [0.52, 0.6, 0.2, 0.1] 
    
    load_and_evaluate(
        mode=TEST_MODE,
        custom_image_path=MY_IMAGE_PATH,
        custom_task_id=MY_TASK_ID,
        custom_target_box=MY_TARGET_BOX
    )