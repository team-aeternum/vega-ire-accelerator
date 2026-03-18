import os
import json
import time
import zipfile
import shutil
from io import BytesIO

import requests
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.quantization

import hls4ml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

os.makedirs('./dataset/images', exist_ok=True)
os.makedirs('./dataset/annotations', exist_ok=True)

def download_with_progress(url, dest_path):
    """Downloads large files with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.split('/')[-1],
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))


if not os.path.exists('./dataset/annotations/task_1_train.json'):
    print("Downloading annotations...")
    ann_url = "https://github.com/coco-tasks/dataset/archive/refs/heads/master.zip"
    download_with_progress(ann_url, './dataset/annotations.zip')

    with zipfile.ZipFile('./dataset/annotations.zip', 'r') as zip_ref:
        zip_ref.extractall('./dataset/temp_ann')

    for root, dirs, files in os.walk('./dataset/temp_ann'):
        for file in files:
            if file.endswith('.json') and 'train' in file:
                shutil.move(os.path.join(root, file), os.path.join('./dataset/annotations', file))

    shutil.rmtree('./dataset/temp_ann')
    os.remove('./dataset/annotations.zip')


if not os.path.exists('./dataset/images') or len(os.listdir('./dataset/images')) < 100:
    print("Downloading COCO 2014 Images...")
    img_url = "http://images.cocodataset.org/zips/train2014.zip"
    download_with_progress(img_url, './dataset/train2014.zip')

    print("Extracting images...")
    with zipfile.ZipFile('./dataset/train2014.zip', 'r') as zip_ref:
        zip_ref.extractall('./dataset/temp_images')
    
    for file in os.listdir('./dataset/temp_images/train2014'):
        shutil.move(os.path.join('./dataset/temp_images/train2014', file), './dataset/images/')
        
    shutil.rmtree('./dataset/temp_images')
    os.remove('./dataset/train2014.zip')

class COCOTasksDataset(Dataset):
    def __init__(self, img_dir, annotation_file, task_id, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.task_data = json.load(f)

        self.samples = self._parse_to_flat_list(self.task_data, task_id)

    def _parse_to_flat_list(self, raw_json, task_id):
        parsed_data = []
        annotations = raw_json['annotations']

        # Lookup dictionary for the original width and height of every image
        image_info = {img['id']: {'w': img['width'], 'h': img['height']} for img in raw_json['images']}
        images_dict = {}

        for ann in annotations:
            if ann['category_id'] == 1:
                img_id = ann['image_id']
                orig_w = image_info[img_id]['w']
                orig_h = image_info[img_id]['h']
                x_min, y_min, w, h = ann['bbox']

                # Convert to normalized Center-X, Center-Y, Width, Height (YOLO format)
                norm_w = w / orig_w
                norm_h = h / orig_h
                norm_cx = (x_min + (w / 2.0)) / orig_w
                norm_cy = (y_min + (h / 2.0)) / orig_h

                converted_bbox = [norm_cx, norm_cy, norm_w, norm_h]

                if img_id not in images_dict:
                    file_name = f"COCO_train2014_{img_id:012d}.jpg"
                    images_dict[img_id] = {
                        'file_name': file_name,
                        'task_id': task_id,
                        'bbox': converted_bbox
                    }

        for img_id, data in images_dict.items():
            img_path = os.path.join(self.img_dir, data['file_name'])
            if os.path.exists(img_path):
                parsed_data.append(data)

        return parsed_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, sample['file_name'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        task_id = torch.tensor(sample['task_id'] - 1, dtype=torch.long)
        target_bbox = torch.tensor(sample['bbox'], dtype=torch.float32)

        return image, task_id, target_bbox

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

all_task_datasets = []

for task_id in range(1, 15):
    json_path = f"./dataset/annotations/task_{task_id}_train.json"
    if os.path.exists(json_path):
        task_dataset = COCOTasksDataset('./dataset/images', json_path, task_id=task_id, transform=transform)
        all_task_datasets.append(task_dataset)

if len(all_task_datasets) > 0:
    full_train_dataset = ConcatDataset(all_task_datasets)
    real_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    print(f"Loaded {len(full_train_dataset)} training images.")

class TaskAwareEdgeDetector(nn.Module):
    def __init__(self, num_tasks=14):
        super(TaskAwareEdgeDetector, self).__init__()
        
        # Lightweight MobileNetV2 for FPGA deployment
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.vision_backbone = mobilenet.features
        self.num_channels = 1280
        
        # Language Encoder
        self.task_embedding = nn.Embedding(num_embeddings=num_tasks, embedding_dim=self.num_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(self.num_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(256, 4)
        )

    def forward(self, image, task_id):
        vision_features = self.vision_backbone(image)
        text_features = self.task_embedding(task_id).unsqueeze(2).unsqueeze(3)
        
        # Cross-Modal Spatial Attention Fusion
        fused_features = vision_features * torch.sigmoid(text_features)
        
        pooled_features = self.pool(fused_features)
        flattened = pooled_features.view(pooled_features.size(0), -1)
        raw_box_coords = self.regression_head(flattened)
        
        # Sigmoid clamps output to [0, 1] for normalized coordinates
        return torch.sigmoid(raw_box_coords)

def prepare_model_for_fpga(model):
    """Simulates edge-device INT8 precision during the training loop."""
    model.train()
    # 'qnnpack' is the standard backend for ARM and edge deployments
    model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(model, inplace=True)
    return model

base_model = TaskAwareEdgeDetector(num_tasks=14)
qat_model = prepare_model_for_fpga(base_model)
qat_model = qat_model.to(device)

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(qat_model.parameters(), lr=0.0001)
epochs = 75

if len(all_task_datasets) > 0:
    print(f"Starting training for {epochs} epochs on {device}")
    
    for epoch in range(epochs):
        qat_model.train()
        running_loss = 0.0

        for batch_idx, (images, task_ids, target_bboxes) in enumerate(real_train_loader):
            images = images.to(device)
            task_ids = task_ids.to(device)
            target_bboxes = target_bboxes.to(device)

            optimizer.zero_grad()
            
            predicted_bboxes = qat_model(images, task_ids)
            predicted_bboxes = predicted_bboxes.view(predicted_bboxes.size(0), -1)[:, :4]

            loss = criterion(predicted_bboxes, target_bboxes)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch [{batch_idx}/{len(real_train_loader)}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(real_train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Completed - Avg Loss: {epoch_loss:.4f}")

    save_path = './dvcon_task_aware_model_v2.pth'
    torch.save(qat_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

TASK_NAMES = [
    "step on", "sit comfortably", "place flowers", "get potatoes out of fire",
    "water plant", "get lemon out of tea", "dig hole", "open bottle of beer",
    "open parcel", "serve wine", "pour sugar", "smear butter", "extinguish fire", "pound carpet"
]

if len(all_task_datasets) > 0:
    qat_model.eval()
    images, task_ids, target_bboxes = next(iter(real_train_loader))
    images, task_ids_gpu = images.to(device), task_ids.to(device)

    with torch.no_grad():
        predicted_bboxes = qat_model(images, task_ids_gpu)
        predicted_bboxes = predicted_bboxes.view(predicted_bboxes.size(0), -1)[:, :4]

    idx = 0
    image_tensor = images[idx].cpu()
    pred_box = predicted_bboxes[idx].cpu().numpy()
    target_box = target_bboxes[idx].numpy()

    task_id_num = task_ids[idx].item()
    task_name = TASK_NAMES[task_id_num]

    # Unpack Center-Based Coordinates
    tcx, tcy, tw, th = target_box
    real_tw, real_th = tw * 224.0, th * 224.0
    tx_min, ty_min = (tcx * 224.0) - (real_tw / 2.0), (tcy * 224.0) - (real_th / 2.0)

    pcx, pcy, pw, ph = pred_box
    real_pw, real_ph = pw * 224.0, ph * 224.0
    px_min, py_min = (pcx * 224.0) - (real_pw / 2.0), (pcy * 224.0) - (real_ph / 2.0)

    img_for_plot = image_tensor.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img_for_plot)
    ax.set_title(f"Task: \"{task_name}\" (ID: {task_id_num + 1})", fontsize=14, fontweight='bold')

    true_rect = patches.Rectangle((tx_min, ty_min), real_tw, real_th, linewidth=3, edgecolor='#00FF00', facecolor='none', label='Correct Answer')
    ax.add_patch(true_rect)

    pred_rect = patches.Rectangle((px_min, py_min), real_pw, real_ph, linewidth=3, edgecolor='#FF0000', facecolor='none', label='Model Prediction')
    ax.add_patch(pred_rect)

    plt.legend(frameon=True, facecolor='white', framealpha=1.0, fontsize=12, loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.show()