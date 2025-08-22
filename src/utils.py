#---------------------------------------------------------------------------------------------------------------------------------
#==============================================Segmentation=======================================================================
#----------------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import cv2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np
from sklearn.metrics import jaccard_score
import pandas as pd

#-------------------------------------------------------------------------------------------------------------------------------
def create_mask_from_polygons(polygons, height, width):
    from pycocotools import mask as maskUtils
    """
    Convert polygon annotations to binary mask
    Args:
        polygons (list): list of polygon points from COCO
        height (int): image height
        width(int): image width
    Returns:
        np.ndarray: binary mask (H x W)
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    mask = maskUtils.decode(rle)

    return mask

#-------------------------------------------------------------------------------------------------------------------------

def convert_coco_to_masks(coco_json_path, images_dir, masks_dir):
    import os
    import json
    import numpy as np
    from PIL import Image

    # Load COCO annotations
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Create output mask directory
    os.makedirs(masks_dir, exist_ok=True)

    # Map image ids to file names and sizes
    images_info = {img['id']: img for img in coco['images']}

    # Group annotations by image id
    anns_per_img = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        anns_per_img.setdefault(img_id, []).append(ann)

    # Process each image
    for img_id, anns in anns_per_img.items():
        info = images_info[img_id]
        height, width = info['height'], info['width']
        # initialize empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # combine all polygons into single mask (binary mask for conjunctiva)
        for ann in anns:
            polygons = ann['segmentation']
            poly_mask = create_mask_from_polygons(polygons, height, width)
            mask = np.maximum(mask, poly_mask*255)

        # save mask as PNG image
        image_filename = info['file_name']
        mask_filename = os.path.splitext(image_filename)[0] + '.png'
        mask_path = os.path.join(masks_dir, mask_filename)
        Image.fromarray(mask).save(mask_path)

    print(f'saved masks to {masks_dir}')

#---------------------------------------------------------------------------------------------------------------------

class ConjunctivaSegDataset(Dataset):


    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Skip any non-image files like .json
        self.image_filenames = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        basename = os.path.splitext(image_filename)[0]

        img_path = os.path.join(self.images_dir, image_filename)
        mask_path = os.path.join(self.masks_dir, basename + '.png')

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found or corrupted: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found or corrupted: {mask_path}")
        mask = (mask > 0).astype('float32')

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # already a tensor: [3, H, W]
            mask = augmented['mask'].unsqueeze(0)  # [1, H, W]

        return image, mask

#--------------------------------------------------------------------------------------------------------------------------

import segmentation_models_pytorch as smp
import torch.nn as nn

class ConjunctivaSegmentationModel(nn.Module):
    def __init__(self):
        super(ConjunctivaSegmentationModel, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet18",        # You can also try: efficientnet-b0, mobilenet_v2
            encoder_weights="imagenet",     # Pretrained encoder on ImageNet
            in_channels=3,                  # RGB input
            classes=1,                      # Binary segmentation
            activation=None                 # We'll apply sigmoid during loss or evaluation
        )

    def forward(self, x):
        return self.model(x)

#---------------------------------------------------------------------------------------------------------------------------

# Dice Loss Definition
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# Evaluation Function
def evaluate(model, val_loader, bce_loss, dice_loss, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            if images.dim() != 4 or images.size(1) != 3:
                raise ValueError(f"Expected input shape [B, 3, H, W], but got {images.shape}")

            outputs = model(images)
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds).reshape(-1)
    all_masks = np.concatenate(all_masks).reshape(-1)

    iou = jaccard_score(all_masks, all_preds, zero_division=0)
    dice = (2. * (all_preds * all_masks).sum()) / (all_preds.sum() + all_masks.sum() + 1e-7)

    return total_loss / len(val_loader), dice, iou

# Training Function
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    dice_loss = DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            if images.dim() != 4 or images.size(1) != 3:
                raise ValueError(f"Expected input shape [B, 3, H, W], but got {images.shape}")

            outputs = model(images)
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

        val_loss, dice_score, iou = evaluate(model, val_loader, bce_loss, dice_loss, device)
        scheduler.step(val_loss)
        print(f"Validation Loss: {val_loss:.4f}, Dice: {dice_score:.4f}, IoU: {iou:.4f}")

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")


    return model

#---------------------------------------------------------------------------------------------------------------------------------
def segment_and_crop(image: Image.Image, unet_model, device):
    # from utils import val_transforms

    seg_transforms = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img_tensor = seg_transforms(image).unsqueeze(0).to(device)

    unet_model.eval()
    with torch.no_grad():
        mask = torch.sigmoid(unet_model(img_tensor))
    
    mask = (mask > 0.7).float().squeeze().cpu().numpy()

    orig_w, orig_h = image.size
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    img_np = np.array(image)
    conjunctiva = cv2.bitwise_and(img_np, img_np, mask=mask_resized.astype(np.uint8))

    coords = cv2.findNonZero(mask_resized.astype(np.uint8))
    x, y, w, h = cv2.boundingRect(coords)
    cropped = conjunctiva[y:y+h, x:x+w]

    # final_tensor = val_transforms(Image.fromarray(cropped)).unsqueeze(0).to(device)

    # return final_tensor, cropped, mask_resized
    return cropped, mask_resized

#---------------------------------------------------------------------------------------------------------------------------------------
# ===================================================Regression====================================================
#---------------------------------------------------------------------------------------------------------------------------------------

class RGB2LAB_CLAHE(object):

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        # PIL -> array
        img = np.array(img)
        # RGB -> LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # Split channels
        L, A, B = cv2.split(lab)
        # Apply CLAHE only on L channel
        L = self.clahe.apply(L)
        # Merge back 
        lab = cv2.merge((L, A, B))
        # Convert back to RGB (optional)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        # Ensuring valid uint8 range
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        return Image.fromarray(rgb)
    
#-------------------------------------------------------------------------------------------------------------------

def compute_lab_mean_std(image_dir, clip_limit=2.0, tile_grid_size=(8,8)):
    from tqdm import tqdm
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    means, stds = [], []

    # loop through images
    for img_name in tqdm(os.listdir(image_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(image_dir, img_name)
        img = np.array(Image.open(img_path).convert("RGB"))

        # RGB -> LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # CLAHE on L-channel
        L, A, B = cv2.split(lab)
        L = clahe.apply(L)
        lab = cv2.merge((L, A, B))

        # scale to [0,1] before computing mean/std
        lab = lab.astype(np.float32) / 255.0

        # per-channel mean/std
        means.append(lab.reshape(-1, 3).mean(axis=0))
        stds.append(lab.reshape(-1, 3).std(axis=0))

    # average across dataset
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return mean, std

#------------------------------------------------------------------------------------------------------------------

class LogCoshLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        x = pred - target

        x = x.clamp(min=-20, max=20)
        loss = x + torch.nn.functional.softplus(-2.0 * x) - torch.log(torch.tensor(2.0, device=x.device))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

        return loss
    
#-----------------------------------------------------------------------------------------------------------------------

train_transforms = transforms.Compose([
    RGB2LAB_CLAHE(), 
    transforms.Resize((256, 256)), 
    transforms.RandomResizedCrop(224, scale=(0.9, 1.25)), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomRotation(15), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.289, 0.545, 0.505], std=[0.2864, 0.0456, 0.0205])
])

val_transforms = transforms.Compose([
    RGB2LAB_CLAHE(), 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.289, 0.545, 0.505], std=[0.2864, 0.0456, 0.0205])
])

#--------------------------------------------------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        attn = self.conv(x)   # B, 1, H, W
        attn = torch.sigmoid(attn)  # [0, 1]

        return x * attn   # broadcast multiply
    
#-------------------------------------------------------------------------------------------------------------------------

class HgbCNN(nn.Module):
    def __init__(self):
        super(HgbCNN, self).__init__()

        self.features = nn.Sequential(
            ## First Convolutional Block
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=2, groups=3),
            nn.GroupNorm(num_groups=1, num_channels=3), 
            nn.ReLU(),  
            
            # Second Convolutional Block
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=2), 
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third Convolutional Block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, groups=32), 
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth Convolutional Block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=1, dilation=2), 
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU()
        )

        # Spatial Attention Block
        self.attn = SpatialAttention(in_channels=128)
            
        # Classifier Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # init
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x), inplace=True)
        x = nn.functional.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)

        return x
    
#--------------------------------------------------------------------------------------------------------------------------

def model_summary(model):
    print('===============================================Model Summary=============================================\n')

    for name, param in model.named_parameters():
        status = 'Trainable' if param.requires_grad else 'Non-Trainable (Frozen)'
        print(f'{name:30} | {status}')

    print('\n========================================================================================================\n')

    total_params = sum(param.numel() for param in model.parameters()) 
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen_params = total_params - trainable_params

    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')
    print(f'Frozen Parameters: {frozen_params:,}\n')

#--------------------------------------------------------------------------------------------------------------------------------

class ConjunctivaDataset(Dataset):

    def __init__(self, image_dir, metadata_csv_path, transforms=None):
        self.image_dir = image_dir
        self.metadata = pd.read_csv(metadata_csv_path)
        self.transforms = transforms

        self.valid_exts = ('.jpg', '.jpeg', '.png')
        self.image_files = [img for img in os.listdir(self.image_dir) 
                            if img.lower().endswith(self.valid_exts) and os.path.splitext(img)[0] in list(self.metadata['Number'])]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_id = os.path.splitext(img_path)[0]
        
        image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
        hb_val = float(self.metadata.loc[self.metadata['Number'] == img_id, 'Hgb'].values[0])

        if self.transforms:
            image = self.transforms(image)
            
        return image, hb_val

#------------------------------------------------------------------------------------------------------------------------

class EarlyStopping():
    def __init__(self, patience=5, tol=0.01):
        self.patience = patience
        self.tol = tol
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.tol:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

#-------------------------------------------------------------------------------------------------------

def train_model(model, model_name, train_loader, val_loader, epochs=50, learning_rate=1e-1):
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    # loss_fn = LogCoshLoss()
    # loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, epochs=epochs, 
    #                                                 steps_per_epoch=len(train_loader), pct_start=0.1, 
    #                                                 div_factor=10, final_div_factor=100)
    
    early_stopping = EarlyStopping(patience=5)
    
    writer = SummaryWriter(log_dir=f'../reports/{model_name}')

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        train_progress_bar = tqdm(train_loader, desc=f'Epoch: {epoch+1}/{epochs}', leave=True)

        # image, hb_val => x, y
        for x, y in train_progress_bar:
            x, y = x.to(device), y.float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            train_loss += loss.item() * x.size(0)

            train_progress_bar.set_postfix(loss=f'{loss.item():.3f}')

        train_epoch_loss = train_loss / len(train_loader.dataset)


        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc=f'Epoch: {epoch+1}/{epochs}', leave=True)
        
        with torch.no_grad():
            for x, y in val_progress_bar:
                x, y = x.to(device), y.float().unsqueeze(1).to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                val_loss += loss.item() * x.size(0)

                val_progress_bar.set_postfix(loss=f'{loss.item():.3f}')

        val_epoch_loss = val_loss / len(val_loader.dataset)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), f'../models/{model_name}_best_v1.pt')
        
        scheduler.step(val_epoch_loss)

        writer.add_scalars('Loss', {'Train': train_epoch_loss, 'Val': val_epoch_loss}, epoch+1)
        
        print(f'Train Loss: {train_epoch_loss:.3f} | Val Loss: {val_epoch_loss:.3f}')
        
        early_stopping(val_epoch_loss)
        
        if early_stopping.early_stop:
            print('Early Stopping triggered...')
            print(f'Best Val Loss: {early_stopping.best_loss}')
            break

    writer.close()

#-------------------------------------------------------------------------------------------------------------

def load_model(model_name:str, model_path:str='../models/HgbCNN_best_v1.pt'):
    import torchvision.models as models

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name=='HgbCNN':
        model_state_dict = torch.load(model_path, map_location=device)
        model = HgbCNN()
        model.load_state_dict(model_state_dict)

        return model.to(device)
        
    elif model_name=='ResNet18': 
        model_state_dict = torch.load(model_path, map_location=device)              
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        for param in model.parameters():
            param.requires_grad = False

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Linear(in_features, out_features=256), 
            nn.ReLU(),  
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )

        model.load_state_dict(model_state_dict)
        
        return model.to(device)
    
    elif model_name=='ConvNeXt': 
        model_state_dict = torch.load(model_path, map_location=device)

        model = models.convnext_tiny(weights='IMAGENET1K_V1')

        for param in model.parameters():
            param.requires_grad = False

        model.classifier[2] = nn.Sequential(
            nn.Linear(768, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        model.load_state_dict(model_state_dict)
        
        return model.to(device)

    elif model_name=='UNet': 
        model_state_dict = torch.load(model_path, map_location=device)
        model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", 
                         in_channels=3, classes=1,  # Binary segmentation
                         activation=None  # Output will be logits
                         )
        model.load_state_dict(model_state_dict)
        
        return model.to(device)
    
#-----------------------------------------------------------------------------------------------------





