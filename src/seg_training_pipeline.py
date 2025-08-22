# train data
train_coco_json_path = '../data/Conjunctiva/Segmentation/Conjunctiva_Segmentation.v5i.coco/train/_annotations.coco.json'
train_images_dir = '../data/Conjunctiva/Segmentation/Conjunctiva_Segmentation.v5i.coco/train'
train_masks_dir = '../data/Conjunctiva/Segmentation/train_segmentation_masks'

# valid data
valid_coco_json_path = '../data/Conjunctiva/Segmentation/Conjunctiva_Segmentation.v5i.coco/valid/_annotations.coco.json'
valid_images_dir = '../data/Conjunctiva/Segmentation/Conjunctiva_Segmentation.v5i.coco/valid'
valid_masks_dir = '../data/Conjunctiva/Segmentation/valid_segmentation_masks'

# test data
test_coco_json_path = '../data/Conjunctiva/Segmentation/Conjunctiva_Segmentation.v5i.coco/test/_annotations.coco.json'
test_images_dir = '../data/Conjunctiva/Segmentation/Conjunctiva_Segmentation.v5i.coco/test'
test_masks_dir = '../data/Conjunctiva/Segmentation/test_segmentation_masks'

# convert_coco_to_masks(test_coco_json_path, test_images_dir, test_masks_dir)

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-15, 15), p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.1),
    A.GaussianBlur(blur_limit=(3, 3), p=0.05),
    A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

from torch.utils.data import DataLoader
from utils import ConjunctivaSegDataset

train_dataset = ConjunctivaSegDataset(
    images_dir='../data/Conjunctiva/Segmentation/Conjunctiva_Segmentation.v5i.coco/train',
    masks_dir='../data/Conjunctiva/Segmentation/train_segmentation_masks',
    transform=train_transforms
)

val_dataset = ConjunctivaSegDataset(
    images_dir='../data/Conjunctiva/Segmentation/Conjunctiva_Segmentation.v5i.coco/valid',
    masks_dir='../data/Conjunctiva/Segmentation/valid_segmentation_masks',
    transform=val_transforms
)

test_dataset = ConjunctivaSegDataset(
    images_dir='../data/Conjunctiva/Segmentation/Conjunctiva_Segmentation.v5i.coco/test',
    masks_dir='../data/Conjunctiva/Segmentation/test_segmentation_masks',
    transform=test_transforms
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)


import segmentation_models_pytorch as smp
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,  # Binary segmentation
    activation=None  # Output will be logits
)

# trained_model = train_model(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     epochs=20,
#     lr=1e-4
# )

## Save model
# torch.save(trained_model.state_dict(), 'conjunctiva_segmentation_unet.pth')


