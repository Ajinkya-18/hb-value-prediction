from utils import val_transforms, load_reg_model, RGB2LAB_CLAHE, DataLoader, ConjunctivaDataset, ConjunctivaSegmentationModel
from PIL import Image
import os
import numpy as np
import cv2
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seg_model_state_dict = torch.load('../models/Segmentation\conjunctiva_segmentation_unet.pth', map_location=device)

seg_model = ConjunctivaSegmentationModel().load_state_dict(seg_model_state_dict)


