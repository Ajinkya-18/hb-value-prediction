import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils import load_model, val_transforms, segment_and_crop


@st.cache_resource
def get_model(model_name: str, model_path:str): 
    model = load_model(model_name, model_path)
    model.eval()

    return model.to(device)

#--------------------------------------------------------------------------------------------------------------

def predict(image, model, transforms):
    img_tensor = transforms(image).unsqueeze(0).to(device)

    with torch.no_grad(): 
        output = model(img_tensor)

    return output.item()

#--------------------------------------------------------------------------------------------------------------------------

st.title('Hemoglobin Predictor (Conjunctiva Image-based)')

uploaded_file = st.file_uploader('Upload a conjunctiva image', type=['jpg', 'jpeg', 'png'])

model_choice = st.selectbox('Choose Model', ['HgbCNN', 'ResNet18', 'ConvNeXt'])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'Segmentation')
UNET_PATH = os.path.join(MODEL_DIR, 'conjunctiva_segmentation_unet.pth')

model_paths = {
    'HgbCNN': os.path.join(BASE_DIR,'models/HgbCNN_best_v1.pt'), 
    'ResNet18': os.path.join(BASE_DIR,'models/ResNet18_best_v1.pt'), 
    'ConvNeXt': os.path.join(BASE_DIR,'models/ConvNeXt_best_v1.pt')
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    unet = load_model('UNet', UNET_PATH)

    cropped_img, mask = segment_and_crop(image, unet, device=device)

    st.image(cropped_img, caption='Segmented & Cropped Image', use_container_width=True)

    reg_model = get_model(model_choice, model_paths[model_choice])
    prediction = predict(cropped_img, reg_model, val_transforms)

    st.success(f'Predicted Hemoglobin Level: {prediction:.2f} +/- 1.5 g/dL')





