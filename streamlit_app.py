import os
import torch
import torch.nn as nn
import streamlit as st

from PIL import Image
from torchvision import transforms, models

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "best_hat_detector.pth"

@st.cache_resource
def load_model():

    model = models.efficientnet_b0()

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 2)
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

classes = [
    "Tidak Pakai Topi",
    "Pakai Topi"
]

st.title("🧢 Deteksi Topi")

uploaded_file = st.file_uploader(
    "Upload Foto",
    type=["jpg","jpeg","png"]
)

camera_image = st.camera_input(
    "Atau Ambil Foto"
)

image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

elif camera_image:
    image = Image.open(camera_image).convert("RGB")

if image:

    st.image(image)

    tensor = transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_idx = torch.max(
            probs,
            1
        )

    prediction = classes[
        predicted_idx.item()
    ]

    st.success(
        prediction
    )

    st.metric(
        "Confidence",
        f"{confidence.item()*100:.2f}%"
    )