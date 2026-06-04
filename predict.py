import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def predict_image(image_path, model_path="best_hat_detector.pth"):
    # 1. Validasi Input
    if not os.path.exists(image_path):
        print(f"Error: File gambar '{image_path}' tidak ditemukan.")
        return None
        
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan training dengan 'python train_model.py' terlebih dahulu.")
        return None

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Definisikan Struktur Model CNN EfficientNet-B0 (Sama persis dengan saat training)
    model = models.efficientnet_b0()
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, 2)
    )
    
    # Load Model Weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error saat memuat model weights: {e}")
        return None

    # 3. Definisikan Transformasi Gambar (Sama dengan validation transform, resolusi 128x128)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Load & Preprocess Gambar
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device) # Tambahkan batch dimension
    except Exception as e:
        print(f"Error saat membaca gambar: {e}")
        return None

    # 5. Inference / Prediksi
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    classes = ["Tidak Pakai Topi", "Pakai Topi"]
    predicted_class = classes[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    print("="*50)
    print("HASIL PREDIKSI DETEKSI TOPI (CNN EfficientNet-B0)")
    print("="*50)
    print(f"File Gambar   : {image_path}")
    print(f"Hasil Prediksi: {predicted_class}")
    print(f"Keyakinan     : {confidence_score:.2f}%")
    print("="*50)
    
    return predicted_class, confidence_score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cara penggunaan:")
        print("  python predict.py <path_ke_file_gambar>")
        print("\nContoh:")
        print("  python predict.py test_wajah.jpg")
    else:
        img_path = sys.argv[1]
        predict_image(img_path)
