import os
import sys
import io
import base64
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_hat_detector.pth"
model = None

def load_model():
    global model
    if not os.path.exists(model_path):
        print(f"Error: Berkas model '{model_path}' tidak ditemukan.")
        print("Silakan jalankan training terlebih dahulu dengan 'python train_model.py'.")
        return False
        
    try:
        # Inisialisasi arsitektur EfficientNet-B0 sesuai training
        model = models.efficientnet_b0()
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 2)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("Model berhasil dimuat.")
        return True
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return False

# Transformasi Gambar (Resolusi 128x128 sesuai training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_hat(image):
    try:
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        classes = ["Tidak Pakai Topi", "Pakai Topi"]
        predicted_class = classes[predicted_idx.item()]
        confidence_score = confidence.item() * 100
        return predicted_class, confidence_score
    except Exception as e:
        print(f"Error saat melakukan prediksi: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model belum dimuat di server."}), 500
        
    try:
        image = None
        
        # Skenario 1: Input berupa file upload (Multipart Form)
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "Tidak ada file yang dipilih."}), 400
            image = Image.open(file.stream).convert('RGB')
            
        # Skenario 2: Input berupa Base64 (Webcam Capture dari JS JSON)
        elif request.json and 'image' in request.json:
            img_data = request.json['image']
            if img_data.startswith('data:'):
                img_data = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
        if image is None:
            return jsonify({"error": "Format input gambar tidak valid."}), 400
            
        # Lakukan prediksi
        predicted_class, confidence = predict_hat(image)
        
        if predicted_class is None:
            return jsonify({"error": "Gagal memproses gambar."}), 500
            
        return jsonify({
            "class": predicted_class,
            "confidence": round(confidence, 2)
        })
        
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan di server: {str(e)}"}), 500

if __name__ == '__main__':
    # Muat model saat startup
    if load_model():
        # Jalankan di port 5000 dan buat agar bisa diakses dari perangkat mana saja dalam satu jaringan (host='0.0.0.0')
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Gagal memulai server karena model gagal dimuat.")
