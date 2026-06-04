import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Cek pustaka yang dibutuhkan
try:
    from PIL import Image
    import torchvision
    import matplotlib
except ImportError:
    print("Pustaka yang diperlukan belum lengkap.")
    print("Silakan jalankan perintah berikut untuk menginstal:")
    print("pip install torch torchvision pillow matplotlib scikit-learn tqdm")
    sys.exit(1)

# Kelas Dataset Kustom untuk Membaca Dataset Wajah
class HatDataset(Dataset):
    def __init__(self, base_path, split="train", transform=None):
        self.split = split
        self.base_path = os.path.join(base_path, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Folder yang dikategorikan sebagai MEMAKAI TOPI (Class 1)
        self.wearing_dirs = [
            os.path.join("headwear", "headtop"),
            os.path.join("headwear", "helmet"),
            os.path.join("headwear", "hoodie")
        ]
        
        # Folder yang dikategorikan sebagai TIDAK MEMAKAI TOPI (Class 0)
        self.not_wearing_dirs = [
            os.path.join("headwear", "no_headwear"),
            os.path.join("nowear", "plain"),
            os.path.join("nowear", "facialhair"),
            os.path.join("nowear", "facemarks"),
            os.path.join("nowear", "facepaint")
        ]
        
        self._load_dataset()
        
    def _load_dataset(self):
        # Memuat gambar kelas 1 (Pakai Topi)
        for sub_dir in self.wearing_dirs:
            full_path = os.path.join(self.base_path, sub_dir)
            if os.path.exists(full_path):
                for f in os.listdir(full_path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(full_path, f))
                        self.labels.append(1)  # Label 1 untuk Pakai Topi
                        
        # Memuat gambar kelas 0 (Tidak Pakai Topi)
        for sub_dir in self.not_wearing_dirs:
            full_path = os.path.join(self.base_path, sub_dir)
            if os.path.exists(full_path):
                for f in os.listdir(full_path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(full_path, f))
                        self.labels.append(0)  # Label 0 untuk Tidak Pakai Topi
                        
        print(f"Split {self.split.upper()} berhasil dimuat:")
        print(f"  - Pakai Topi (Class 1)       : {self.labels.count(1)} gambar")
        print(f"  - Tidak Pakai Topi (Class 0) : {self.labels.count(0)} gambar")
        print(f"  - Total                      : {len(self.image_paths)} gambar")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Jika gambar rusak, gunakan gambar kosong/hitam sebagai fallback
            print(f"Warning: Gagal membuka gambar {img_path}. Menggunakan fallback.")
            image = Image.new("RGB", (128, 128))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Fungsi untuk Melatih Model
def train_model():
    # Parameter dan Konfigurasi (Dioptimalkan untuk CPU agar cepat)
    DATASET_PATH = os.path.join("archive", "face-attributes-grouped")
    BATCH_SIZE = 16  # Dikurangi ke 16 untuk mempercepat kalkulasi per batch di CPU
    EPOCHS = 5  # Dikurangi ke 5 Epoch agar training selesai lebih cepat di CPU
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Menggunakan Device: {DEVICE}")
    print("Optimasi CPU aktif: Menggunakan resolusi gambar 128x128 dengan EfficientNet-B0.")
    
    # 1. Image Transform (Resolusi diturunkan ke 128x128 untuk menghemat beban CPU ~67%)
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Inisialisasi Dataset & DataLoader
    print("Memuat data...")
    train_dataset = HatDataset(DATASET_PATH, split="train", transform=train_transforms)
    val_dataset = HatDataset(DATASET_PATH, split="val", transform=val_test_transforms)
    test_dataset = HatDataset(DATASET_PATH, split="test", transform=val_test_transforms)
    
    if len(train_dataset) == 0:
        print("Error: Dataset train kosong. Pastikan data Anda diletakkan dengan benar di folder archive/face-attributes-grouped/")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 3. Definisikan Model CNN Akurasi Tinggi & Ringan di CPU (EfficientNet-B0)
    print("Mempersiapkan model CNN: EfficientNet-B0 (Pretrained)...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Modifikasi classifier terakhir untuk 2 kelas (Topi / Tidak Topi)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, 2)
    )
    model = model.to(DEVICE)
    
    # 4. Optimizer, Loss Function, dan LR Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
    
    # Simpan riwayat training untuk diplot nanti
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }
    
    best_val_acc = 0.0
    model_save_path = "best_hat_detector.pth"
    
    print("\nMemulai Pelatihan Model (CNN EfficientNet-B0)...")
    print("="*60)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Loop training
        total_batches = len(train_loader)
        print(f"\n--- Memulai Epoch {epoch+1}/{EPOCHS} ---")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Print kemajuan secara berkala tiap 10 batch agar tidak membeku di PowerShell
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                print(f"  Batch {batch_idx+1}/{total_batches} | Loss: {loss.item():.4f} | Batch Acc: {batch_acc*100:.1f}%", flush=True)
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = correct_train / total_train
        
        # Evaluasi pada Validation Set
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = correct_val / total_val
        
        # Update Scheduler
        scheduler.step(epoch_val_loss)
        
        # Simpan metrik
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")
        
        # Simpan model terbaik
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Model terbaik disimpan dengan Akurasi Val: {best_val_acc*100:.2f}%")
            
    print("="*60)
    print("Pelatihan selesai!")
    print(f"Akurasi Validasi Terbaik: {best_val_acc*100:.2f}%")
    print(f"Model disimpan di '{model_save_path}'")
    
    # 5. Evaluasi pada Test Set menggunakan Model Terbaik
    print("\nMelakukan Evaluasi pada Test Set menggunakan Model Terbaik...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    correct_test = 0
    total_test = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            # Hitung Precision & Recall
            for p, l in zip(predicted, labels):
                if p == 1 and l == 1:
                    true_positives += 1
                elif p == 1 and l == 0:
                    false_positives += 1
                elif p == 0 and l == 1:
                    false_negatives += 1
                    
    test_acc = correct_test / total_test
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("-"*60)
    print("HASIL PENGUJIAN AKHIR (TEST SET):")
    print(f"  - Akurasi Test : {test_acc*100:.2f}%")
    print(f"  - Precision    : {precision*100:.2f}%")
    print(f"  - Recall       : {recall*100:.2f}%")  
    print(f"  - F1-Score     : {f1_score*100:.2f}%")
    print("-"*60)
    
    # 6. Plot & Simpan Grafik Pelatihan
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Grafik kurva pelatihan disimpan di 'training_curves.png'")

if __name__ == "__main__":
    train_model()
