import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# Cek pustaka OpenCV (cv2)
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

class HatDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Scan Topi - Live Camera CNN")
        self.root.geometry("700x750")
        self.root.configure(bg="#1e1e2e")  # Tema dark mode modern
        
        # Load Model
        self.model_path = "best_hat_detector.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        # Inisialisasi Kamera
        self.cap = None
        self.camera_active = False
        
        # Inisialisasi UI
        self._create_widgets()
        
        # Tampilkan peringatan jika OpenCV belum diinstal
        if not OPENCV_AVAILABLE:
            messagebox.showwarning(
                "Pustaka OpenCV Belum Terinstal",
                "OpenCV tidak terdeteksi. Fitur kamera tidak dapat digunakan.\n"
                "Silakan jalankan perintah ini di terminal Anda:\n\n"
                "pip install opencv-python"
            )
        else:
            # Otomatis nyalakan kamera jika tersedia
            self.toggle_camera()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror(
                "Model Tidak Ditemukan",
                f"Berkas '{self.model_path}' tidak ditemukan.\nPastikan Anda sudah melatih model terlebih dahulu."
            )
            sys.exit(1)
            
        try:
            # Struktur arsitektur EfficientNet-B0 sesuai dengan training
            model = models.efficientnet_b0()
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(num_features, 2)
            )
            
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            messagebox.showerror("Error Load Model", f"Gagal memuat model: {e}")
            sys.exit(1)

    def _create_widgets(self):
        # Header Label
        header = tk.Label(
            self.root, 
            text="APLIKASI SCAN TOPI (LIVE CAMERA)", 
            font=("Helvetica", 18, "bold"), 
            fg="#cdd6f4", 
            bg="#1e1e2e",
            pady=15
        )
        header.pack()
        
        # Panel Live Preview Kamera / Gambar
        self.image_label = tk.Label(
            self.root, 
            text="Kamera Mati\n\n(Silakan klik 'Nyalakan Kamera' atau 'Pilih Gambar')",
            font=("Helvetica", 12),
            fg="#a6adc8",
            bg="#313244", 
            width=480,
            height=360,
            bd=2,
            relief="solid",
            highlightthickness=0
        )
        self.image_label.pack(pady=15)
        
        # Label Hasil Prediksi
        self.result_label = tk.Label(
            self.root, 
            text="Hasil Prediksi: -", 
            font=("Helvetica", 16, "bold"), 
            fg="#bac2de", 
            bg="#1e1e2e"
        )
        self.result_label.pack(pady=10)
        
        # Label Keyakinan (Confidence)
        self.confidence_label = tk.Label(
            self.root, 
            text="Keyakinan: -", 
            font=("Helvetica", 12), 
            fg="#a6adc8", 
            bg="#1e1e2e"
        )
        self.confidence_label.pack(pady=5)
        
        # Frame Tombol Utama
        btn_frame = tk.Frame(self.root, bg="#1e1e2e")
        btn_frame.pack(pady=20)
        
        # Tombol Nyalakan/Matikan Kamera
        self.btn_camera_toggle = tk.Button(
            btn_frame, 
            text="Nyalakan Kamera", 
            command=self.toggle_camera,
            font=("Helvetica", 11, "bold"),
            bg="#f9e2af", 
            fg="#11111b",
            activebackground="#fae3b0",
            padx=15, 
            pady=8,
            bd=0,
            cursor="hand2"
        )
        self.btn_camera_toggle.grid(row=0, column=0, padx=8)
        
        # Tombol Ambil Foto & Scan
        self.btn_capture = tk.Button(
            btn_frame, 
            text="Ambil Foto & Scan", 
            command=self.capture_and_predict,
            font=("Helvetica", 11, "bold"),
            bg="#89b4fa", 
            fg="#11111b",
            activebackground="#b4befe",
            padx=15, 
            pady=8,
            bd=0,
            state="disabled",  # Hanya aktif jika kamera menyala
            cursor="hand2"
        )
        self.btn_capture.grid(row=0, column=1, padx=8)
        
        # Tombol Pilih File Gambar
        btn_browse = tk.Button(
            btn_frame, 
            text="Pilih Gambar dari File", 
            command=self.browse_image,
            font=("Helvetica", 11, "bold"),
            bg="#a6e3a1", 
            fg="#11111b",
            activebackground="#94e2d5",
            padx=15, 
            pady=8,
            bd=0,
            cursor="hand2"
        )
        btn_browse.grid(row=0, column=2, padx=8)
        
        # Tombol Keluar
        btn_exit = tk.Button(
            self.root, 
            text="Keluar Aplikasi", 
            command=self.close_app,
            font=("Helvetica", 10, "bold"),
            bg="#f38ba8", 
            fg="#11111b",
            activebackground="#eba0b2",
            padx=20, 
            pady=6,
            bd=0,
            cursor="hand2"
        )
        btn_exit.pack(pady=10)

    def toggle_camera(self):
        if not OPENCV_AVAILABLE:
            messagebox.showerror("Error Kamera", "OpenCV tidak tersedia. Jalankan: pip install opencv-python")
            return
            
        if self.camera_active:
            # Matikan Kamera
            self.camera_active = False
            self.btn_capture.config(state="disabled")
            self.btn_camera_toggle.config(text="Nyalakan Kamera", bg="#f9e2af")
            if self.cap:
                self.cap.release()
                self.cap = None
            self.image_label.config(text="Kamera Mati\n\n(Silakan klik 'Nyalakan Kamera' atau 'Pilih Gambar')", image="")
        else:
            # Nyalakan Kamera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error Kamera", "Gagal membuka kamera. Pastikan kamera terhubung dan tidak sedang digunakan oleh aplikasi lain.")
                self.cap = None
                return
                
            self.camera_active = True
            self.btn_capture.config(state="normal")
            self.btn_camera_toggle.config(text="Matikan Kamera", bg="#f38ba8")
            self.update_frame()

    def update_frame(self):
        if self.camera_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Balik gambar secara horizontal (mirror mode) agar lebih natural
                frame = cv2.flip(frame, 1)
                
                # Simpan frame terakhir untuk kebutuhan pengambilan foto
                self.current_frame = frame.copy()
                
                # Konversi BGR (OpenCV) ke RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                
                # Resize agar muat di preview GUI
                img = img.resize((480, 360), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image=img)
                
                # Tampilkan di label
                self.image_label.config(image=self.photo, text="")
                
            # Loop pembaruan frame setiap 10ms untuk mendapatkan video feed yang lancar
            self.root.after(10, self.update_frame)

    def capture_and_predict(self):
        if self.camera_active and hasattr(self, 'current_frame'):
            # Matikan kamera sementara agar gambar yang difoto freeze (membeku) di layar
            self.camera_active = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.btn_capture.config(state="disabled")
            self.btn_camera_toggle.config(text="Nyalakan Kamera", bg="#f9e2af")
            
            # Simpan foto hasil capture secara temporer ke disk
            temp_path = "temp_capture.jpg"
            cv2.imwrite(temp_path, self.current_frame)
            
            # Jalankan prediksi
            self.predict(temp_path)
            
            # Hapus file sementara setelah diproses
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def browse_image(self):
        # Matikan kamera jika menyala saat pengguna memilih gambar dari file
        if self.camera_active:
            self.toggle_camera()
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
            
        try:
            # Tampilkan gambar di GUI
            img = Image.open(file_path)
            img.thumbnail((480, 360))
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo, text="")
            
            # Jalankan prediksi
            self.predict(file_path)
            
        except Exception as e:
            messagebox.showerror("Error Gambar", f"Gagal membuka gambar: {e}")

    def predict(self, img_path):
        # 1. Transformasi Gambar (Sama dengan saat training)
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # 2. Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
            classes = ["Tidak Pakai Topi", "Pakai Topi"]
            predicted_class = classes[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            # 3. Update Tampilan Hasil
            self.result_label.config(text=f"Hasil Prediksi: {predicted_class}")
            self.confidence_label.config(text=f"Keyakinan: {confidence_score:.2f}%")
            
            # Ubah warna teks berdasarkan hasil
            if predicted_idx.item() == 1:
                self.result_label.config(fg="#a6e3a1") # Hijau pastel jika pakai topi
            else:
                self.result_label.config(fg="#f38ba8") # Merah pastel jika tidak pakai topi
                
        except Exception as e:
            messagebox.showerror("Error Prediksi", f"Gagal menganalisis gambar: {e}")

    def close_app(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = HatDetectorApp(root)
    root.mainloop()
