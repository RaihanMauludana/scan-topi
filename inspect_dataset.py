import os

def inspect_dataset():
    base_path = os.path.join("archive", "face-attributes-grouped")
    if not os.path.exists(base_path):
        print(f"Error: Folder '{base_path}' tidak ditemukan.")
        print("Pastikan folder 'archive' sudah diekstrak di workspace ini.")
        return

    splits = ["train", "val", "test"]
    
    # Kategori yang diidentifikasi sebagai memakai topi (Class 1)
    wearing_hat_dirs = [
        os.path.join("headwear", "headtop"),
        os.path.join("headwear", "helmet"),
        os.path.join("headwear", "hoodie")
    ]
    
    # Kategori yang diidentifikasi sebagai tidak memakai topi (Class 0)
    not_wearing_hat_dirs = [
        os.path.join("headwear", "no_headwear"),
        os.path.join("nowear", "plain"),
        os.path.join("nowear", "facialhair"),
        os.path.join("nowear", "facemarks"),
        os.path.join("nowear", "facepaint")
    ]

    print("="*60)
    print("INSPEKSI DATASET: DETEKSI TOPI")
    print("="*60)
    
    for split in splits:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            print(f"Folder split '{split}' tidak ditemukan.")
            continue
            
        wearing_count = 0
        not_wearing_count = 0
        
        # Hitung kelas 1 (Pakai Topi)
        for sub_dir in wearing_hat_dirs:
            full_path = os.path.join(split_path, sub_dir)
            if os.path.exists(full_path):
                files = [f for f in os.listdir(full_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                wearing_count += len(files)
                
        # Hitung kelas 0 (Tidak Pakai Topi)
        for sub_dir in not_wearing_hat_dirs:
            full_path = os.path.join(split_path, sub_dir)
            if os.path.exists(full_path):
                files = [f for f in os.listdir(full_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                not_wearing_count += len(files)
                
        print(f"Split: {split.upper()}")
        print(f"  - Pakai Topi (Class 1)       : {wearing_count} gambar")
        print(f"  - Tidak Pakai Topi (Class 0) : {not_wearing_count} gambar")
        print(f"  - Total Gambar               : {wearing_count + not_wearing_count} gambar")
        print("-"*60)

if __name__ == "__main__":
    inspect_dataset()
