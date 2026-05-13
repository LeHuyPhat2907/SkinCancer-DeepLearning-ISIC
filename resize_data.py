import os
import cv2
from multiprocessing import Pool
from tqdm import tqdm

# Cấu hình
input_dir = r'd:\SkinCancer_AI_ISIC\data_refined'
output_dir = r'd:\SkinCancer_AI_ISIC\data_processed\resized_224'
size = (224, 224)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def resize_image(image_name):
    try:
        img_path = os.path.join(input_dir, image_name)
        img = cv2.imread(img_path)
        if img is not None:
            # Dùng INTER_AREA cho kết quả tốt nhất khi thu nhỏ ảnh y khoa
            res = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(output_dir, image_name), res)
    except Exception as e:
        print(f"Lỗi file {image_name}: {e}")

if __name__ == '__main__':
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"🚀 Đang Resize {len(images)} ảnh về {size}...")
    
    # Sử dụng đa nhân CPU để tăng tốc
    with Pool(os.cpu_count()) as p:
        list(tqdm(p.imap(resize_image, images), total=len(images)))

    print(f"✅ Hoàn tất! Ảnh đã được lưu tại: {output_dir}")