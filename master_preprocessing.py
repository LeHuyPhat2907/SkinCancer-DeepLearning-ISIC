import os
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

def process_one_image(img_name):
    try:
        input_path = os.path.join(r'd:\SkinCancer_AI_ISIC\data_refined', img_name)
        output_path = os.path.join(r'd:\SkinCancer_AI_ISIC\data_processed\final_cleaned', img_name)
        
        img = cv2.imread(input_path)
        if img is None: return

        # STEP 1: Resize 224x224
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        # STEP 2: Median Filter (Denoising)
        img = cv2.medianBlur(img, 3)

        # STEP 3: DullRazor (Hair Removal)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        img = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)

        # STEP 4: CLAHE (Contrast Enhancement)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.merge((l, a, b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

        cv2.imwrite(output_path, img)
    except Exception as e:
        print(f"Error {img_name}: {e}")

if __name__ == '__main__':
    images = os.listdir(r'd:\SkinCancer_AI_ISIC\data_refined')
    os.makedirs(r'd:\SkinCancer_AI_ISIC\data_processed\final_cleaned', exist_ok=True)
    
    print("🚀 Đang bắt đầu xử lý tổng hợp 51,620 ảnh. Hãy pha một ly cà phê nhé!")
    with Pool(os.cpu_count()) as p:
        list(tqdm(p.imap(process_one_image, images), total=len(images)))
    print("✅ HOÀN TẤT! Dữ liệu sạch đã sẵn sàng tại thư mục final_cleaned.")