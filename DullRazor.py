import cv2
import numpy as np
import matplotlib.pyplot as plt

def dull_razor(img):
    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2. Black Top-hat để tìm lông
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Tạo mặt nạ lông (Thresholding)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 4. Inpainting (Lấp đầy vùng lông bằng thuật toán Telea)
    result = cv2.inpaint(img, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    
    return result, mask

# Test thực tế
img_path = r'd:\SkinCancer_AI_ISIC\data_processed\resized_224\ISIC_0000229.jpg' # Thay bằng tấm nhiều lông
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

clean_img, hair_mask = dull_razor(img_rgb)

# Hiển thị kết quả
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(img_rgb); plt.title('Ảnh gốc có lông'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(hair_mask, cmap='gray'); plt.title('Mặt nạ lông (Mask)'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(clean_img); plt.title('Kết quả sau DullRazor'); plt.axis('off')
plt.show()