import cv2
import matplotlib.pyplot as plt
import os

# Đường dẫn ảnh mẫu sau khi đã resize
img_path = r'd:\SkinCancer_AI_ISIC\data_processed\resized_224\ISIC_0000015.jpg' # Thay bằng tên file có thật
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Áp dụng Gaussian Blur (Kernel size 3x3 hoặc 5x5)
gaussian_img = cv2.GaussianBlur(image_rgb, (3, 3), 0)

# 2. Áp dụng Median Filter (Kernel size 3 hoặc 5)
median_img = cv2.medianBlur(image_rgb, 3)

# Hiển thị so sánh
titles = ['Gốc', 'Gaussian Blur', 'Median Filter']
images = [image_rgb, gaussian_img, median_img]

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.show()