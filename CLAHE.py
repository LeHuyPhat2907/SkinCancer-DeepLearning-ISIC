import cv2
import matplotlib.pyplot as plt

# Đọc ảnh mẫu
img_path = r'd:\SkinCancer_AI_ISIC\data_processed\resized_224\ISIC_0000043.jpg'
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Chuyển sang hệ màu LAB
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# Áp dụng CLAHE lên kênh L (Lightness)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l)

# Gộp lại và chuyển về RGB
limg = cv2.merge((cl, a, b))
final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

# Hiển thị so sánh
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Trước khi CLAHE')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(final_img)
plt.title('Sau khi CLAHE')
plt.axis('off')
plt.show()