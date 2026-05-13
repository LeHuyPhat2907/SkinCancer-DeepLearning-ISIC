import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt

# Tích hợp toàn bộ Augmentation cho tập Train
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # Rotation nhẹ: limit=45 nghĩa là xoay ngẫu nhiên từ -45 đến 45 độ
    # border_mode=cv2.BORDER_CONSTANT giúp xử lý vùng trống ở góc khi xoay
    A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Script test nhanh kết quả xoay
def test_rotation(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 5))
    plt.suptitle("Kiểm tra Rotation Augmentation (Random -45 to 45 deg)")
    for i in range(5):
        augmented = train_transform(image=image)['image']
        aug_img = augmented.permute(1, 2, 0).numpy()
        # Scale lại để hiển thị (do Normalize làm ảnh đổi màu)
        aug_img = (aug_img - aug_img.min()) / (aug_img.max() - aug_img.min())
        
        plt.subplot(1, 5, i+1)
        plt.imshow(aug_img)
        plt.axis('off')
    plt.show()

# Chạy thử với ảnh trong dataset của bạn
test_rotation(r'd:\SkinCancer_AI_ISIC\data_processed\final_cleaned\ISIC_0000002.jpg')