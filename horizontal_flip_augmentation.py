import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt

# Sub-task 1: Configure horizontal flip
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5), # Xác suất 50% mỗi tấm ảnh sẽ bị lật
    A.VerticalFlip(p=0.5),   # Khuyến khích thêm cả lật dọc vì ảnh y tế rất hợp
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Chuẩn hóa ImageNet
    ToTensorV2()
])

# Sub-task 2: Test augmentation samples
def test_augmentation(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Thử nghiệm lật 5 lần để xem kết quả ngẫu nhiên
    plt.figure(figsize=(15, 5))
    for i in range(5):
        augmented = train_transform(image=image)['image']
        # Chuyển tensor về lại dạng ảnh để hiển thị
        aug_img = augmented.permute(1, 2, 0).numpy()
        plt.subplot(1, 5, i+1)
        plt.imshow(aug_img)
        plt.axis('off')
        plt.title(f"Mẫu {i+1}")
    plt.show()

# Thử nghiệm thực tế với 1 tấm trong final_cleaned
test_augmentation(r'd:\SkinCancer_AI_ISIC\data_processed\final_cleaned\ISIC_0000002.jpg')