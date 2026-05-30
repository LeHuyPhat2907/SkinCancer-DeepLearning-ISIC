import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt

# Pipeline hoàn chỉnh cho Training
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    # Brightness/Contrast Augmentation
    # brightness_limit=0.15: thay đổi độ sáng ngẫu nhiên trong khoảng [-15%, +15%]
    # contrast_limit=0.15: thay đổi độ tương phản ngẫu nhiên trong khoảng [-15%, +15%]
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Script test output
def test_lighting_aug(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 5))
    plt.suptitle("Kiểm tra Brightness & Contrast Augmentation")
    for i in range(5):
        augmented = train_transform(image=image)['image']
        aug_img = augmented.permute(1, 2, 0).numpy()
        # Chuẩn hóa lại để hiển thị
        aug_img = (aug_img - aug_img.min()) / (aug_img.max() - aug_img.min())
        
        plt.subplot(1, 5, i+1)
        plt.imshow(aug_img)
        plt.axis('off')
        plt.title(f"Biến thể {i+1}")
    plt.show()

test_lighting_aug(r'd:\SkinCancer_AI_ISIC\data_processed\final_cleaned\ISIC_0000002.jpg')