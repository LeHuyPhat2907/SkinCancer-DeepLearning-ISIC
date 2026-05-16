import torch

file_path = "models/checkpoint.pth"

checkpoint = torch.load(file_path, map_location="cpu")

print("===== TRAINING RESULT =====")
print(f"Best Epoch     : {checkpoint['epoch'] + 1}")
print(f"Best Accuracy  : {checkpoint['best_val_acc']:.2f}%")
print("===========================")