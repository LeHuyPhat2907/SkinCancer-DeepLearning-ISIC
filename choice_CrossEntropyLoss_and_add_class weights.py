import torch
import torch.nn as nn

# Mảng trọng số đã tính toán từ EDA (Phát copy lại con số chính xác của bạn nhé)
# Thứ tự tương ứng với nhãn 0-7: [BCC, BKL, DF, MEL, NV, SCC, UNK, VASC] (Ví dụ)
weights = [0.102026, 0.068182, 0.115642, 0.602601, 2.352134, 4.005241, 0.686849, 0.067321]

# Chuyển mảng weights sang Tensor (Máy văn phòng chạy CPU nên để device='cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

# Khởi tạo Loss Function với weights
criterion = nn.CrossEntropyLoss(weight=class_weights)

print("✅ Đã cấu hình Weighted CrossEntropyLoss thành công!")

# Giả lập đầu ra mô hình (batch_size=2, num_classes=8)
dummy_outputs = torch.randn(2, 8) 
# Giả lập nhãn thật (Ví dụ: lớp 5 và lớp 7)
dummy_labels = torch.tensor([5, 7]) 

loss = criterion(dummy_outputs, dummy_labels)
print(f"📊 Giá trị Loss mẫu: {loss.item():.4f}")