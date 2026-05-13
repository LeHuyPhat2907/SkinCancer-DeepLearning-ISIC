import numpy as np
import torch

# Số lượng mẫu từ kết quả thống kê của Phát
counts = np.array([9775, 14627, 8624, 0, 1655, 424, 249, 1452, 14814]) # Thứ tự theo label 0-8
# Lưu ý: Lớp số 3 (AK) đang bị 0 ảnh trong thống kê cũ, Phát cần kiểm tra lại file metadata

# Tính trọng số nghịch đảo (Inverse Class Frequency)
weights = 1. / counts
weights = weights / weights.sum() * len(counts) # Normalize về mức trung bình quanh 1.0

class_weights = torch.tensor(weights, dtype=torch.float)
print(f"⚖️ Class Weights tối ưu: \n{class_weights}")