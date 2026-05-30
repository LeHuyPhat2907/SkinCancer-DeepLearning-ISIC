import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Định nghĩa danh sách các nhãn (Labels) theo đúng thứ tự
labels = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# 2. Nhập dữ liệu ma trận nhầm lẫn từ kết quả của bạn
# Mỗi hàng tương ứng với một lớp Thực tế (Actual)
matrix_data = [
    [1567, 167, 43, 39, 6, 1, 15, 117],  # Actual_MEL
    [234, 2462, 58, 26, 3, 1, 5, 136],  # Actual_NV
    [69, 30, 1462, 15, 16, 2, 37, 94],  # Actual_BCC
    [80, 58, 29, 99, 4, 1, 13, 47],  # Actual_AK
    [16, 16, 23, 3, 12, 0, 2, 13],  # Actual_BKL
    [6, 4, 7, 1, 1, 24, 1, 6],  # Actual_DF
    [30, 2, 61, 10, 2, 1, 156, 29],  # Actual_VASC
    [89, 84, 104, 17, 7, 2, 32, 2627],  # Actual_SCC
]

cm = np.array(matrix_data)

# 3. Tạo nhãn văn bản (Text Annotations) kết hợp cả Số lượng và Phần trăm
# Tính tổng theo từng hàng (thực tế) để chia tỷ lệ %
row_sums = cm.sum(axis=1)[:, np.newaxis]
# Tránh chia cho 0 nếu có hàng nào toàn số 0
row_sums[row_sums == 0] = 1
cm_percentage = (cm / row_sums) * 100

annot_labels = np.empty_list = []
blanks = ["" for _ in range(cm.size)]
annot_labels = np.asarray(blanks).reshape(cm.shape[0], cm.shape[1]).astype(object)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        pct = cm_percentage[i, j]
        if count > 0:
            annot_labels[i, j] = f"{count}\n({pct:.1f}%)"
        else:
            annot_labels[i, j] = "0"

# 4. Khởi tạo biểu đồ
plt.figure(figsize=(12, 10))

# Sử dụng màu xanh dương (Blues) hoặc "YlOrRd" (Vàng-Cam-Đỏ) tùy sở thích
sns.heatmap(
    cm,
    annot=annot_labels,
    fmt="",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    square=True,
    cbar_kws={"label": "Số lượng ca bệnh"},
    annot_kws={"size": 10, "weight": "bold"},
)

# 5. Cấu hình tiêu đề và các trục
plt.title(
    "MA TRẬN NHẦM LẪN (CONFUSION MATRIX)\nMÔ HÌNH HYBRID CHÍNH ĐỀ TÀI",
    fontsize=16,
    weight="bold",
    pad=20,
)
plt.xlabel("MÔ HÌNH DỰ ĐOÁN (Predicted)", fontsize=12, weight="bold", labelpad=15)
plt.ylabel("THỰC TẾ (Actual)", fontsize=12, weight="bold", labelpad=15)

# Xoay nhãn trục X để không bị dính vào nhau
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

# 6. Xuất hình ảnh chất lượng cao (DPI=300 thích hợp để in ấn/báo cáo)
plt.savefig("confusion_matrix_hybrid.png", dpi=300, bbox_inches="tight")
print("🎉 Đã xuất ma trận nhầm lẫn thành công ra file 'confusion_matrix_hybrid.png'!")
plt.show()