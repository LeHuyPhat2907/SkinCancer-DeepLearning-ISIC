import os
from PIL import Image, ImageDraw

# 1. Khai báo chính xác đường dẫn chứa ảnh của bạn
folder_path = r"d:\SkinCancer_AI_ISIC\image_paths"

# Danh sách 6 file ảnh cần gộp
image_names = [
    "baselineCNN_curves.PNG",
    "efficientnet_pure_learning_curves.png",
    "pure_vit_learning_curves.png",
    "pure_deit_learning_curves.png",
    "cbam_pure_learning_curves.png",
    "hybrid_learning_curves.png",
]

# 2. Thiết lập thông số khung ảnh lưới (3 hàng x 2 cột)
target_width = 1200
target_height = 500
cols = 2
rows = 3

grid_width = cols * target_width
grid_height = rows * target_height

# Tạo một phông nền trắng lớn
grid_image = Image.new("RGB", (grid_width, grid_height), color="white")
draw = ImageDraw.Draw(grid_image)

print(f"🤖 Đang quét thư mục: {folder_path} ...\n" + "-" * 50)

# 3. Tiến hành đọc và xử lý ảnh
for index, name in enumerate(image_names):
    # Kết hợp đường dẫn thư mục với tên file
    full_path = os.path.join(folder_path, name)

    # Tính toán tọa độ đặt ảnh
    col_idx = index % cols
    row_idx = index // cols
    x_offset = col_idx * target_width
    y_offset = row_idx * target_height

    if os.path.exists(full_path):
        # Mở ảnh, resize về chuẩn chung để tránh bị lệch khung
        img = Image.open(full_path).convert("RGB")
        img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Dán vào vị trí tương ứng trên ảnh tổng
        grid_image.paste(img_resized, (x_offset, y_offset))
        print(f"✅ Đã tìm thấy và xử lý thành công: {name}")
    else:
        # Nếu thiếu ảnh, tạo một ô xám thế chỗ để giữ đúng kết cấu lưới
        draw.rectangle(
            [x_offset, y_offset, x_offset + target_width, y_offset + target_height],
            fill="#EAEAEA",
        )
        print(f"❌ KHÔNG TÌM THẤY file: {name} (Vui lòng kiểm tra lại tên file)")

    # Vẽ đường viền xám mỏng quanh mỗi ô cho ngay ngắn
    draw.rectangle(
        [x_offset, y_offset, x_offset + target_width, y_offset + target_height],
        outline="#D3D3D3",
        width=3,
    )

# 4. Lưu ảnh tổng hợp chất lượng cao (300 DPI)
output_name = "comparison_learning_curves_grid.png"
output_full_path = os.path.join(folder_path, output_name)
grid_image.save(output_full_path, dpi=(300, 300))

print("-" * 50)
print(f"🎉 HOÀN THÀNH! Ảnh so sánh dạng lưới đã được xuất ra.")
print(f"📍 Đường dẫn file: {output_full_path}")