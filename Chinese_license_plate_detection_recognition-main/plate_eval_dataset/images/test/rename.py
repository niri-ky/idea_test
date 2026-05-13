import os
import glob

def rename_images_in_order(directory):
    # 支持的图片扩展名
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    
    # 收集所有图片文件路径
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(directory, ext)))
    
    # 按修改时间排序（最早的在最前面）
    all_images.sort(key=os.path.getmtime)  # 也可用 os.path.getctime 按创建时间排序
    
    # 开始重命名
    for idx, old_path in enumerate(all_images, start=1):
        # 获取文件扩展名
        ext = os.path.splitext(old_path)[1]
        
        # 构建新文件名
        new_name = f"{idx}{ext}"
        new_path = os.path.join(directory, new_name)
        
        # 避免覆盖已有文件
        # counter = 1
        # while os.path.exists(new_path):
        #     new_name = f"{idx}_{counter}{ext}"
        #     new_path = os.path.join(directory, new_name)
        #     counter += 1
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名: {os.path.basename(old_path)} -> {new_name}")

# 使用示例
target_directory = r'D:\my_plate_dataset\images\test'  # 替换为你的实际路径
rename_images_in_order(target_directory)