import torch
from models.experimental import attempt_load

def check_model_classes(weights_path):
    """
    加载一个YOLOv5模型并打印出它内部存储的类别名称。
    """
    try:
        # 选择一个设备，CPU即可，因为我们只加载模型结构和元数据
        device = torch.device('cpu')
        
        # 加载模型
        print(f"正在加载模型: {weights_path}")
        # 使用不安全的模式加载，以确保能打开这个旧文件
        model = attempt_load(weights_path, map_location=device)
        print("模型加载成功！")
        
        # 检查模型是否有名为 'names' 的属性
        if hasattr(model, 'names') and model.names:
            print("\n模型内部存储的类别信息如下：")
            class_names = model.names
            for i, name in enumerate(class_names):
                print(f"  - 类别索引 {i}: '{name}'")
        else:
            print("\n警告：在这个模型文件中没有找到明确的类别名称列表。")
            print("这可能是一个较旧的模型，或者类别信息存储在别处。")

    except Exception as e:
        print(f"\n加载或检查模型时发生错误: {e}")

if __name__ == '__main__':
    # ！！！把这里换成你的权重文件路径！！！
    model_weights_path = 'G:\下载\Chinese_license_plate_detection_recognition-main\Chinese_license_plate_detection_recognition-main\weights\plate_detect.pt'
    check_model_classes(model_weights_path)