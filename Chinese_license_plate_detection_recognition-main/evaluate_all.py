import argparse
import os
import time
import pandas as pd
from tqdm import tqdm
import torch
import cv2
from utils.cv_puttext import cv2ImgAddText
from detect_plate import load_model, detect_Recognition_plate, cv_imread, draw_result
from plate_recognition.plate_rec import init_model
import numpy as np # 用于创建预热图像

def evaluate(opt):
    # 1. 初始化模型并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    print("正在加载模型...")
    detect_model = load_model(opt.detect_model, device)
    rec_model = init_model(device, opt.rec_model, is_color=True)
    print("模型加载完成。")
    
    # ！！！新增：在加载模型后重置显存计数器！！！
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")

    # 2. 加载标准答案
    # ... (这部分代码和之前一样) ...
    try:
        gt_df = pd.read_csv(opt.gt_path, encoding='utf-8')
        gt_dict = {row['filename']: row for _, row in gt_df.iterrows()}
    except FileNotFoundError:
        print(f"错误：找不到标准答案文件 {opt.gt_path}")
        return
    image_filenames = list(gt_dict.keys())
    print(f"找到 {len(image_filenames)} 张待评估的图片。")
    
    # ！！！新增：GPU预热！！！
    if use_cuda:
        print("正在进行GPU预热...")
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            _ = detect_Recognition_plate(detect_model, dummy_img, device, rec_model, opt.img_size, is_color=True)
        print("预热完成。")

    # 3. 初始化所有变量
    results_data = []
    total_count = 0
    correct_all_tasks = 0
    detection_failures = 0
    total_inference_time = 0.0 # 新增：用于累计总推理时间
    type_mapping = {0: '单层', 1: '双层'}

    # 4. 循环处理每张图片
    
    # ！！！新增：为 torch.cuda.Event 计时器做准备！！！
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for filename in tqdm(image_filenames, desc="正在评估"):
        image_path = os.path.join(opt.image_path, filename)
        if not os.path.exists(image_path): continue
        img = cv_imread(image_path)
        if img is None: continue
        
        img_to_draw = img.copy()
        
        # ！！！核心改动：使用新的计时方法！！！
        start_event.record()
        
        # 核心推理步骤 (保持不变)
        predictions = detect_Recognition_plate(detect_model, img, device, rec_model, opt.img_size, is_color=True)
        
        end_event.record()
        torch.cuda.synchronize() # 等待事件完成
        
        # 累加时间，单位是毫秒(ms)，所以最后要除以1000
        total_inference_time += start_event.elapsed_time(end_event)
        total_count += 1
        
        # 结果比较和保存部分 (保持不变)
        gt = gt_dict[filename]
        current_result = { 'filename': filename, 'ground_truth_plate_no': gt['plate_number'], 'predicted_plate_no': '检测失败',
                           'ground_truth_type': gt['plate_type'], 'predicted_type': '检测失败',
                           'ground_truth_color': gt['plate_color'], 'predicted_color': '检测失败', 'is_correct': False }
        if len(predictions) == 1:
            pred = predictions[0]
            current_result.update({ 'predicted_plate_no': pred['plate_no'], 'predicted_type': type_mapping.get(pred['plate_type'], '未知'), 'predicted_color': pred['plate_color'] })
            is_correct = (current_result['predicted_plate_no'] == gt['plate_number'] and current_result['predicted_type'] == gt['plate_type'] and current_result['predicted_color'] == gt['plate_color'])
            if is_correct:
                correct_all_tasks += 1
                current_result['is_correct'] = True
            ori_img_with_result = draw_result(img_to_draw, predictions)
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, ori_img_with_result)
        else:
            detection_failures += 1
            fail_text = f"Detection Failed: Found {len(predictions)} targets"
            ori_img_with_result = cv2ImgAddText(img_to_draw, fail_text, 20, 20, (0, 0, 255), 40)
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, ori_img_with_result)
        results_data.append(current_result)

    # 5. 保存CSV结果
    # ... (这部分代码和之前一样) ...
    results_df = pd.DataFrame(results_data)
    output_csv_path = 'evaluation_results.csv'
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n详细对比结果已保存至: {output_csv_path}")

    # 6. 计算并打印最终的评估报告
    print("\n" + "="*50)
    print("          综合性能评估报告")
    print("="*50)
    
    # ！！！新增：健壮性检查！！！
    # 检查是否有任何图片被成功处理了
    if total_count == 0:
        print("错误：没有找到或处理任何有效的图片，无法生成评估报告。")
        print("请检查以下几点：")
        print(f"  1. --image_path 指定的文件夹 '{opt.image_path}' 是否正确，并且里面有图片。")
        print(f"  2. --gt_path 指定的CSV文件 '{opt.gt_path}' 是否存在。")
        print("  3. CSV文件中的 'filename' 列是否与图片文件夹中的实际文件名匹配。")
        return # 提前退出函数

    # --- 准确率指标 ---
    print(f"【准确率指标】")
    print(f"总计评估图片: {total_count} 张")
    print(f"检测失败或检测到多个目标的图片: {detection_failures} 张")
    
    successful_predictions = results_df[results_df['predicted_plate_no'] != '检测失败']
    acc_no = (successful_predictions['ground_truth_plate_no'] == successful_predictions['predicted_plate_no']).sum() / total_count * 100
    acc_type = (successful_predictions['ground_truth_type'] == successful_predictions['predicted_type']).sum() / total_count * 100
    acc_color = (successful_predictions['ground_truth_color'] == successful_predictions['predicted_color']).sum() / total_count * 100
    acc_all = (correct_all_tasks / total_count) * 100
    print(f"- 车牌号码识别准确率: {acc_no:.2f}%")
    print(f"- 车牌类型分类准确率: {acc_type:.2f}%")
    print(f"- 车牌颜色分类准确率: {acc_color:.2f}%")
    print(f"- 综合完全正确率: {acc_all:.2f}%")
    print("-" * 50)
    
    # --- 效率与资源指标 ---
    print(f"【效率与资源指标】")
    if total_count > 0:
        # total_inference_time 的单位已经是毫秒了，所以先除以1000变回秒
        avg_time_per_image_sec = (total_inference_time / 1000.0) / total_count
        fps = 1.0 / avg_time_per_image_sec
        print(f"- 平均单张图片处理耗时: {avg_time_per_image_sec * 1000:.2f} ms")
        print(f"- 模型吞吐率 (FPS): {fps:.2f} 帧/秒")
    else:
        print("- 平均单张图片处理耗时: N/A")
        print("- 模型吞吐率 (FPS): N/A")

    if use_cuda:
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"- 峰值显存占用 (VRAM): {peak_memory_mb:.2f} MB")
    else:
        print("- 显存占用: N/A (当前使用CPU)")

    print("="*50)

if __name__ == '__main__':
    # ... (参数部分和之前一样) ...
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', type=str, default='weights/plate_detect.pt', help='检测模型路径')
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth', help='识别模型路径')
    parser.add_argument('--image_path', type=str, required=True, help='测试图片文件夹路径')
    parser.add_argument('--gt_path', type=str, required=True, help='标准答案CSV文件路径')
    parser.add_argument('--img_size', type=int, default=640, help='网络输入图片大小')
    opt = parser.parse_args()
    evaluate(opt)