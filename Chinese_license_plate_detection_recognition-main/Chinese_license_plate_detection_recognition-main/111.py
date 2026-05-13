# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# 颜色定义（BGR格式）
COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

def preprocess_image(img):
    """图像预处理"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义车牌颜色范围（蓝色示例）
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return mask

def find_plate_contours(mask, min_area=500, max_area=5000):
    """查找候选车牌轮廓"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
            
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4:
            candidates.append(cnt)
    
    return sorted(candidates, key=cv2.contourArea, reverse=True)[:3]

def order_points(pts):
    """将四个点按左上、右上、右下、左下顺序排列"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """透视变换矫正车牌"""
    rect = order_points(pts.astype('float32'))
    (tl, tr, br, bl) = rect
    
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))

def segment_characters(plate):
    """字符分割（简单阈值法）"""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    return [cv2.bitwise_not(thresh, None, mask=cv2.drawContours(thresh.copy(), [c], -1, 255, -1)) for c in chars]

def recognize_characters(chars):
    """简单字符识别（需自定义模板或使用OCR）"""
    template = {
        '沪': '沪', 'A': 'A', '1': '1', '2': '2', '3': '3',
        '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '0': '0'
    }
    return ''.join([template.get(chr(65 + i%26), '?') for i, _ in enumerate(chars)])

def visualize_results(image, plate_rect, warped_plate, chars):
    """可视化结果"""
    x,y,w,h = cv2.boundingRect(plate_rect)
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
    
    h_offset = 20
    image[h_offset:h_offset+warped_plate.shape[0], 10:10+warped_plate.shape[1]] = warped_plate
    
    y_offset = h_offset + warped_plate.shape[0] + 20
    for i, char in enumerate(chars):
        x_pos = 10 + i*30
        image[y_offset:y_offset+30, x_pos:x_pos+30] = char
    
    return image

if __name__ == "__main__":
    # 直接在代码中设置路径
    input_path = r"E:\yolov5\process_images\example\0475-68_277-140&493_368&683-368&570_162&683_140&590_350&493-0_0_5_25_29_32_32_30-170-444.jpg"   # 输入图片路径
    output_path = r"E:\yolov5\process_images\crop"  # 输出结果路径

    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {input_path}")

    # 预处理
    mask = preprocess_image(img)
    
    # 查找车牌候选
    candidates = find_plate_contours(mask)
    if not candidates:
        raise ValueError("未检测到车牌")
    
    # 选择最大候选
    plate_rect = max(candidates, key=cv2.contourArea)
    
    # 透视变换
    warped = four_point_transform(img, cv2.approxPolyDP(plate_rect, 0.02*cv2.arcLength(plate_rect, True), True))
    
    # 字符分割
    chars = segment_characters(warped)
    
    # 字符识别（需替换为实际OCR）
    plate_number = recognize_characters(chars)
    
    # 可视化结果
    output = visualize_results(img.copy(), plate_rect, warped, chars)
    cv2.putText(output, f"识别结果: {plate_number}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imwrite(output_path, output)
    print(f"结果已保存至: {output_path}")
    print(f"识别车牌号: {plate_number}")