import cv2
import numpy as np

# 用于选择颜色的回调函数
def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = hsv[y, x]
        print(f'Selected HSV color: {color}')
        global lower_red, upper_red
        lower_red = np.array([max(0, color[0] - 10), 70, 50])
        upper_red = np.array([min(180, color[0] + 10), 255, 255])
        cv2.destroyWindow('Select Color')

# 读取图片
image = cv2.imread('a.jpg')

# 转换为HSV颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 显示图像以手动选择红色
cv2.imshow('Select Color', image)
cv2.setMouseCallback('Select Color', pick_color)
cv2.waitKey(0)

# 创建掩码
mask = cv2.inRange(hsv, lower_red, upper_red)

# 查找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(mask, (x-30, y-30), (x+w+30, y+h+30), (255), thickness=cv2.FILLED)

# 反向掩码，用于保留非红色区域
# mask_inv = cv2.bitwise_not(mask)

# 保留原图中的非红色部分
# result = cv2.bitwise_and(image, image, mask=mask_inv)

# 使用 inpainting 技术修复被遮盖的部分
# inpainted = cv2.inpaint(result, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# 保存结果图片
# cv2.imwrite('output.jpg', inpainted)

# 显示原图和处理后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Processed Image', inpainted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()