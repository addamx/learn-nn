import cv2
import numpy as np

# 用于选择颜色的回调函数
def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global selectedColor
        selectedColor = color = hsv[y, x]
        print(f'Selected HSV color: {color}')
        global lower_red, upper_red
        lower_red = np.array([max(0, color[0] - 10), 70, 50])
        upper_red = np.array([min(180, color[0] + 10), 255, 255])
        cv2.destroyWindow('Select Color')

# 读取图片
image = cv2.imread('b.jpg')

# 转换为HSV颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 显示图像以手动选择红色
cv2.imshow('Select Color', image)
cv2.setMouseCallback('Select Color', pick_color)
cv2.waitKey(0)

# 创建掩码
# lower_red = np.array([0, 195, 0])
# upper_red = np.array([5, 255, 140])
#b.jpg
# 178 209 194,0 247 224
mask = cv2.inRange(hsv, lower_red, upper_red)

# # 查找轮廓
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(mask, (x-30, y-30), (x+w+30, y+h+30), (255), thickness=cv2.FILLED)


# 将掩码膨胀以确保完全覆盖目标区域
kernel = np.ones((3,3),np.uint8)
mask = cv2.dilate(mask, kernel, iterations = 1)

# 反转掩码
mask_inv = cv2.bitwise_not(mask)

# 原图像中的指定颜色区域变为黑色
image[mask != 0] = [0, 0, 0]

# 使用 inpainting 方法修复图像
restored_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

# 显示结果
# cv2.imshow('Original Image', image)
cv2.imshow('Restored Image', restored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()