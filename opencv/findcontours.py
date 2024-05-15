import cv2
import numpy as np

# 读取图片
image = cv2.imread('a.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow('gray', gray)
# cv2.waitKey(0)

# 应用二值化处理 对灰度图像进行二值化处理。将灰度值大于 127 的像素设为 255（白色），其余的设为 0（黑色
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)

# 查找轮廓
# cv2.RETR_EXTERNAL：只检索外部轮廓。
# cv2.CHAIN_APPROX_SIMPLE：压缩水平方向、垂直方向和对角线方向的轮廓点，只保留轮廓的终点坐标。
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 画出轮廓
# -1：绘制所有轮廓。
# (0, 255, 0)：轮廓的颜色（绿色）。
# 2：轮廓线条的厚度。
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# 显示结果
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
