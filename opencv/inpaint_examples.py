import cv2
import numpy as np
import matplotlib.pyplot as plt


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def empty(a):
    pass


path = 'b.jpg'
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 250)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 5, 255, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 161, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 251, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 158, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 243, 255, empty)
# 经过测试得到的掩码58 60 73 255 36 255
while True:
    def pick_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global selectedColor
            selectedColor = color = imgHSV[y, x]
            print(f'Selected HSV color: {color}')


    img = cv2.imread(path)
    # 图像转化为HSV格式，H:色调S:饱和度V:明度
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)

    # 178 209 194,0 247 224
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # 创建一个蒙版，提取需要的颜色为白色，不需要的颜色为白色
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    # mask = cv2.inRange(imgHSV, lower, upper)
    # mask = cv2.inRange(imgHSV, np.array([0, 161, 158]), np.array([10, 251, 243])) + cv2.inRange(imgHSV, np.array([170, 161, 158]), np.array([180, 251, 243]))
    mask1 = cv2.inRange(imgHSV, np.array([0, s_min, v_min]), np.array([10, s_max, v_max]))
    mask2 = cv2.inRange(imgHSV, np.array([170, s_min, v_min]), np.array([180, s_max, v_max]))
    mask = mask1 + mask2
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    # lower_red = np.array([0, 195, 0])
    # upper_red = np.array([5, 255, 140])

    # kernel = np.ones((3, 3), np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    restored_image = cv2.inpaint(img, mask, 5, cv2.INPAINT_NS)

    imgStack = stackImages(0.7, ([img, imgHSV], [mask, restored_image]))
    cv2.imshow("Stacked Images", imgStack)
    # cv2.setMouseCallback('Select Color', pick_color)

    cv2.waitKey(1)
