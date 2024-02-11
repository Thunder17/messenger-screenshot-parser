import numpy as np
import cv2

img = cv2.medianBlur(cv2.imread('test_images/test_img.jpg'), 1)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

horizontal = np.array([
    [2],
    [2],
    [-2],
    [-2]
]) * np.ones((1, 4))
vertical = np.array([
    2, 2, -2, -2
]) * np.ones((4, 1))

_, conv_h_1 = cv2.threshold(cv2.filter2D(src=gray, ddepth=-1, kernel=horizontal), 127, 255, cv2.THRESH_BINARY)
_, conv_h_2 = cv2.threshold(cv2.filter2D(src=gray, ddepth=-1, kernel=-horizontal), 127, 255, cv2.THRESH_BINARY)
_, conv_v_1 = cv2.threshold(cv2.filter2D(src=gray, ddepth=-1, kernel=vertical), 127, 255, cv2.THRESH_BINARY)
_, conv_v_2 = cv2.threshold(cv2.filter2D(src=gray, ddepth=-1, kernel=-vertical), 127, 255, cv2.THRESH_BINARY)

cv2.imwrite('b.jpg', conv_h_1 + conv_h_2 + conv_v_1 + conv_v_2)