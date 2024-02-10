import numpy as np
import cv2

#
# img = cv2.imread('test_images/test_img.jpg', cv2.IMREAD_GRAYSCALE)
# kernel = np.array([2, 2, -2, -2]) * np.ones((4, 1))
# conv = cv2.filter2D(src=img, ddepth=-1, kernel=-kernel)
# cv2.imshow('c', conv)
# cv2.waitKey(0)
from sklearn.preprocessing import minmax_scale

print(minmax_scale([0, 0, 1]))