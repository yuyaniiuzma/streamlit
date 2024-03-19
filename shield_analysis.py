import cv2
import numpy as np
from matplotlib import pyplot as plt

blocksize = 293
C = 12
filename = r"shieldinput\ML108_240226\ML108_240226.JPG"
img = cv2.imread(filename, 0)
origin_list = [[1126, 741], [2969, 691], [1174, 2116], [2979, 2089]]

trans_list = [[250, 200], [3766, 200], [250, 2926], [3766, 2926]]
p_original = np.float32(origin_list)
p_trans = np.float32(trans_list)
M = cv2.getPerspectiveTransform(p_original, p_trans)
trans = cv2.warpPerspective(img, M, (4032, 3024))

trans_eq = cv2.equalizeHist(trans)
blurred = cv2.medianBlur(trans_eq, 11)

# th = cv2.adaptiveThreshold(
#     blurred,
#     255,
#     cv2.ADAPTIVE_THRESH_MEAN_C,
#     cv2.THRESH_BINARY,
#     blocksize,
#     C - 10,
# ()

# ret, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
dst = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blocksize,
    C - 10,
)

RGB = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
RGB = cv2.line(RGB, pt1=(250, 200), pt2=(3766, 2926), color=(255, 0, 0), thickness=3)
RGB = cv2.line(RGB, pt1=(3766, 200), pt2=(250, 2926), color=(255, 0, 0), thickness=3)
RGB = cv2.line(RGB, pt1=(2008, 200), pt2=(2008, 2926), color=(255, 0, 0), thickness=3)

cv2.imwrite(filename.split(".")[0] + "_trans.JPG", trans)
cv2.imwrite(filename.split(".")[0] + "_trans_eq.JPG", blurred)
cv2.imwrite(filename.split(".")[0] + "_th.JPG", RGB)

plt.subplot(231), plt.imshow(img, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(trans, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(trans_eq, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(blurred, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(dst, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(RGB)
plt.xticks([]), plt.yticks([])
plt.show()
