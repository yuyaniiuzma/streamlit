import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# 二値化閾値とファイルネームを指定
threshold = 152
# threshold2 = 240
filename = r"target input\ML108_240228_SIB43-647-1\ML108_SIB43-647-1.JPG"

# カラー画像の取り込み
imgColorBGR = cv2.imread(filename, cv2.IMREAD_COLOR)
imgColorRGB = cv2.cvtColor(imgColorBGR, cv2.COLOR_BGR2RGB)

# 射影変換前後の対応点を4点設定(左右上下の順)
origin_list = [[592, 2116], [2816, 2036], [1720, 796], [1700, 3372]]
vert = origin_list[1][0] - origin_list[0][0]
hori = origin_list[3][1] - origin_list[2][1]
trans_list = [
    [
        origin_list[0][0] - (hori - vert) / 2,
        (origin_list[0][1] + origin_list[1][1]) / 2,
    ],
    [
        origin_list[1][0] + (hori - vert) / 2,
        (origin_list[0][1] + origin_list[1][1]) / 2,
    ],
    [(origin_list[2][0] + origin_list[3][0]) / 2, origin_list[2][1]],
    [(origin_list[2][0] + origin_list[3][0]) / 2, origin_list[3][1]],
]
p_original = np.float32(origin_list)
p_trans = np.float32(trans_list)

# 射影変換
M = cv2.getPerspectiveTransform(p_original, p_trans)
i_trans = cv2.warpPerspective(imgColorRGB, M, (3024, 4032))

i2_trans = i_trans.copy()

cv2.line(
    i2_trans,
    pt1=(444, 1420),
    pt2=(1064, 1204),
    color=(0, 0, 0),
    thickness=1,
    lineType=cv2.LINE_4,
    shift=0,
)

cv2.line(
    i2_trans,
    pt1=(404, 2656),
    pt2=(968, 2880),
    color=(0, 0, 0),
    thickness=1,
    lineType=cv2.LINE_4,
    shift=0,
)

imgGray = cv2.cvtColor(i2_trans, cv2.COLOR_RGB2GRAY)  # BGRをグレイスケールに変換

# 手動二値化
ret, imgBin = cv2.threshold(imgGray, threshold, 255, cv2.THRESH_BINARY)
# 範囲指定二値化
# imgBin = cv2.inRange(imgGray, threshold, threshold2)

# ret2はラベル数,markersはラベル番号を配列した画像
# statsは物体ごとの座標と面積,centroidsは中心座標

ret2, markers, stats, centroids = cv2.connectedComponentsWithStats(imgBin)

color_src = i2_trans.copy()
height, width = imgBin.shape[:2]

# 領域の数だけ用意
colors = []
for i in range(1, ret2 + 1):
    colorR = random.randint(0, 255)
    colorG = random.randint(0, 255)
    colorB = random.randint(0, 255)
    colors.append(np.array([colorR, colorG, colorB]))

# markersの値に応じて色を決めていく
for y in range(0, height):
    for x in range(0, width):
        if markers[y, x] > 0:
            _, _, _, _, size = stats[markers[y, x]]
            if size >= 10000:
                color_src[y, x] = colors[markers[y, x]]
            else:
                color_src[y, x] = [0, 0, 0]
        else:
            color_src[y, x] = [0, 0, 0]

# 各ラベルの座標と面積を描画する
s_img = color_src.copy()
for coord in stats[1:]:
    if coord[4] >= 10000:
        l_top = (coord[0], coord[1])
        r_bot = (coord[0] + coord[2], coord[1] + coord[3])
        s_img = cv2.rectangle(s_img, l_top, r_bot, (255, 0, 0), 20)
        s_img = cv2.putText(
            s_img,
            str(round(coord[4])),
            l_top,
            cv2.FONT_HERSHEY_PLAIN,
            15,
            (255, 0, 0),
            15,
        )
        print(str(round(coord[4])))
print(trans_list)

plt.subplot(221), plt.imshow(imgColorRGB, cmap="gray")
plt.title("img"), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(imgGray, cmap="gray")
plt.title("img"), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(imgBin, cmap="gray")
plt.title("img"), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(s_img, cmap="gray")
plt.title("img"), plt.xticks([]), plt.yticks([])
plt.show()

cv2.imwrite(filename.split(".")[0] + "_color.png", color_src)
cv2.imwrite(filename.split(".")[0] + "_stats.png", s_img)
cv2.imwrite(filename.split(".")[0] + "_gray.png", imgGray)
cv2.imwrite(filename.split(".")[0] + "_thresh.png", imgBin)
