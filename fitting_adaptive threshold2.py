import cv2


# コールバック関数（トラックバーが変更されたときに呼ばれる関数）
def on_trackbar(val):
    if img is not None:
        C = 12
        # 二値化
        dst = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            val,
            C - 10,
        )
        # 画像の表示
        cv2.imshow("img", dst)


# ウィンドウの作成
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# トラックバーの作成
cv2.createTrackbar(
    "Threshold",  # トラックバーの名前
    "img",  # トラックバーを表示するウィンドウのタイトル
    101,  # 初期値
    1001,  # 最大値(最小値は０で固定)
    on_trackbar,  # コールバック関数
)

# 画像の読込
img = cv2.imread(r"shieldinput\ML103_240228\ML103_240228_trans_eq.JPG", 0)

# トラックバーの値を取得
track_value = cv2.getTrackbarPos("Threshold", "img")
# 最初の１回目の処理を取得した値で実行
on_trackbar(track_value)

# キー入力待ち
cv2.waitKey(0)
