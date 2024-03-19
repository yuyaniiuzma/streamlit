import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
import numpy as np
from PIL import Image
import random


def pil2cv(image):
    """PIL型->OpenCV型"""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def sidebar_parm():
    uploaded_file = st.sidebar.file_uploader("画像アップロード", type="jpg")
    mode = st.sidebar.selectbox("モードの選択", ("射影変換なし", "射影変換あり"))
    mode2 = st.sidebar.selectbox(
        "二値化手法の選択",
        ("threshold", "adaptive_threshold_M", "adaptive_threshold_G", "otsu"),
    )
    if mode2 == "threshold":
        th = st.sidebar.slider("threshold value", 0, 255, 125)
        bl = None
        C = None
    elif mode2 == "adaptive_threshold_M" or mode2 == "adaptive_threshold_G":
        th = None
        bl = st.sidebar.slider("blocksize", 1, 501, 3, 2)
        C = st.sidebar.slider("C", 1, 100, 20)
    elif mode2 == "otsu":
        th = None
        bl = None
        C = None
    button_run = st.sidebar.button("measure")
    return uploaded_file, mode, mode2, th, bl, C, button_run


def get_coordinate(uploaded_file):
    if "coord_lst" not in st.session_state:
        st.session_state["coord_lst"] = []
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        height_img_base, width_img_base = img_array.shape[:2]
        expander_width = 700
        aspect_ratio = width_img_base / height_img_base
        new_width = expander_width
        new_height = int(expander_width / aspect_ratio)
        img_resize = cv2.resize(img_array, (new_width, new_height))
        ratio_w = width_img_base / new_width
        ratio_h = height_img_base / new_height

        value = streamlit_image_coordinates(img_resize)
        if value is not None:
            if len(st.session_state["coord_lst"]) < 4:
                coordinates = int(value["x"] * ratio_w), int(value["y"] * ratio_h)
                st.session_state["coord_lst"].append(coordinates)
                st.write(st.session_state["coord_lst"])
                if len(st.session_state["coord_lst"]) == 4:
                    st.success(
                        "座標の入力が完了しました。やり直しがしたい場合はResetボタンをクリックしてください。"
                    )
                    if st.button("Reset", type="primary"):
                        st.session_state["coord_lst"] = []
                    else:
                        coordinate_list = st.session_state["coord_lst"]
                        return coordinate_list
            elif len(st.session_state["coord_lst"]) > 4:
                st.warning(
                    "座標が4個以上入力されました。Resetボタンをクリックしてください。"
                )
                if st.button("Reset", type="primary"):
                    st.session_state["coord_lst"] = []


def transform(uploaded_file, coordinate_list):
    if uploaded_file is not None and coordinate_list is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        origin_list = coordinate_list
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

        M = cv2.getPerspectiveTransform(p_original, p_trans)
        i_trans = cv2.warpPerspective(img_array, M, (4032, 3024))
        col1, col2 = st.columns(2)
        col1.header("元画像")
        col2.header("射影変換画像")
        col1.image(img_array)
        col2.image(i_trans)
        return i_trans


def threshold(uploaded_file, i_trans, mode, mode2, th, bl, C):
    col3, col4 = st.columns(2)
    col3.header("元画像")
    col4.header("二値化画像")
    if uploaded_file is not None:
        if mode == "射影変換なし":
            image = Image.open(uploaded_file)
            img_array = np.array(image)
        else:
            img_array = i_trans
        col3.image(img_array)
        # img = pil2cv(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        if mode2 == "threshold":
            ret, th1 = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
        elif mode2 == "adaptive_threshold_M":
            th1 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bl, C
            )
        elif mode2 == "adaptive_threshold_G":
            th1 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bl, C
            )
        elif mode2 == "otsu":
            ret, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        col4.image(th1)
        return th1  # elseのときに返り値ないのが問題?


def rabeling(th1):
    col5, col6 = st.columns(2)
    col5.header("rabeling")
    col6.header("result")
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(th1)
    height, width = th1.shape[:2]

    colors = []
    for i in range(1, ret + 1):
        colorR = random.randint(0, 255)
        colorG = random.randint(0, 255)
        colorB = random.randint(0, 255)
        colors.append(np.array([colorR, colorG, colorB]))

    image = Image.open(uploaded_file)
    img = pil2cv(image)
    color_src = img.copy()
    for y in range(0, height):
        for x in range(0, width):
            if markers[y, x] > 0:
                _, _, _, _, size = stats[markers[y, x]]
                if size >= 50000:
                    color_src[y, x] = colors[markers[y, x]]
                else:
                    color_src[y, x] = [0, 0, 0]
            else:
                color_src[y, x] = [0, 0, 0]
    col5.image(color_src)

    s_img = color_src.copy()
    for coord in stats[1:]:
        if coord[4] >= 50000:
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
    col6.image(s_img)


if __name__ == "__main__":
    st.title("ターゲット曇り面積計測")
    uploaded_file, mode, mode2, th, bl, C, button_run = sidebar_parm()
    if mode == "射影変換あり":
        coordinate_list = get_coordinate(uploaded_file)
        i_trans = transform(uploaded_file, coordinate_list)
        print(i_trans, coordinate_list)
        if i_trans is not None:
            th1 = threshold(uploaded_file, i_trans, mode, mode2, th, bl, C)
            rabeling(th1)
        # if button_run is True:
        # th1 = threshold(uploaded_file, i_trans, mode, mode2, th, bl, C)
        # rabeling(th1)
    else:
        # i_trans = None
        i_trans = uploaded_file
        print(uploaded_file, i_trans, mode, mode2, th, bl, C)
        th1 = threshold(uploaded_file, i_trans, mode, mode2, th, bl, C)
        if button_run is True:
            rabeling(th1)