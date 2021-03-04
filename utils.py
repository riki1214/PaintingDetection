import operator
from yolo.darknet import Darknet
import cv2
import numpy as np
import os
import pandas as pd
from itertools import combinations


def convertScale(img, alpha, beta):
    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

def sobel_operator(img_gray):

    sobel_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
    sobel = np.hypot(sobel_x, sobel_y)
    painting2 = np.clip(sobel, 0, 255).astype(np.uint8)

    # Reduce noise by zeroing any value which is less than mean
    mean = np.mean(painting2)
    painting2[painting2 <= mean] = 0

    return painting2



# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):  # 25%
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


def crop_img(img, cut):
    xi = int(img.shape[0] * cut)
    xf = int(img.shape[0] - xi)
    yi = int(img.shape[1] * cut)
    yf = int(img.shape[1] - yi)
    crop = img[xi:xf, yi: yf]
    return crop


def check_room(dict, frame_img):
    if len(dict) != 0:
        room = max(dict.items(), key=operator.itemgetter(1))[0]
        cv2.putText(frame_img, str("Room: ") + str(room),
                    (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        room = ""
    return room


def read_model():
    model = Darknet('yolo/cfg/yolov3.cfg')
    model.load_weights('yolo/yolov3.weights')
    return model


def initialize_orb():
    orb = cv2.ORB_create(scaleFactor=1.4, WTA_K=2, scoreType=cv2.ORB_FAST_SCORE)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    return orb, bf

class Image:
    def __init__(self):
        self.filename = ''
        self.image = ''
        self.descriptors = ''
        self.keypoints = ''
        self.matches = None
        self.title = ''
        self.author = ''
        self.room = ''


def read_csv_db(CSV_PATH, IMAGES_PATH):
    orb = cv2.ORB_create(scaleFactor=1.4, WTA_K=2, scoreType=cv2.ORB_FAST_SCORE)
    data = pd.read_csv(CSV_PATH, sep=",", encoding='utf-8')
    title = data["Title"]
    author = data["Author"]
    room = data["Room"]
    paintings = []

    with os.scandir(IMAGES_PATH) as images:
        for t, a, r, i in zip(title, author, room, images):
            if i.name.endswith(".png") and i.is_file():
                img = cv2.imread(i.path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray, None)
                im = Image()
                im.title, im.author, im.room, im.filename, im.image, im.keypoints, im.descriptors = t, a, r, i.name, img, kp, des
                paintings.append(im)

    return paintings


def remove_bb_internal(bbc):
    lista = []
    removed = 0
    bbc_new = bbc.copy()
    for i, b in enumerate(bbc):
        lista.append(i)

    combinazioni = list(combinations(lista, 2))
    for i in combinazioni:
        if bbc[i[0]][0] >= bbc[i[1]][0] and bbc[i[0]][1] >= bbc[i[1]][1] and (bbc[i[0]][0] + bbc[i[0]][2]) <= (
                bbc[i[1]][0] + bbc[i[1]][2]) and (bbc[i[0]][1] + bbc[i[0]][3]) <= (bbc[i[1]][1] + bbc[i[1]][3]):
            bbc_new.pop(i[0] - removed)
            removed += 1
        if bbc[i[1]][0] >= bbc[i[0]][0] and bbc[i[1]][1] >= bbc[i[0]][1] and (bbc[i[1]][0] + bbc[i[1]][2]) <= (
                bbc[i[0]][0] + bbc[i[0]][2]) and (bbc[i[1]][1] + bbc[i[1]][3]) <= (bbc[i[0]][1] + bbc[i[0]][3]):
            bbc_new.pop(i[1] - removed)
            removed += 1

    return bbc_new


def remove_bb_proportion(bbc, scale=0.5):
    j = 0
    for b in bbc.copy():
        if b[2] < b[3] and b[2] / b[3] <= scale:
            bbc.pop(j)
            continue
        if b[3] < b[2] and b[3] / b[2] <= scale:
            bbc.pop(j)
            continue
        j += 1
    return bbc


def remove_img_one_color(imgs, bbc):
    j = 0
    for i in imgs.copy():
        q = np.zeros(256, dtype=int)
        q[imgs[j].ravel()] = 1
        v = np.nonzero(q)[0]
        if v.size <= 10:
            imgs.pop(j)
            bbc.pop(j)
        else:
            j += 1
    return imgs, bbc
