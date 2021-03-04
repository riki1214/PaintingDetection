import numpy as np
import scipy.stats
import cv2
from utils import remove_bb_proportion, remove_bb_internal, remove_img_one_color, sobel_operator


def get_bounding_boxes(img, krnl_size, min_area_percent):
    img = cv2.medianBlur(img, krnl_size)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = sobel_operator(img_gray)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    significant_contours = [contours[i] for i in np.flatnonzero(hierarchy[0, :, 3] == -1) if
                            cv2.contourArea(contours[i]) > (img.size * min_area_percent)]

    bounding_boxes_coordinates = [cv2.boundingRect(contour) for contour in significant_contours
                                  if cv2.boundingRect(contour)[2] < img.shape[1] * 0.95
                                  if cv2.boundingRect(contour)[3] < img.shape[0] * 0.95]

    return bounding_boxes_coordinates


def get_classified_painting(img, bounding_boxes_coordinates, entropy_threshold, krnl_size_denoise_entropy):
    classified_paintings = []
    bounding_boxes_coordinates_new = []
    for coordinates in bounding_boxes_coordinates:
        x, y, w, h = coordinates
        obj = img[y:y + h, x:x + w, :]
        obj_blur = cv2.medianBlur(obj, krnl_size_denoise_entropy)

        # calculate entropy
        hist_b = cv2.calcHist([obj_blur], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([obj_blur], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([obj_blur], [2], None, [256], [0, 256])
        hist = np.concatenate((hist_b, hist_g, hist_r))
        hist_normalized = hist / obj_blur.size

        # classification
        if scipy.stats.entropy(hist_normalized) > entropy_threshold:
            classified_paintings.append(obj)
            bounding_boxes_coordinates_new.append(coordinates)


    return classified_paintings , bounding_boxes_coordinates_new


def get_rectified_painting(classified_paintings):
    rectified_paintings = []
    for painting in classified_paintings:
        img_gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)

        painting2 = cv2.dilate(sobel_operator(img_gray), None, iterations=2)
        #
        contours = cv2.findContours(painting2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # we sort and take the max contour of the classified image
        max_contour = sorted(contours[0], key=cv2.contourArea, reverse=True)[0]
        epsilon = 0.015 * cv2.arcLength(max_contour, True)
        approx_corners = cv2.approxPolyDP(max_contour, epsilon, True)
        # if approx_corners.shape[0] == 4:
        approx_corners = sorted(np.concatenate(approx_corners).tolist())
        x = [a[0] for a in approx_corners]
        y = [a[1] for a in approx_corners]
        sum = np.array(x) + np.array(y)
        difference = np.array(x) - np.array(y)
        # we are going to take the four corners [tl, br, bl, tr]
        tl, br, bl, tr = approx_corners[np.argmin(sum)], approx_corners[np.argmax(sum)], \
                         approx_corners[np.argmin(difference)], approx_corners[np.argmax(difference)]

        corners = [tl, bl, br, tr]

        # we need new height and width for the new image
        w_up, w_down, h_left, h_right = \
            np.sqrt(np.power((br[0] - bl[0]), 2) + np.power((br[1] - bl[1]), 2)), \
            np.sqrt(np.power((tr[0] - tl[0]), 2) + np.power((tr[1] - tl[1]), 2)), \
            np.sqrt(np.power((tr[0] - br[0]), 2) + np.power((tr[1] - br[1]), 2)), \
            np.sqrt(np.power((tl[0] - bl[0]), 2) + np.power((tl[1] - bl[1]), 2))

        # we keep the max of these dimensions, to be sure the paint is in the box
        maxWidth = max(int(w_up), int(w_down))
        maxHeight = max(int(h_left), int(h_right))
        point_src = np.float32(corners)
        point_dst = np.array([[0, 0], [0, maxHeight],
                              [maxWidth, maxHeight],
                              [maxWidth, 0]
                              ], dtype="float32")
        H = cv2.getPerspectiveTransform(point_src, point_dst)
        rectified_image = cv2.warpPerspective(painting, np.float32(H), (maxWidth, maxHeight))
        rectified_paintings.append(rectified_image)

    return rectified_paintings


def get_paintings(img, krnl_size=5, min_area_percent=0.01, entropy_threshold=5, krnl_size_denoise_entropy=25):
    bounding_boxes_coordinates = get_bounding_boxes(np.copy(img), krnl_size, min_area_percent)  # x,y,w,h

    bounding_boxes_coordinates = remove_bb_proportion(remove_bb_internal(bounding_boxes_coordinates))

    classified_paintings, bounding_boxes_coordinates = get_classified_painting(np.copy(img), bounding_boxes_coordinates, entropy_threshold,
                                                   krnl_size_denoise_entropy)

    rectified_paintings = get_rectified_painting(classified_paintings)

    if not rectified_paintings:
        return bounding_boxes_coordinates, classified_paintings, []
    else:
        rectified_paintings = [rectified_painting for rectified_painting in rectified_paintings if
                               rectified_painting is not None]

    rectified_paintings, bbc_filtered = remove_img_one_color(rectified_paintings,
                                                             bounding_boxes_coordinates)

    return bbc_filtered, classified_paintings, rectified_paintings
