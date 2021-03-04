import cv2
from utils import automatic_brightness_and_contrast, check_room, crop_img

def orb_detection(rectified_paintings, paintings, bbc, frame_img, orb, bf, dict, frm):
    room = check_room(dict, frame_img)

    for j, r in enumerate(rectified_paintings):
        score_list = []

        r, alpha, beta = automatic_brightness_and_contrast(r)
        r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        r = crop_img(r, 0.15)
        kp1, des1 = orb.detectAndCompute(r, None)

        for i, paint in enumerate(paintings):
            matches = bf.knnMatch(des1, paint.descriptors, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])

            if len(good) > 10:
                score_list.append((len(good), paint))

        if len(score_list) > 0:
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            if dict.get(str(score_list[0][1].room)):
                dict[str(str(score_list[0][1].room))] += 1
            else:
                dict[str(str(score_list[0][1].room))] = 1

            if frm==0 and j == len(rectified_paintings)-1 :
                room = check_room(dict, frame_img)

            if frm == 0:
                cv2.rectangle(frame_img, (bbc[j][0], bbc[j][1]), (bbc[j][0] + bbc[j][2], bbc[j][1] + bbc[j][3]),
                              (36, 255, 12), 1)
                cv2.putText(frame_img, (str(score_list[0][1].filename + " " + score_list[0][1].title)),
                            (bbc[j][0], bbc[j][1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
            else:
                if str(score_list[0][1].room) == room :
                    cv2.rectangle(frame_img, (bbc[j][0], bbc[j][1]), (bbc[j][0] + bbc[j][2], bbc[j][1] + bbc[j][3]),
                                  (36, 255, 12), 1)
                    cv2.putText(frame_img, (str(score_list[0][1].filename + " " + score_list[0][1].title)),
                                (bbc[j][0], bbc[j][1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                else:
                    cv2.rectangle(frame_img, (bbc[j][0], bbc[j][1]), (bbc[j][0] + bbc[j][2], bbc[j][1] + bbc[j][3]),
                                  (0, 0, 255), 1)

        else:
            cv2.rectangle(frame_img, (bbc[j][0], bbc[j][1]), (bbc[j][0] + bbc[j][2], bbc[j][1] + bbc[j][3]),
                          (0, 0, 255), 1)
    return room