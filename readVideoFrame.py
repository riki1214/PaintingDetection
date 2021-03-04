import cv2
from detection_rectification import get_paintings
from retrieval import orb_detection
from utils import initialize_orb
from yolo.detect import people_detection

def readFrame(path, paintings, model, fps):
    frm = 0
    orb, bf = initialize_orb()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        exit(0)
    width, height = int(cap.get(3)), int(cap.get(4))
    rescaling = 0.8
    frame_dimensions = (int(width * rescaling), int(height * rescaling))
    dict = {}
    if fps is None or fps <= 0:
        fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_count = 0

    while f_count <= frame_count - fps:
        _, img = cap.read()
        if img is None:
            f_count += fps
            continue
        f_count += fps
        cap.set(1, f_count)
        img = cv2.resize(img, frame_dimensions, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

        #  here we detect and rectificate paintings found in frame
        bounding_boxes_coordinates, classified_paintings, rectified_paintings = get_paintings(img)

        room=-1
        if rectified_paintings or len(rectified_paintings) != 0:
            # orb detection
            room=orb_detection(rectified_paintings, paintings, bounding_boxes_coordinates, img, orb, bf, dict, frm)
        else:
            if room != -1:
                cv2.putText(img, str("Room: ") + str(room),
                            (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        people_detection(img, bounding_boxes_coordinates, model)

        cv2.imshow('final_img', cv2.resize(img, (1280, 720)))

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
        frm+=1

    cap.release()
