import sys
from readVideoFrame import readFrame
from utils import read_csv_db, read_model

CSV_PATH = "project_material/data.csv"
DEFAULT_IMAGES_PATH = "project_material/paintings_db"
DEFAULT_VIDEO_PATH = "project_material/videos/001/GOPR5826.MP4"

def main():
    model = read_model()
    root = sys.argv[0].replace("main.py", "")
    if len(sys.argv) == 3:
        video_path = root + "project_material/videos/" + sys.argv[1]
        paintings = read_csv_db(CSV_PATH, DEFAULT_IMAGES_PATH)
        readFrame(video_path, paintings, model, int(sys.argv[2]))
    elif len(sys.argv) == 2:
        video_path = root + "project_material/videos/" + sys.argv[1]
        paintings = read_csv_db(CSV_PATH, DEFAULT_IMAGES_PATH)
        readFrame(video_path, paintings, model, 0)
    elif len(sys.argv) == 1:
        paintings = read_csv_db(root + CSV_PATH, root + DEFAULT_IMAGES_PATH)
        readFrame(root + DEFAULT_VIDEO_PATH, paintings, model, 0)
    else:
        return "Parameters should be set correctly --> Options: No parameters | directory_number/filename (e.g. 001/GOPR5827.MP4) | " \
               "directory_number/filename fps"


if __name__ == "__main__":
    main()
