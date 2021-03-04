# VCS project - Group 15
Final project for Vision and Cognitive System

## Requirements
- numpy
- opencv-python
- opencv-contrib-python
- torch
- pandas
- os
- scipy
- python 3.7

## Pipeline
For each video frame:

### Painting detection & segmentation
Predict a ROI for each painting:
1. Median Filtering
2. Edge Detection with Sobel
3. Mean and Thresholding
5. Dilate Transformation
6. Find Contours and check for Significant Contours
    -(cv2.findContours)
7. Discard false positives:
   - Check dimensions of bounding boxes
   - Classification Histogram
   - Minimum of colors in classificated paintings

### Painting rectification
Starting from contours found in previous point (7) and considering one contour at a time:
1. Polygonal approximation (cv2.approxPolyDP)
2. Order vertices
3. Compute top-left, top-right, bottom-left, bottom-right corners
4. Compute maxHeight and maxWidth for rectified images
7. Get perspective and Warp perspective

### Painting retrieval & localization
Match each detected painting to the paintings DB:
1. Find descriptors with ORB
2. Find best matches (BFMatcher with Hamming normalization and knnMatcher)
3. Find room in which paintings are collocated

### People detection
Predict a ROI around each person:
1. YOLO v3 (from OpenCV)
   For each detection:
   - Predict a score for each class
   - Take only the score belonging to the person class
   - Thresholding
   - Non-maximum suppression

2. Discard people in paintings


## How to run the code
### First choice
1. Load the complete project on your IDE (we used PyCharm)
2. Put videos and painting_db in the respective empty directory already created.
	- videos put in 
	
	           -"project_material/videos/<number_of_directory>/<video_name>"
	- database paintings put in
	           
	           - "project_material/paintings_db"
3. Run project from main.py:

		- choose to run with default parameters <number_of_directory> <video_name> already set (no parameters to specify)	
		- choose <number_of_directory> and <video_name>	--- e.g ---> 001/GOPR5826.MP4
		- choose <number_of_directory>,  <video_name> and <fps> --- e.g ---> 001/GOPR5826.MP4 10
	You can set parameters from the run configuration panel of the IDE.
	
### Second choice
1. Open a bash terminal on the project directory
2. Execute 
    
    - python main.py <parameters>
	- Parameters choice is the same described above.

### Upload yolo.weight and put in yolo folder
