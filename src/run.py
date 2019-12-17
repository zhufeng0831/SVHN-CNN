import sys
import cv2
import os
from SingleDetectionAndRecognition import detect_and_classify

arguments = len(sys.argv)
if arguments == 2:
    path = sys.argv[1]
    image = cv2.imread(os.path.join(path))
    out_image = detect_and_classify(image)
    cv2.imwrite("out_image.jpg", out_image)
else:
    print("Error - invalid number of arguments - Usage: python run.py path")