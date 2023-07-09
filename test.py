import os
import cv2
import numpy as np
import glob
# name = os.path.splitext("C:/Users/hitzo/Desktop/dronelabel/0100001.xml")[0]
# print(name)
# path = os.path.dirname(name)
# number = name[len(path):]
# #number = number.split('/')
# print(number)

# f = cv2.imread("C:/Users/hitzo/Desktop/0100002.jpg", cv2.IMREAD_COLOR)
# f = np.zeros((1, 36))
# print(f)
# d = np.ones((1, 36))
# x = []
# x.extend(d)
# x.extend(f)
# print(x)

for image_path in glob.glob("C:/Users/hitzo/Desktop/drone_dataset/train/img/*.jpg"):
    image_name = image_path.split('\\')[-1]
    print(image_name)
