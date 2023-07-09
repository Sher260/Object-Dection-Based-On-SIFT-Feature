from sklearn.svm import LinearSVC,SVC
from skimage.io import imread
from skimage import transform, data
import joblib
import matplotlib.pyplot as plt
import os
import random
import time as tm
import cv2
import numpy as np

NONE_TYPE = type(None)  # fix some error case

# 加载分类器模型
model_path = 'C:/projects/svm_detection-master/svm_model_fenlei'
clf = joblib.load(model_path)

# 加载测试集
path = "C:/Users/hitzo/Desktop/datami/test"
file_name_list = os.listdir(path)
file_name_list_new = []  # 图片所在文件夹
for file_name in file_name_list:
    tmp_file = path + '/' + file_name
    a = os.path.isdir(tmp_file)
    # b = os.path.isfile(tmp_file)
    if os.path.isdir(tmp_file):
        file_name_list_new.append(tmp_file)
N_file = len(file_name_list_new)  # 标注图片数量
# random.shuffle(file_name_list_new)
N_win = 80  # 块大小
N_x = 480 // N_win
N_y = 800 // N_win
N_block = N_x * N_y  # 块的数量
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
plt.show()

detect, extract = cv2.SIFT_create(), cv2.SIFT_create()
index_params = dict(algorithm=1, trees=6)  # 5棵树的核密度树索引算法
search_params = dict(checks=100)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)
bow_kmeans_trainer = cv2.BOWKMeansTrainer(144)
vocabulary = np.loadtxt('vocabulary.txt',dtype=np.float32)
extract_bow.setVocabulary(vocabulary)#把词表分配给BoW描述符提取器

for i_file, file_name in enumerate(file_name_list_new):
    x_list = []  # 非常多的小块
    y_list = []  # 对应的标签
    t1 = tm.time()

    x_path = file_name + '/img.jpg'
    y_path = file_name + '/label.jpg'

    x_im = imread(x_path, as_gray=True)
    y_im = imread(y_path, as_gray=True)

    x_dst = transform.resize(x_im, (480, 800))  # 小图
    y_dst = transform.resize(y_im, (480, 800))  # 小图
    y_dst = y_dst / np.max(y_dst)  # 归一化

    target_area = np.sum(y_dst)  # 目标面积大小

    # 划分小块
    position_list = np.zeros([N_block, 2])  # 每一个小块的坐标
    tmp = 0
    for i in range(N_x):
        for j in range(N_y):
            position_list[tmp, :] = [i * N_win, j * N_win]
            x_tmp = x_dst[i * N_win:(i + 1) * N_win, j * N_win:(j + 1) * N_win]
            x_tmp = cv2.normalize(x_tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            bowData = extract_bow.compute(x_tmp, detect.detect(x_tmp))
            if type(bowData) != NONE_TYPE:
                x_list.extend(bowData)
            else:
                x_list.extend(np.zeros((1, 144)))
            y_tmp = y_dst[i * N_win:(i + 1) * N_win, j * N_win:(j + 1) * N_win]
            y_flag = np.sum(y_tmp) > target_area * 0.3  # 包含目标的小块是正样本
            y_list.append(y_flag)
            tmp += 1

    pred = clf.predict(x_list)
    pred1 = clf.decision_function(x_list) > 0.4
    N_target = np.sum(np.array(y_list) == 1)  # 总目标数量
    N_detet = np.sum(pred[np.array(y_list) == 1])  # 检测到的目标
    pd = N_detet / N_target

    detect_position = position_list[pred1 == 1, :]
    ax.cla()
    ax.imshow(x_dst)
    for i in range(detect_position.shape[0]):
        x0 = detect_position[i, 0]
        y0 = detect_position[i, 1]

        rect = plt.Rectangle([y0, x0], N_win, N_win, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    a = 1
    filename = 'C:/Users/hitzo/Desktop/picsift/' + str(i_file) + '.jpg'
    plt.savefig(filename)