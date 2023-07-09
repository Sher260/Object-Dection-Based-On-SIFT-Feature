from cv2 import CV_32F
from skimage.io import imread
from skimage import transform, data
import joblib
import numpy as np
import os
# from config import *
import matplotlib.pyplot as plt
import time as tm
import cv2

NONE_TYPE = type(None)# fix some error case
# READ_TYPE = img_color

#============2 提取SIFT特征==============
def extract_sift(img, extractor, detector):
  """
  提取sift特征
  """
  #im = cv2.imread(fn,READ_TYPE)
  return extractor.compute(img, detector.detect(img))[1]
# ======================================

# ===========1 获取数据文件===============
path = 'C:/Users/hitzo/Desktop/datami/playground'  # 这里是背景为操场的数据集

file_name_list = os.listdir(path)#返回路径下文件名称的列表
file_name_list_new = []  # 图片所在文件夹
for file_name in file_name_list:
    tmp_file = path + '/' + file_name
    a = os.path.isdir(tmp_file)
    # b = os.path.isfile(tmp_file)
    if os.path.isdir(tmp_file):
        file_name_list_new.append(tmp_file)
N_file = len(file_name_list_new)  # 标注图片数量
# ======================================
a = 1
##图片大小和分得小块的数量
N_win = 80  # 块大小
N_x = 480 // N_win
N_y = 800 // N_win
N_block = N_x * N_y  # 块的数量，此算法表示未采用滑动窗

x_list = []  # 非常多的小块 的特征
y_list = []  # 对应的标签

# ===========3 初始化bow提取器和bow训练器===============
detect, extract = cv2.SIFT_create(), cv2.SIFT_create()
index_params = dict(algorithm=1, trees=6)  # 5棵树的核密度树索引算法
search_params = dict(checks=100)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)
bow_kmeans_trainer = cv2.BOWKMeansTrainer(144)
# ==================================================

t1 = tm.time()
# 根据数据集生成词汇表时不注释，后面在对数据集分类做检测时需要重新提取SIFT特征，但是是基于所有数据的词汇表，不用重新生成，可以注释
# ===========4 提取sift特征，利用bow进行聚类，得到36个聚类中心===============
# for i_file,file_name in enumerate(file_name_list_new):
#
#     x_path = file_name + '/img.jpg'
#     y_path = file_name + '/label.jpg'
#
#     x_im = imread(x_path, cv2.IMREAD_COLOR)  #这里不知道为什么，读取结果是灰度
#     y_im = imread(y_path, cv2.IMREAD_COLOR)
#
#     # cv2.imshow("image", x_im)
#     #
#     # cv2.waitKey(0)
#
#     x_dst = transform.resize(x_im, (480, 800))  # 小图
#     y_dst = transform.resize(y_im, (480, 800))  # 小图
#     y_dst = y_dst / np.max(y_dst)  # 归一化
#
#
#     target_area = np.sum(y_dst)  # 目标面积大小
#
#     for i in range(N_x):
#         for j in range(N_y):
#             x_tmp = x_dst[i * N_win:(i + 1) * N_win, j * N_win:(j + 1) * N_win]  #小图位置
#             x_tmp = cv2.normalize(x_tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  #数据格式转化为可提取sift特征的
#             siftData = extract_sift(x_tmp, extract, detect)
#             if type(siftData) != NONE_TYPE:  # 避免负样本内没有sift特征
#                 bow_kmeans_trainer.add(siftData)
#
#     print(file_name)
#
# t2 = tm.time()
#
# vocabulary = bow_kmeans_trainer.cluster()#特征添加完后聚类得到voc，也就是聚类的中心
# np.savetxt('vocabulary.txt', vocabulary)
# # vocabulary = np.loadtxt('vocabulary.txt')
vocabulary = np.loadtxt('vocabulary.txt', dtype=np.float32)  # 加载生成的词汇表
extract_bow.setVocabulary(vocabulary)#把词表分配给BoW描述符提取器
# ==================================================

# print(f'完成词汇表提取 | 时间：{t2-t1}')

# ===========5 提取sift特征，转化为bow特征，即36维的特征直方图===============
for i_file,file_name in enumerate(file_name_list_new):
    t1 = tm.time()
    x_path = file_name + '/img.jpg'
    y_path = file_name + '/label.jpg'

    x_im = imread(x_path, as_gray=True)
    y_im = imread(y_path, as_gray=True)
    # x_im = imread(x_path, 0)  # 这里不知道为什么，读取结果是灰度
    # y_im = imread(y_path, 0)

    x_dst = transform.resize(x_im, (480, 800))  # 小图
    y_dst = transform.resize(y_im, (480, 800))  # 小图
    y_dst = y_dst / np.max(y_dst)  # 归一化

    target_area = np.sum(y_dst)  # 目标面积大小

    for i in range(N_x):
        for j in range(N_y):
            x_tmp = x_dst[i * N_win:(i + 1) * N_win, j * N_win:(j + 1) * N_win]
            x_tmp = cv2.normalize(x_tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            descriptors_1 = detect.detect(x_tmp)
            bowData = extract_bow.compute(x_tmp, detect.detect(x_tmp))

            if type(bowData) != NONE_TYPE:
                x_list.extend(bowData)
            else:
                x_list.extend(np.zeros((1, 144)))
            y_tmp = y_dst[i * N_win:(i + 1) * N_win, j * N_win:(j + 1) * N_win]
            y_flag = np.sum(y_tmp) > target_area * 0.4  # 包含目标的小块是正样本
            y_list.append(y_flag)

    t2 = tm.time()
    print(f'文件：{i_file}/{N_file} | 时间：{t2-t1}')

    # if i_file > 3:
    #     break
# ==================================================


# # 将list格式--->矩阵格式
N_block_all = len(x_list) # 总块数量
N_feature = x_list[0].size
x_list1 = np.zeros((N_block_all, N_feature))
y_list1 = np.zeros((N_block_all, 1))
for i_block in range(N_block_all):
    x_list1[i_block, :] = x_list[i_block]
    y_list1[i_block, :] = y_list[i_block]

np.save('data_x_playground.npy', x_list1)
np.save('data_y_playground.npy', y_list1)

aa = 1