import os
import shutil

# #=======####移除label文件夹内多余的.xml标签文件=========
# picpath = 'C:/Users/hitzo/Desktop/drone_dataset/train/img'##数据集图片所在位置
# labelpath = "C:/Users/hitzo/Desktop/drone_dataset/train/xmllabels"##标签所在位置
# pic_name_list = os.listdir(picpath)
# label_name_list = os.listdir(labelpath)
#
# filename_pic = []
# filename_label = []
# ##统计出所有图片的名称（不包含后缀
# for file_name in pic_name_list:
#     file_name = file_name.split('.')[0]#去除后缀
#     filename_pic.append(file_name)
#
# a = 0
# ##统计出所有标签的名称（不包含后缀
# for file_name in label_name_list:
#     file_name = file_name.split('.')[0]
#     if file_name not in filename_pic:#此标签没有对应的图片
#         file_name_dot = labelpath + '/' + file_name + '.xml'#拼凑出标签路径
#         print(file_name_dot)
#         a = a + 1
#         os.remove(file_name_dot)#删掉
#
# print(a)#算一下数字对不对

#===========添加图片到bound文件夹内==============
# for file_name in pic_name_list:
#     file_name_last = file_name.split('.')[0]  # 去除后缀，得到文件名
#     file_name_all = picpath + '/' + file_name   #图片文件路径
#     print(file_name)
#     folder = "C:/Users/hitzo/Desktop/dataset_bound/" + file_name_last  #bound文件路径
#     #os.makedirs(folder) #创建bound文件夹
#     shutil.copy(file_name_all , folder)  #添加图片到bound文件夹内/

#===========原图和mask图片重合了，改一下mask图片名==============
# maskpath = "C:/Users/hitzo/Desktop/img"
# mask_name_list = os.listdir(maskpath)
# for file_name in mask_name_list:
#     file_name_last = file_name.split('.')[0]  # 去除后缀，得到文件名
#     file_name_old = maskpath + '/' + file_name
#     file_name_new = 'C:/Users/hitzo/Desktop/dataset_bound/' + file_name_last + '/img.jpg'
#     os.rename(file_name_old, file_name_new)
a = 0
# #===========添加mask文件到bound文件夹内==============
# for file_name in label_name_list:
#     file_name_last = file_name.split('.')[0]  # 去除后缀，得到文件名
#     file_name_all = picpath + '/' + file_name   #图片文件路径
#     print(file_name)
#     folder = "C:/Users/hitzo/Desktop/dataset_bound/" + file_name_last  #bound文件路径
#     shutil.copy(file_name_all , folder)  #添加图片到bound文件夹内

# pathAll = 'C:/Users/hitzo/Desktop/drone_dataset/train/img'
# path1 = 'C:/Users/hitzo/Desktop/drone_dataset/fields'
# path2 = 'C:/Users/hitzo/Desktop/drone_dataset/mountain'
# path3 = 'C:/Users/hitzo/Desktop/drone_dataset/sky'
# path4 = 'C:/Users/hitzo/Desktop/drone_dataset/playground'
# path5 = 'C:/Users/hitzo/Desktop/drone_dataset/urban'
pathAll = 'C:/Users/hitzo/Desktop/datami/datamini'
path1 = 'C:/Users/hitzo/Desktop/datami/field'
path2 = 'C:/Users/hitzo/Desktop/datami/mountains'
path3 = 'C:/Users/hitzo/Desktop/datami/sky'
path4 = 'C:/Users/hitzo/Desktop/datami/playground'
path5 = 'C:/Users/hitzo/Desktop/datami/urban'
name_list_All = os.listdir(pathAll)
name_list_1 = os.listdir(path1)
name_list_2 = os.listdir(path2)
name_list_3 = os.listdir(path3)
name_list_4 = os.listdir(path4)
name_list_5 = os.listdir(path5)

for file_name in name_list_3:
    if (file_name in name_list_2) | (file_name in name_list_1) | (file_name in name_list_4) | (file_name in name_list_5):
        print(file_name)
        a += 1
print(a)
a = 0

# for file_name in name_list_1:
#     if (file_name not in name_list_All):
#         print(file_name)
#         a += 1
# for file_name in name_list_2:
#     if (file_name not in name_list_All):
#         print(file_name)
#         a += 1
# for file_name in name_list_3:
#     if (file_name not in name_list_All):
#         print(file_name)
#         a += 1
# for file_name in name_list_4:
#     if (file_name not in name_list_All):
#         print(file_name)
#         a += 1
# for file_name in name_list_5:
#     if (file_name not in name_list_All):
#         print(file_name)
#         a += 1
# print(a)

# pathpic = 'C:/Users/hitzo/Desktop/drone_dataset/train/img'
# pathmask = 'C:/Users/hitzo/Desktop/drone_dataset/train/train_labels'
# name_list_pic = os.listdir(pathpic)
# name_list_mask = os.listdir(pathmask)
# a = 0
# for file_name in name_list_mask:
#     file_name_jpg = file_name.split('.')[0] + '.jpg'
#     if(file_name_jpg not in name_list_pic):
#         fullname = pathmask + '/' + file_name
#         # fullname = file_name.split('.')[1]
#         # shutil.rmtree(fullname)  # 删除非空文件夹
#         os.remove(fullname)
#         a += 1
# print(a)
