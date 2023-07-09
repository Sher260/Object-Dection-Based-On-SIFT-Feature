# -*- coding: utf-8 -*-
import simplejson
import xmltodict
import numpy as np
import cv2
import os

#定义xml转json的函数
def xmltojson(xmlstr, k, filename):#filename为一个个.xml文件
    xmlparse = xmltodict.parse(xmlstr)
    jsonstr = simplejson.dumps(xmlparse,indent=1)
    simplejson_list = simplejson.loads(jsonstr, encoding='utf-8', strict=False)

    mask_dir = 'C:/Users/hitzo/Desktop/dronemask'  # mask保存路径
    image_dir = 'C:/Users/hitzo/Desktop/dronewithsqure'  # 含所需目标的原图保存路径
    imgread_dir = 'C:/Users/hitzo/Desktop/dronepic'  # 原图读取路径
    mask = np.zeros((2160, 3840))

    annotation_objs = simplejson_list['annotation']['object']#取出object信息
    obj_str = 'UAV'  # 所需目标类别
    # 先判断图像中是否有多个目标
    if isinstance(annotation_objs, list):
        len_obj = len(annotation_objs)
        for i in range(len_obj):
            obj_name = annotation_objs[i]['name']
            if obj_name == obj_str:
                k = k+1
                obj_bnd = annotation_objs[i]['bndbox']
                xmin = int(obj_bnd['xmin'])
                ymin = int(obj_bnd['ymin'])
                xmax = int(obj_bnd['xmax'])
                ymax = int(obj_bnd['ymax'])

                mask[ymin:ymax, xmin:xmax] = 1


    else:
        obj_name = annotation_objs['name']
        if obj_name == obj_str:
            k = k + 1
            obj_bnd = annotation_objs['bndbox']
            xmin = int(obj_bnd['xmin'])
            ymin = int(obj_bnd['ymin'])
            xmax = int(obj_bnd['xmax'])
            ymax = int(obj_bnd['ymax'])

            mask[ymin:ymax, xmin:xmax] = 1

    if k != 0:
        #这里主要是拼出mask文件的地址，然后把生成的黑白mask写进去，是边调试边写的所以代码比较乱
        name = os.path.splitext(filename)[0] + '.jpg'#分离文件名与拓展名，得到.ipg的文件
        filename = os.path.splitext(filename)[0]
        #img_rgb = cv2.imread(imgread_dir +name)
        path = os.path.dirname(filename)
        number = filename[len(path):]
        #number = number.split('/')
        mask_name = mask_dir + '/' + number + '.jpg'
        #img_name = image_dir + name
        cv2.imwrite(mask_name, mask*255)
        #cv2.imwrite(img_name, img_rgb)



for filename in os.listdir('C:/Users/hitzo/Desktop/dronelabel'):  # listdir的参数是.xml文件夹的路径
    print(filename)
    f = open('C:/Users/hitzo/Desktop/dronelabel/' + filename)  #读取.xml
    data = f.read()
    k = 0
    xmltojson(data, k, filename)
