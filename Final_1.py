

import numpy as np
import cv2
import os
import math
import random
import tensorflow as tf
import re
import cfg
import tf_slim as slim

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, concatenate, BatchNormalization, Lambda, Input, multiply, add, ZeroPadding2D, Activation, Layer, MaxPooling2D, Dropout, UpSampling2D
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

from PIL import Image, ImageDraw
from tqdm import tqdm

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from PIL import Image, ImageDraw
from PIL import ImagePath
import pandas as pd
from os import path
import json

import matplotlib.pyplot as plt
import urllib
import time
import pytesseract
import time



# In[4]:

split_rate = 0.1
data_dir = 'Final'
IMG_SIZE = 512
SRK_RATE = 0.1
input_size= 512

feature_layers_range = range(5, 1, -1)
feature_layers_num = len(feature_layers_range)
locked_layers = False




def resize_with_padding(img, points, output_width, output_height):
    div = 1.0 * output_width / output_height
    input_height, input_width, _ = img.shape
    scale = 1.0
    if input_width == div * input_height:
        img = cv2.resize(img, (int(output_width), int(output_height)))
    elif input_width > div * input_height:
        padding = int((input_width / div - input_height) / 2)
        points[0][1] = points[0][1] + padding
        points[1][1] = points[1][1] + padding
        points[2][1] = points[2][1] + padding
        points[3][1] = points[3][1] + padding
        scale = 1.0 * input_width / output_width
        img = cv2.copyMakeBorder(img, padding, int(input_width / div - input_height - padding), 0, 0,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        padding = int((div * input_height - input_width) / 2)
        points[0][0] = points[0][0] + padding
        points[1][0] = points[1][0] + padding
        points[2][0] = points[2][0] + padding
        points[3][0] = points[3][0] + padding
        scale = 1.0 * input_height / output_height
        img = cv2.copyMakeBorder(img, 0, 0, padding, int(input_height * div - input_width - padding),
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (output_width, output_height))
    points = np.array(points) / scale
    return img, points.astype('int')


# In[14]:


from shapely.geometry import Polygon


# In[15]:


def intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


# In[16]:


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8]) / (g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g


# In[17]:


def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]

    return S[keep]


# In[18]:


def la_nms(polys, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)


# In[19]:


def restore_rectangle(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


# In[20]:


def post_process(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.3, nms_thres=0.1):
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4,
                                                         geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    
    # nms part
    
    boxes = la_nms(boxes.astype(np.float64), nms_thres)
    

    if boxes.shape[0] == 0:
        return None,

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes


# In[15]:




def load_text_recognizer(img,boxes_pred):
    pred = []
    blog=[]
    configuration = ("-l eng --oem 1 --psm 7")
    
    temp = ""
    for i in range(boxes_pred.shape[0]):
        x,y,w,h =cv2.boundingRect(np.int32(boxes_pred[i]))
        crop=img[y:y+h+4, x-1:x+w+8]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kernel1 = np.array([[0, -1, 0], 
                   [-1, 5,-1], 
                   [0, -1, 0]])
        crop = cv2.filter2D(crop, -1, kernel1)           
          
        
        
        temp = pytesseract.image_to_string(crop, config=configuration).strip()
        pred.append(temp)
        blog.append(crop)
    pred_ = pred.copy()

    #Clean predictions
    pred = re.sub(r"\n", " ", " ".join(pred))
    pred = re.sub(r"\t", " ", pred)
    pred = re.sub(r"[^0-9a-zA-Z]", " ", pred)
    return pred,pred_,blog



EAST_quant = tf.lite.Interpreter(model_path="/content/drive/MyDrive/case_2_ICDAR_2015/tflite_model_1.tflite",)
EAST_quant.allocate_tensors()


# In[25]:


input_quant = EAST_quant.get_input_details()
output_quant = EAST_quant.get_output_details()


# In[1]:


def load_image(img1):
    
    
    
    
    img_pad,_=resize_with_padding(img1,np.zeros([4, 2]),512,512)
    im=img_pad.copy()
   
    img2 = image.img_to_array(img_pad)
    img_array_pred = preprocess_input(img2)
  
    EAST_quant.set_tensor( input_quant[0]['index'],np.expand_dims(img_array_pred,axis=0).astype(np.float32))
    
    EAST_quant.invoke()
    pred_result=EAST_quant.get_tensor(output_quant[0]['index'])
    scores_pred=pred_result[:, :, :, 0:1]
    geometry_pred=pred_result[:, :, :, 1:6]
    boxes_pred=post_process(scores_pred,geometry_pred)
    boxes_pred = boxes_pred[:, :8].reshape((-1, 4, 2))
    EAST_quant.get_tensor_details
 
    pred1,pred_,blog = load_text_recognizer(im, boxes_pred)
    pred = pred1.lower().split()
   


    cv2.polylines(img_pad, np.int32(boxes_pred), True, (0, 0, 255))
    for i in range( boxes_pred.shape[0]):
        startX,startY,endX, endY =cv2.boundingRect(np.int32(boxes_pred[i]))
        
        cv2.putText(img_pad, pred_[i].upper(), (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)

    return img_pad,blog,pred_
    

