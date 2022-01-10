import os
import cv2
import time

from multiprocessing import Process
from multiprocessing import Manager

import cv2
import numpy as np
import random
import datetime
import torch
import json
from PIL import Image

# 手检测库和预训练模型
from hand_detect.yolo_v3_hand import yolo_v3_hand_model
from hand_keypoints.handpose_x import handpose_x_model
from classify_imagenet.imagenet_c import classify_imagenet_model
from image_caption.attention_cap import image_caption_predict
import pytesseract

import sys
sys.path.append("./lib/caption_lib/")

from cores.handpose_fuction import handpose_track_keypoints21_pipeline
from cores.handpose_fuction import hand_tracking, draw_click_lines
from cores.handpose_fuction import make_crop

from utils.utils import parse_data_cfg

# 设置gstreamer 可根据自身情况调整摄像头参数 framerate 21
def gstreamer_pipeline(capture_width=1640, capture_height=1232, display_width=1640, display_height=1232, framerate=21, flip_method=2):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), " # (memory:NVMM)
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        ) 
    )

def caption_x_process(info_dict,config):

    # ctx = torch.multiprocessing.get_context("spawn")
    # pool = ctx.Pool(3)

    # 加载模型
    print("load model component  ...")

    hand_detect_model = yolo_v3_hand_model(conf_thres=float(config["detect_conf_thres"]),nms_thres=float(config["detect_nms_thres"]),
        model_arch = config["detect_model_arch"],model_path = config["detect_model_path"],yolo_anchor_scale = float(config["yolo_anchor_scale"]),
        img_size = float(config["detect_input_size"]))

    handpose_model = handpose_x_model(model_arch = config["handpose_x_model_arch"],model_path = config["handpose_x_model_path"])

    # torch.multiprocessing.set_start_method('spawn') "cuda" if torch.cuda.is_available() else

    device = torch.device("cpu") 

    # Load model
    model_path = './path/to/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    word_map_path = './path/to/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
    beam_size = 5

    checkpoint = torch.load(model_path, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    
    

    # 开启相机
    print("start caption process")
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER) # 开启摄像机
    info_dict["caption_process_ready"] = True #多进程间的开始同步信号

    # 初始化变量
    gesture_lines_dict = {} # 点击使能时的轨迹点
    hands_dict = {} # 手的信息
    hands_click_dict = {} #手的按键信息计数
    track_index = 0 # 跟踪的全局索引

    while True:
        ret, img = cap.read() # 读取相机图像
        img_path = ''
        if ret: # 读取相机图像成功
            algo_img = img.copy()

            hand_bbox = hand_detect_model.predict(img, vis = True) # 检测手，获取手的边界框crop
            hands_dict,track_index = hand_tracking(data = hand_bbox,hands_dict = hands_dict,track_index = track_index) # 手跟踪，目前通过IOU方式进行目标跟踪
            
            # 检测每个手的关键点及相关信息
            handpose_list, gesture_list = handpose_track_keypoints21_pipeline(img, hands_dict = hands_dict, hands_click_dict = hands_click_dict, track_index = track_index, algo_img = algo_img,
                handpose_model = handpose_model, gesture_model = None,
                icon = None, vis = True)
            # print('center:', center)
            # print('handpose_list:', handpose_list)
            # print('gesture_list', gesture_list)
            
            id_list = [] # 获取跟踪到的手ID，可以有多只手
            for i in range(len(handpose_list)):
                _,_,_,dict_ = handpose_list[i]
                id_list.append(dict_["id"]) # 把手的id加入id_list
            # print('id_list:', id_list)
            # print('handpose_list:', handpose_list)
            
            id_del_list = [] # 获取需要删除的手的ID
            for k_ in gesture_lines_dict.keys():
                if k_ not in id_list: # 去除已经跟踪失败的目标手的相关轨迹，相当于手移出摄像头范围
                    id_del_list.append(k_)
            for k_ in id_del_list:
                del gesture_lines_dict[k_] # 轨迹删除
                del hands_click_dict[k_] # 点击信息删除

            # 更新检测手的轨迹信息，手点击使能时的上升沿、下降沿信号
            double_en_pts = []
            for i in range(len(handpose_list)):
                _,_,_,dict_ = handpose_list[i]
                id_ = dict_["id"]
                if dict_["click"]: # 该只手处于点击状态
                    if id_ not in gesture_lines_dict.keys(): # 手不在轨迹中，加入
                        gesture_lines_dict[id_] = {} # 创建字典，存储lines信息
                        gesture_lines_dict[id_]["pts"]=[] # 初始化
                        gesture_lines_dict[id_]["line_color"] = (random.randint(100,255),random.randint(100,255),random.randint(100,255))
                        gesture_lines_dict[id_]["click"] = None
                    # 判断是否上升沿
                    if gesture_lines_dict[id_]["click"] is not None:
                        if gesture_lines_dict[id_]["click"] == False: # 上升沿计数器
                            info_dict["click_up_cnt"] += 1
                    # 获得点击状态
                    gesture_lines_dict[id_]["click"] = True # 将对应手id的click状态设置为True，代表可能触发点击事件
                    # 获得坐标
                    gesture_lines_dict[id_]["pts"].append(dict_["choose_pt"]) # 将坐标载入
                    double_en_pts.append(dict_["choose_pt"]) # 将坐标载入double_en_pts
                else: # 该只手未处于点击状态
                    if id_ not in gesture_lines_dict.keys(): # id本来就不在lines中
                        gesture_lines_dict[id_] = {} # 将该只手加入，初始化
                        gesture_lines_dict[id_]["pts"]=[]
                        gesture_lines_dict[id_]["line_color"] = (random.randint(100,255),random.randint(100,255),random.randint(100,255))
                        gesture_lines_dict[id_]["click"] = None
                    elif id_ in gesture_lines_dict.keys(): # 手已经在轨迹中，但松开了
                        gesture_lines_dict[id_]["pts"]=[] # 需要清除轨迹
                        #判断是否下降沿
                        if gesture_lines_dict[id_]["click"] == True: # 下降沿计数器
                            info_dict["click_dw_cnt"] += 1
                        # 更新点击状态为false
                        gesture_lines_dict[id_]["click"] = False

            # 绘制手click状态时的大拇指和食指中心坐标点轨迹
            draw_click_lines(img, gesture_lines_dict, vis = bool(config["vis_gesture_lines"]))

            # 进行image caption推理过程
            flag_click_stable = 1

            crop, state = make_crop(img, algo_img, info_dict, double_en_pts, flag_click_stable)
            if state:
                # 将手指选区截图保存
                nowtime = datetime.datetime.now()
                img_path = './saved/saved.jpg'
                cv2.imwrite(img_path, crop)

                # 调用caption模型推理预测                  
                result = image_caption_predict(encoder, decoder, img_path, word_map, rev_word_map, beam_size)
                # ocr_result = pytesseract.image_to_string(Image.open('./saved/saved.jpg'))
                if result:
                    # 将结果写入txt文件，以特殊格式保存历史记录
                    cv2.putText(crop, result, (5,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                    cv2.imwrite(img_path, crop)

                    history = 'time:{}, result:{}\n'.format(nowtime, result)
                    with open('./saved/history.txt','w') as f:
                        f.write(history)
                    print('result from image caption:', result)

            cv2.putText(img, 'HandNum:[{}]'.format(len(hand_bbox)), (5,25),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0),5)
            cv2.putText(img, 'HandNum:[{}]'.format(len(hand_bbox)), (5,25),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255))

            cv2.namedWindow("image", 0)
            cv2.resizeWindow("image", 1280, 960) # 此处可调整展示窗口大小
            cv2.imshow("image", img)
           
            if cv2.waitKey(1) == 27:
                info_dict["break"] = True
                break
        else:
            print('camera opening error.')
            break
 
    cap.release()
    cv2.destroyAllWindows()

def main_caption_x(cfg_file):
    config = parse_data_cfg(cfg_file)

    print("\n/---------------------- main_caption_x config ------------------------/\n")
    for k_ in config.keys():
        print("{} : {}".format(k_,config[k_]))
    print("\n/---------------------------------------------------------------------/\n")

    print(" loading caption_x local demo ...")
    g_info_dict = Manager().dict()# 多进程共享字典初始化, 用于多进程间的 key:value 操作
    g_info_dict["caption_process_ready"] = False # 进程间的开启同步信号
    g_info_dict["break"] = False # 进程间的退出同步信号
    g_info_dict["click_up_cnt"] = 0
    g_info_dict["click_dw_cnt"] = 0

    print(" multiprocessing dict key:\n")
    for key_ in g_info_dict.keys():
        print( " -> ",key_)
    print()

    # 初始化各进程
    process_list = []
    t = Process(target=caption_x_process,args=(g_info_dict,config,))
    process_list.append(t)

    for i in range(len(process_list)):
        process_list[i].start()
    for i in range(len(process_list)):
        process_list[i].join() # 设置主线程等待子线程结束
    del process_list
