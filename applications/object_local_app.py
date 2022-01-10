import os
import cv2
import time

from multiprocessing import Process
from multiprocessing import Manager

import cv2
import numpy as np
import random
import time

# 加载模型组件库
from hand_detect.yolo_v3_hand import yolo_v3_hand_model
from hand_keypoints.handpose_x import handpose_x_model
from classify_imagenet.imagenet_c import classify_imagenet_model

# 加载工具库
import sys
sys.path.append("./lib/object_lib/")

from cores.handpose_fuction import handpose_track_keypoints21_pipeline
from cores.handpose_fuction import hand_tracking,audio_recognize,judge_click_stabel,draw_click_lines
from utils.utils import parse_data_cfg
from playsound import playsound

# 设置gstreamer 可根据自身情况调整摄像头参数
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

def handpose_x_process(info_dict,config):
    # 模型初始化
    print("load model component  ...")
    # yolo v3 手部检测模型初始化 # model_half = config["detect_model_half"],
    hand_detect_model = yolo_v3_hand_model(conf_thres=float(config["detect_conf_thres"]),nms_thres=float(config["detect_nms_thres"]),
        model_arch = config["detect_model_arch"],model_path = config["detect_model_path"],yolo_anchor_scale = float(config["yolo_anchor_scale"]),
        img_size = float(config["detect_input_size"]),
        )
    # handpose_x 21 关键点回归模型初始化
    handpose_model = handpose_x_model(model_arch = config["handpose_x_model_arch"],model_path = config["handpose_x_model_path"])
    #
    gesture_model = None # 目前缺省
    #
    object_recognize_model = classify_imagenet_model(model_arch = config["classify_model_arch"],model_path = config["classify_model_path"],
    num_classes = int(config["classify_model_classify_num"])) # 识别分类模型

    #
    img_reco_crop = None
    
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER) # 开启摄像机
    print("start handpose process.")

    info_dict["handpose_procss_ready"] = True #多进程间的开始同步信号

    gesture_lines_dict = {} # 点击使能时的轨迹点

    hands_dict = {} # 手的信息
    hands_click_dict = {} #手的按键信息计数
    track_index = 0 # 跟踪的全局索引

    while True:
        # if cap.isOpened():
            # print("open camera successfully.")
        ret, img = cap.read()# 读取相机图像
        if ret:# 读取相机图像成功
            # img = cv2.flip(img,-1)
            algo_img = img.copy()
            st_ = time.time()
            #------
            hand_bbox =hand_detect_model.predict(img,vis = True) # 检测手，获取手的边界框

            hands_dict,track_index = hand_tracking(data = hand_bbox,hands_dict = hands_dict,track_index = track_index) # 手跟踪，目前通过IOU方式进行目标跟踪
            # 检测每个手的关键点及相关信息
            handpose_list,gesture_list = handpose_track_keypoints21_pipeline(img,hands_dict = hands_dict,hands_click_dict = hands_click_dict,track_index = track_index,algo_img = algo_img,
                handpose_model = handpose_model,gesture_model = gesture_model,
                icon = None,vis = True)
            et_ = time.time()
            fps_ = 1./(et_-st_+1e-8)
            #------------------------------------------ 跟踪手的 信息维护
            #------------------ 获取跟踪到的手ID
            id_list = []
            for i in range(len(handpose_list)):
                _,_,_,dict_ = handpose_list[i]
                id_list.append(dict_["id"])
            print('id_list-----------')
            print(id_list)
            print('handpose_list-----------')
            print(handpose_list)
            #----------------- 获取需要删除的手ID
            id_del_list = []
            for k_ in gesture_lines_dict.keys():
                if k_ not in id_list:#去除过往已经跟踪失败的目标手的相关轨迹
                    id_del_list.append(k_)
            #----------------- 删除无法跟踪到的手的相关信息
            for k_ in id_del_list:
                del gesture_lines_dict[k_]
                del hands_click_dict[k_]

            #----------------- 更新检测到手的轨迹信息,及手点击使能时的上升沿和下降沿信号
            double_en_pts = []
            for i in range(len(handpose_list)):
                _,_,_,dict_ = handpose_list[i]
                id_ = dict_["id"]
                if dict_["click"]:
                    if  id_ not in gesture_lines_dict.keys():
                        gesture_lines_dict[id_] = {}
                        gesture_lines_dict[id_]["pts"]=[]
                        gesture_lines_dict[id_]["line_color"] = (random.randint(100,255),random.randint(100,255),random.randint(100,255))
                        gesture_lines_dict[id_]["click"] = None
                    #判断是否上升沿
                    if gesture_lines_dict[id_]["click"] is not None:
                        if gesture_lines_dict[id_]["click"] == False:#上升沿计数器
                            info_dict["click_up_cnt"] += 1
                    #获得点击状态
                    gesture_lines_dict[id_]["click"] = True
                    #---获得坐标
                    gesture_lines_dict[id_]["pts"].append(dict_["choose_pt"])
                    double_en_pts.append(dict_["choose_pt"])
                else:
                    if  id_ not in gesture_lines_dict.keys():
                        gesture_lines_dict[id_] = {}
                        gesture_lines_dict[id_]["pts"]=[]
                        gesture_lines_dict[id_]["line_color"] = (random.randint(100,255),random.randint(100,255),random.randint(100,255))
                        gesture_lines_dict[id_]["click"] = None
                    elif  id_ in gesture_lines_dict.keys():

                        gesture_lines_dict[id_]["pts"]=[]# 清除轨迹
                        #判断是否上升沿
                        if gesture_lines_dict[id_]["click"] == True:#下降沿计数器
                            info_dict["click_dw_cnt"] += 1
                        # 更新点击状态
                        gesture_lines_dict[id_]["click"] = False

            #绘制手click 状态时的大拇指和食指中心坐标点轨迹
            draw_click_lines(img,gesture_lines_dict,vis = bool(config["vis_gesture_lines"]))
            #判断各手的click状态是否稳定，且满足设定阈值
            # flag_click_stable = judge_click_stabel(img,handpose_list,int(config["charge_cycle_step"]))
            flag_click_stable = 1
            #判断是否启动识别语音,且进行选中目标识别
            img_reco_crop,reco_msg = audio_recognize(img,algo_img,img_reco_crop,object_recognize_model,info_dict,double_en_pts,flag_click_stable)
            print('reco_msg------------')
            print(reco_msg)
            cv2.putText(img, 'HandNum:[{}]'.format(len(hand_bbox)), (5,25),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0),5)
            cv2.putText(img, 'HandNum:[{}]'.format(len(hand_bbox)), (5,25),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255))

            cv2.namedWindow("image",0)
            cv2.resizeWindow("image", 1280, 960)
            cv2.imshow("image",img)
            if cv2.waitKey(1) == 27:
                info_dict["break"] = True
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def main_object_x(cfg_file):
    config = parse_data_cfg(cfg_file)

    print("\n/---------------------- main_handpose_x config ------------------------/\n")
    for k_ in config.keys():
        print("{} : {}".format(k_,config[k_]))
    print("\n/------------------------------------------------------------------------/\n")

    print(" loading handpose_x local demo ...")
    g_info_dict = Manager().dict()# 多进程共享字典初始化：用于多进程间的 key：value 操作
    g_info_dict["handpose_procss_ready"] = False # 进程间的开启同步信号
    g_info_dict["break"] = False # 进程间的退出同步信号
    g_info_dict["double_en_pts"] = False # 双手选中动作使能信号

    g_info_dict["click_up_cnt"] = 0
    g_info_dict["click_dw_cnt"] = 0

    g_info_dict["reco_msg"] = None

    print(" multiprocessing dict key:\n")
    for key_ in g_info_dict.keys():
        print( " -> ",key_)
    print()

    #-------------------------------------------------- 初始化各进程
    process_list = []
    t = Process(target=handpose_x_process,args=(g_info_dict,config,))
    process_list.append(t)

    for i in range(len(process_list)):
        process_list[i].start()

    for i in range(len(process_list)):
        process_list[i].join()# 设置主线程等待子线程结束

    del process_list
