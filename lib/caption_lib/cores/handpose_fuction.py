import cv2
import os
import numpy as np
from hand_keypoints.handpose_x import handpose_x_model,draw_bd_handpose_c
import math
from cores.tracking_utils import tracking_bbox
from cores.hand_pnp import get_hand_pose
from cores.send_msg import send_msg

import numpy as np

# -------------------------------------
# 求解二维向量的角度
def vector_2d_angle(v1,v2):
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_=math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ =65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_

# -------------------------------------
# 获取对应手相关向量的二维角度
def hand_angle(hand_,x=0,y=0):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_['0']['x']+x)- int(hand_['2']['x']+x)),(int(hand_['0']['y']+y)-int(hand_['2']['y']+y))),
        ((int(hand_['3']['x']+x)- int(hand_['4']['x']+x)),(int(hand_['3']['y']+y)- int(hand_['4']['y']+y)))
        )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_['0']['x']+x)-int(hand_['6']['x']+x)),(int(hand_['0']['y']+y)- int(hand_['6']['y']+y))),
        ((int(hand_['7']['x']+x)- int(hand_['8']['x']+x)),(int(hand_['7']['y']+y)- int(hand_['8']['y']+y)))
        )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_['0']['x']+x)- int(hand_['10']['x']+x)),(int(hand_['0']['y']+y)- int(hand_['10']['y']+y))),
        ((int(hand_['11']['x']+x)- int(hand_['12']['x']+x)),(int(hand_['11']['y']+y)- int(hand_['12']['y']+y)))
        )
    angle_list.append(angle_)
    # ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_['0']['x']+x)- int(hand_['14']['x']+x)),(int(hand_['0']['y']+y)- int(hand_['14']['y']+y))),
        ((int(hand_['15']['x']+x)- int(hand_['16']['x']+x)),(int(hand_['15']['y']+y)- int(hand_['16']['y']+y)))
        )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_['0']['x']+x)- int(hand_['18']['x']+x)),(int(hand_['0']['y']+y)- int(hand_['18']['y']+y))),
        ((int(hand_['19']['x']+x)- int(hand_['20']['x']+x)),(int(hand_['19']['y']+y)- int(hand_['20']['y']+y)))
        )
    angle_list.append(angle_)
    return angle_list

# -------------------------------------
# define gesture using x-y surface calculation
# finger id: thumb index middle ring pink
def h_gesture(img, angle_list):
    thr_angle = 65.
    thr_angle_thumb = 53.
    thr_angle_s = 45.
    gesture_str = None
    if 65535. not in angle_list:
        if (angle_list[0]>thr_angle_thumb)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "fist"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "five"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "gun"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "love"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "one"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "six"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]>thr_angle):
            gesture_str = "three"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "thumbup"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "yeah"

        if gesture_str is not None:
            img_gesture = cv2.imread("./materials/vison/gesture_show/{}.jpg".format(gesture_str))
            cv2.rectangle(img, (img.shape[1]-1-152,img.shape[0]-1-152), (img.shape[1]-3,img.shape[0]-3), (255,200,22), 5)
            cv2.rectangle(img, (img.shape[1]-1-152,img.shape[0]-1-152), (img.shape[1]-3,img.shape[0]-3), (25,10,222), 2)
            if gesture_str == "love":
                print('pack and send to server.')
                img_pk = cv2.resize(img_gesture,(150,150))
                img[(img.shape[0]-1-150):(img.shape[0]-1),(img.shape[1]-1-150):(img.shape[1]-1),:] = img_pk
                cv2.putText(img, ' [{}]'.format(gesture_str), (img.shape[1]-1-152,img.shape[0]-1-162),
                            cv2.FONT_HERSHEY_COMPLEX, 1.21, (255, 155, 25),5)
                cv2.putText(img, ' [{}]'.format(gesture_str), (img.shape[1]-1-152,img.shape[0]-1-162),
                            cv2.FONT_HERSHEY_COMPLEX, 1.21, (0, 0, 255))
    return gesture_str

# -------------------------------------
# 手部跟踪算法：采用边界框的IOU方式
def hand_tracking(data,hands_dict,track_index):
    if data is None:
        hands_dict = {}
        track_index = 0
    hands_dict,track_index = tracking_bbox(data,hands_dict,track_index) # 目标跟踪
    return hands_dict,track_index

#-------------------------------------
# 手的21关键点回归检测
# 食指和大拇指的捏和放开判断，即点击（click）判断
# 二维关键点角度约束手势
def handpose_track_keypoints21_pipeline(img,hands_dict,hands_click_dict,track_index,algo_img = None,handpose_model = None,gesture_model = None, icon=None,vis = False,dst_thr = 35,angle_thr = 16.):

    hands_list = []
    gesture_list = []
    if algo_img is not None:

        for idx,id_ in enumerate(sorted(hands_dict.keys(), key=lambda x:x, reverse=False)):

            x_min,y_min,x_max,y_max,score,iou_,cnt_,ui_cnt = hands_dict[id_]

            cv2.putText(img, 'ID {}'.format(id_), (int(x_min+2),int(y_min+15)),cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 0, 0),5)
            cv2.putText(img, 'ID {}'.format(id_), (int(x_min+2),int(y_min+15)),cv2.FONT_HERSHEY_COMPLEX, 0.45, (173255,73))


            # x_min,y_min,x_max,y_max,score = bbox
            w_ = max(abs(x_max-x_min),abs(y_max-y_min))
            if w_< 60:
                continue
            w_ = w_*1.26

            x_mid = (x_max+x_min)/2
            y_mid = (y_max+y_min)/2

            x1,y1,x2,y2 = int(x_mid-w_/2),int(y_mid-w_/2),int(x_mid+w_/2),int(y_mid+w_/2)

            x1 = np.clip(x1,0,img.shape[1]-1)
            x2 = np.clip(x2,0,img.shape[1]-1)

            y1 = np.clip(y1,0,img.shape[0]-1)
            y2 = np.clip(y2,0,img.shape[0]-1)

            bbox_ = x1,y1,x2,y2
            gesture_name  = None
            pts_ = handpose_model.predict(algo_img[y1:y2,x1:x2,:])

            plam_list = []
            pts_hand = {}
            for ptk in range(int(pts_.shape[0]/2)):
                xh = (pts_[ptk*2+0]*float(x2-x1))
                yh = (pts_[ptk*2+1]*float(y2-y1))
                pts_hand[str(ptk)] = {
                    "x":xh,
                    "y":yh,
                    }
                if ptk in [0,1,5,9,13,17]:
                    plam_list.append((xh+x1,yh+y1))
                if ptk == 0: #手掌根部
                    hand_root_ = int(xh+x1),int(yh+y1)
                if ptk == 4: # 大拇指
                    thumb_ = int(xh+x1),int(yh+y1)
                if ptk == 8: # 食指
                    index_ = int(xh+x1),int(yh+y1)
                if vis:
                    if ptk == 0:# 绘制腕关节点
                        cv2.circle(img, (int(xh+x1),int(yh+y1)), 9, (250,60,255),-1)
                        cv2.circle(img, (int(xh+x1),int(yh+y1)), 5, (20,180,255),-1)
                    cv2.circle(img, (int(xh+x1),int(yh+y1)), 4, (255,50,60),-1)
                    cv2.circle(img, (int(xh+x1),int(yh+y1)), 3, (25,160,255),-1)

            # 获得二维角度
            angle_list = hand_angle(pts_hand)
            gesture_ = h_gesture(img,angle_list)
            cv2.putText(img, 'Gesture: {}'.format(gesture_), (int(x_min+2),y2+21),cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 195, 0),5)
            cv2.putText(img, 'Gesture: {}'.format(gesture_), (int(x_min+2),y2+21),cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 150, 255))
            if gesture_ == "love":
                send_msg()

            gesture_list.append(gesture_)
            # print(angle_list)
            # 计算食指和大拇指中心坐标
            choose_pt = (int((index_[0]+thumb_[0])/2),int((index_[1]+thumb_[1])/2))
            # 计算掌心
            plam_list = np.array(plam_list)
            plam_center = (np.mean(plam_list[:,0]),np.mean(plam_list[:,1]))

            # 绘制掌心坐标圆
            if vis:
                cv2.circle(img, (int(plam_center[0]),int(plam_center[1])), 12, (25,160,255),9)
                cv2.circle(img, (int(plam_center[0]),int(plam_center[1])), 12, (255,190,30),2)

            # 计算食指大拇指的距离
            dst = np.sqrt(np.square(thumb_[0]-index_[0]) +np.square(thumb_[1]-index_[1]))
            # 计算大拇指和手指相对手掌根部的角度：
            angle_ = vector_2d_angle((thumb_[0]-hand_root_[0],thumb_[1]-hand_root_[1]),(index_[0]-hand_root_[0],index_[1]-hand_root_[1]))
            # 判断手的点击click状态，即大拇指和食指是否捏合
            click_state = False
            if dst<dst_thr and angle_<angle_thr: # 食指和大拇指的坐标欧氏距离，以及相对手掌根部的相对角度，两个约束关系判断是否点击
                click_state = True
                cv2.circle(img, choose_pt, 6, (0,0,255),-1) # 绘制点击坐标，为轨迹的坐标
                cv2.circle(img, choose_pt, 2, (255,220,30),-1)
                cv2.putText(img, 'Click {:.1f} {:.1f}'.format(dst,angle_), (int(x_min+2),y2-1),cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 0, 0),5)
                cv2.putText(img, 'Click {:.1f} {:.1f}'.format(dst,angle_), (int(x_min+2),y2-1),cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))
            else:
                click_state = False
                cv2.putText(img, 'NONE  {:.1f} {:.1f}'.format(dst,angle_), (int(x_min+2),y2-1),cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 0, 0),5)
                cv2.putText(img, 'NONE  {:.1f} {:.1f}'.format(dst,angle_), (int(x_min+2),y2-1),cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))

            # 记录手的点击（click）计数器，用于判断click稳定输出状态
            if id_ not in hands_click_dict.keys():# 记录手的点击（click）计数器，用于稳定输出
                hands_click_dict[id_] = 0
            if click_state == False:
                hands_click_dict[id_] = 0
            elif click_state == True:
                hands_click_dict[id_] += 1
            hands_list.append((pts_hand,(x1,y1),plam_center,{"id":id_,"click":click_state,"click_cnt":hands_click_dict[id_],"choose_pt":choose_pt})) # 局部21关键点坐标，全局bbox左上坐标，全局掌心坐标
            # 绘制手的关键点连线
            draw_bd_handpose_c(img,pts_hand,x1,y1,2)
        return hands_list,gesture_list

# 图像切割函数
def make_crop(img, algo_img, info_dict, double_en_pts, flag_click_stable):
    crop = None
    state = None

    if (len(double_en_pts) == 2) and (flag_click_stable == True):
        x1,y1 = int(double_en_pts[0][0]),int(double_en_pts[0][1])
        x2,y2 = int(double_en_pts[1][0]),int(double_en_pts[1][1])

        x1_,y1_,x2_,y2_ = min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)
        x1_ = int(np.clip(x1_,0,algo_img.shape[1]-1))
        x2_ = int(np.clip(x2_,0,algo_img.shape[1]-1))
        y1_ = int(np.clip(y1_,0,algo_img.shape[0]-1))
        y2_ = int(np.clip(y2_,0,algo_img.shape[0]-1))

        if info_dict["double_en_pts"] == True:
            if ((x2_-x1_)>0) and ((y2_-y1_)>0):
                crop = cv2.resize(algo_img[y1_:y2_,x1_:x2_,:], (1040, 1040)) # change the size of input image
                state = True
                # max_index,label_msg,score_ = image_caption_model.predict(img_reco_crop)
                # caption = generateCaption(img_reco_crop)
                # reco_msg = {"index":max_index,"label_msg":label_msg,"score":score_}
                # print(" audio_recognize function ->> reco_msg : ", reco_msg)
                # info_dict["caption"] = caption # 加入到info_dict

        if crop is not None: # 绘制方形区域，代表识别截图区域
            print('Found two hands and clicked successfully.')
            h,w,_ = img.shape
            img[(h-1041):(h-1),(w-1041):(w-1),:] = crop
            cv2.rectangle(img, (w-1041,h-1041), (w-1,h-1), (225,66,66), 5)

        info_dict["double_en_pts"] = True

        cv2.rectangle(img, (x1_,y1_), (x2_,y2_), (225,255,62), 5)
        cv2.rectangle(img, (x1_,y1_), (x2_,y2_), (100,180,255), 2)
        # cv2.putText(img, 'captioning:{}'.format(caption), (x1_,y1_), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0),5)
        # cv2.putText(img, 'captioning:{}'.format(caption), (x1_,y1_), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,33,255),1)

    else:
        info_dict["double_en_pts"] = False # 将使能状态重置到0
    return crop, state

# 绘制手click状态时的大拇指和食指中心坐标点轨迹
def draw_click_lines(img,gesture_lines_dict,vis = False):
    if vis:
        for id_ in gesture_lines_dict.keys():
            if len(gesture_lines_dict[id_]["pts"]) >=2:
                for i in range(len(gesture_lines_dict[id_]["pts"])-1):
                    pt1 = gesture_lines_dict[id_]["pts"][i]
                    pt2 = gesture_lines_dict[id_]["pts"][i+1]
                    cv2.line(img,pt1,pt2,gesture_lines_dict[id_]["line_color"],2,cv2.LINE_AA)
