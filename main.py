import os
import argparse
import warnings
import sys

sys.path.append("./components/")
warnings.filterwarnings("ignore")

def logo():
    print("\n/*********************************/")
    print("/---------------------------------/\n")
    print("       WELCOME : Jasper Leo     ")
    print("    Jetson Nano Image Caption Task   ")
    print("       from : dpcas and pytorch   ")
    print("\n/---------------------------------/")
    print("/*********************************/\n")

if __name__ == '__main__':
    logo()
    parser = argparse.ArgumentParser(description= " Captioning Task: Jasper Leo's homework ")
    parser.add_argument('-app', type=int, default = 0,
        help = "please see the main.py to choose the app you want.")
    app_dict = {
        0:"object_detection", 
        1:"caption" # 可添加更多应用，已去除不需要的应用
    }

    args = parser.parse_args()# 解析添加参数
    APP_P = app_dict[args.app]

    if APP_P == "object_detection": # 物体识别任务
        from applications.object_local_app import main_object_x
        cfg_file = "./lib/object_lib/cfg/object.cfg"
        main_object_x(cfg_file)
    if APP_P == "caption": # 图像描述任务
        from applications.caption_local_app import main_caption_x
        cfg_file = "./lib/caption_lib/cfg/caption.cfg"
        main_caption_x(cfg_file)
    print("done here.")
