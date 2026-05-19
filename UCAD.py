"""
                             _ooOoo_
                            o8888888o
                            88" . "88
                            (| -_- |)
                            O\  =  /O
                         ____/`---'\____
                       .'  \\|     |//  `.
                      /  \\|||  :  |||//  \\
                     /  _||||| -:- |||||-  \\
                     |   | \\\  -  /// |   |
                     | \_|  ''\---/''  |   |
                     \  .-\__  `-`  ___/-. /
                   ___`. .'  /--.--\  `. . __
                ."" '<  `.___\_<|>_/___.'  >'"".
               | | :  `- \`.;`\ _ /`;.`/ - ` : | |
               \  \ `-.   \_ __\ /__ _/   .-` /  /
          ======`-.____`-.___\_____/___.-`____.-'======
                             `=---='
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     佛祖保佑        永无BUG
            佛曰:
                   写字楼里写字间，写字间里程序员；
                   程序人员写程序，又拿程序换酒钱。
                   酒醒只在网上坐，酒醉还来网下眠；
                   酒醉酒醒日复日，网上网下年复年。
                   但愿老死电脑间，不愿鞠躬老板前；
                   奔驰宝马贵者趣，公交自行程序员。
                   别人笑我忒疯癫，我笑自己命太贱；
                   不见满街漂亮妹，哪个归得程序员？
"""
# -*- coding: utf-8 -*-
from ctypes import *
import numpy as np
import cv2
import paddle.fluid as fluid
from PIL import Image
import paddle

paddle.enable_static()
import sys, os
import torch
import serial
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (non_max_suppression)
from utils.torch_utils import select_device
from multiprocessing import Process, Queue, Value, Event  # 添加了 Value 和 Event
import time
from hex_change import car_drive

model_paths = {
    #今年
    'turn_left': "../model/4.21_Left_1/",  # 6.5_TurnLeft_1_new
    'turn_right': "../model/5.22_Right_1/",
    'cross_road_to_paper_red': "../model/5.22_Crossing_1/",  # 6.5_TurnLeft_1_new
    'paper_red_to_finish': "../model/5.22_Red_1/",
}


def send_drive_commands(vel, angle, sleep_duration):  # send_drive_commands(1550, 1500, 1)
    a.put(1500)  # 推入堆栈
    # v.put(1550)  # 推入堆栈
    v.put(1600)
    try:
        for _ in range(10):
            ser.write(car_drive(vel, angle))             # 将速度和角度 传入函数 time.sleep 是留以足够的时间去让小车执行对应的角度和速度
        time.sleep(sleep_duration)
    except ValueError as e:
        print(e)


def lane(ser, e, limit_10_flag, cancle_10_flag, turn_left_flag, turn_right_flag, dangerous_flag, cross_road_flag,
         paper_red_flag, paper_green_flag, people_flag, warning_sign_flag):
    # vel = 1550
    vel = 1600

    chose_model = 'turn_right'               # 默认选择右转的模型 先让小车动起来
    e.wait()           # 阻塞线程 直到 e.set（）  函数等待 e进程的触发
    if turn_left_flag.value == 1:
        chose_model = 'turn_left'
    elif turn_right_flag.value == 1:
        chose_model = 'turn_right'
    print('正在使用模型：{}', model_paths[chose_model])          # 打印出现在用的模型

    def dataset(frame):
        # # 黄色HSV阈值
        # lower_hsv_yellow = np.array([11, 43, 46])
        # upper_hsv_yellow = np.array([34, 255, 255])
        # # 红色HSV阈值，红色在HSV中跨越0度，需要两个范围
        # lower_hsv_red1 = np.array([0, 100, 100])
        # upper_hsv_red1 = np.array([10, 255, 255])
        # lower_hsv_red2 = np.array([160, 100, 100])
        # upper_hsv_red2 = np.array([180, 255, 255])

        # 脚本调的
        # 黄色HSV阈值（动态调整 @2025-05-22  20:43）
        lower_hsv_yellow = np.array([18, 50, 150])
        upper_hsv_yellow = np.array([40, 255, 255])

        # 红色HSV阈值（双区间）
        lower_hsv_red1 = np.array([0, 80, 80])
        upper_hsv_red1 = np.array([15, 255, 255])
        lower_hsv_red2 = np.array([165, 80, 80])
        upper_hsv_red2 = np.array([180, 255, 255])

        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建黄色和红色的掩码
        mask_yellow = cv2.inRange(hsv, lower_hsv_yellow, upper_hsv_yellow)
        mask_red1 = cv2.inRange(hsv, lower_hsv_red1, upper_hsv_red1)
        mask_red2 = cv2.inRange(hsv, lower_hsv_red2, upper_hsv_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 合并黄色和红色掩码
        mask = cv2.bitwise_or(mask_yellow, mask_red)

        # 将掩码转换为PIL图像，然后调整大小，转换回numpy数组
        img = Image.fromarray(mask)
        img = img.resize((120, 120), Image.ANTIALIAS)
        # img = img.resize((64, 64), Image.ANTIALIAS)
        img = np.array(img).astype(np.float32)
        # cv2.imshow('hsv_img', img)
        # 将灰度图转换为BGR格式
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 转置维度以符合模型输入
        img = img.transpose((2, 0, 1)) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    # 加载模型（切换测试）
    def load_model(model_name, model_paths):
        """
        根据提供的模型名称加载对应的模型。

        Args:
        model_name (str): 模型名称，应该是 model_paths 字典的键。
        model_paths (dict): 包含模型名称和对应路径的字典。

        Returns:
        tuple: 包含加载的模型程序、输入变量名称列表和目标变量列表。
        """
        # 检查模型名称是否有效
        if model_name not in model_paths:
            raise ValueError("Model name not recognized. Please check the model name and paths.")

        # 获取模型路径
        save_path = model_paths[model_name]

        # 创建PaddlePaddle的CPU执行环境
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        # 清理之前的资源，重新初始化
        exe.close()  # 关闭先前的Executor
        exe = fluid.Executor(place)  # 创建新的Executor
        exe.run(fluid.default_startup_program())  # 初始化新的执行环境

        # 加载模型
        [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)

        # 返回加载的模型相关信息
        return exe, infer_program, feeded_var_names, target_var

    save_path = model_paths[chose_model]
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)
    time.sleep(2)
    cap = cv2.VideoCapture('/dev/cam_lane')
    print('打开lane相机')
    prev_frame_time = time.time()  # 上一帧时间（计算帧率用）
    while True:
        # 分段切换更稳定的模型，提高容错率
        if cross_road_flag.value == 1 and (chose_model == 'turn_left' or chose_model == 'turn_right'):
            chose_model = 'cross_road_to_paper_red'  # 切换为下一赛段车道线模型
            try:
                exe, infer_program, feeded_var_names, target_var = load_model(chose_model, model_paths)
                print("Model switched successfully，正在使用模型：{}", model_paths[chose_model])
            except ValueError as e:
                print(e)
        if paper_green_flag.value == 1 and chose_model == 'cross_road_to_paper_red':
            chose_model = 'paper_red_to_finish'  # 切换为下一赛段车道线模型
            try:
                exe, infer_program, feeded_var_names, target_var = load_model(chose_model, model_paths)
                print("Model switched successfully，正在使用模型：{}", model_paths[chose_model])
            except ValueError as e:
                print(e)

        ret, frame = cap.read()
        if ret:
            # 计算帧率
            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time)
            prev_frame_time = current_time
            fps_display = f"FPS: {fps:.2f}"
            cv2.putText(frame, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

            img = dataset(frame)

            if not e.is_set():  # 判断 flag 是否为True
                print("lane进程在等待")
                e.wait()  # flag 为 False 时阻塞
            elif e.is_set():
                result = exe.run(program=infer_program, feed={feeded_var_names[0]: img}, fetch_list=target_var)
                angle = result[0][0][0]
                angle = int(angle + 0.5)
                if chose_model == 'turn_right':
                    print("正在使用参数1")
                    # 这一段是新加的，使得转向更加灵敏
                    # 这一段是新加的，使得转向更加灵敏
                    if angle < 1300:
                        temp = 1500 - angle
                        # temp = temp * 1.5  # 1.14       1550
                        temp = temp * 1.15  # 1.14      #1600
                        angle = 1500 - temp
                    if 1300 < angle < 1400:
                        temp = 1500 - angle
                        # temp = temp * 1.5  # 1.14       1550
                        temp = temp * 1  # 1.14      #1600
                        angle = 1500 - temp
                    if 1400 < angle < 1500:
                        temp = 1500 - angle
                        # temp = temp * 1.5  # 1.14       1550
                        temp = temp * 0.85  # 1.14      #1600
                        angle = 1500 - temp
                    if 1500 < angle < 1600:
                        temp = 1500 - angle
                        # temp = temp * 1.5  # 1.14       1550
                        temp = temp * 0.85  # 1.14      #1600
                        angle = 1500 - temp
                    if 1600 < angle < 1700:
                        temp = 1500 - angle
                        # temp = temp * 1.5  # 1.14       1550
                        temp = temp * 1  # 1.14      #1600
                        angle = 1500 - temp
                    if angle > 1700:
                        temp = angle - 1500
                        # temp = temp * 1.5          1550
                        temp = temp * 1.15  # 1600
                        angle = 1500 + temp
                    angle = int(angle)

                    if angle < 1100:
                        angle = 1100
                    if angle > 1900:
                        angle = 1900

                elif chose_model == 'cross_road_to_paper_red':
                    if limit_10_flag.value == 1 and not cancle_10_flag.value == 1:
                        print("正在使用参数2")
                        # 这一段是新加的，使得转向更加灵敏
                        if angle < 1300:
                            temp = 1500 - angle
                            # temp = temp * 1.5  # 1.14       1550
                            temp = temp * 1.15  # 1.14      #1600
                            angle = 1500 - temp
                        if 1300 < angle < 1400:
                            temp = 1500 - angle
                            # temp = temp * 1.5  # 1.14       1550
                            temp = temp * 1  # 1.14      #1600
                            angle = 1500 - temp
                        if 1400 < angle < 1500:
                            temp = 1500 - angle
                            # temp = temp * 1.5  # 1.14       1550
                            temp = temp * 0.85  # 1.14      #1600
                            angle = 1500 - temp
                        if 1500 < angle < 1600:
                            temp = 1500 - angle
                            # temp = temp * 1.5  # 1.14       1550
                            temp = temp * 0.85  # 1.14      #1600
                            angle = 1500 - temp
                        if 1600 < angle < 1700:
                            temp = 1500 - angle
                            # temp = temp * 1.5  # 1.14       1550
                            temp = temp * 1  # 1.14      #1600
                            angle = 1500 - temp
                        if angle > 1700:
                            temp = angle - 1500
                            # temp = temp * 1.5          1550
                            temp = temp * 1.15  # 1600
                            angle = 1500 + temp
                        angle = int(angle)

                        if angle < 1100:
                            angle = 1100
                        if angle > 1900:
                            angle = 1900
                    else:
                        print("正在使用参数3")
                        # 这一段是新加的，使得转向更加灵敏
                        if angle < 1300:
                            temp = 1500 - angle
                            # temp = temp * 1.5  # 1.14       1550
                            temp = temp * 1.15  # 1.14      #1600
                            angle = 1500 - temp
                        if 1300 < angle < 1400:
                            temp = 1500 - angle
                            # temp = temp * 1.5  # 1.14       1550
                            temp = temp * 1  # 1.14      #1600
                            angle = 1500 - temp
                        if 1400 < angle < 1500:
                            temp = 1500 - angle
                            # temp = temp * 1.5  # 1.14       1550
                            temp = temp * 0.85  # 1.14      #1600
                            angle = 1500 - temp
                        if 1500 < angle < 1600:
                            temp = 1500 - angle
                            # temp = temp * 1.5  # 1.14       1550
                            temp = temp * 0.85  # 1.14      #1600
                            angle = 1500 - temp
                        if 1600 < angle < 1700:
                            temp = 1500 - angle
                            # temp = temp * 1.5  # 1.14       1550
                            temp = temp * 1  # 1.14      #1600
                            angle = 1500 - temp
                        if angle > 1700:
                            temp = angle - 1500
                            # temp = temp * 1.5          1550
                            temp = temp * 1.15  # 1600
                            angle = 1500 + temp
                        angle = int(angle)

                        if angle < 1100:
                            angle = 1100
                        if angle > 1900:
                            angle = 1900
                elif chose_model == 'paper_red_to_finish':
                    print("正在使用参数4")
                    # 这一段是新加的，使得转向更加灵敏
                    if angle < 1300:
                        temp = 1500 - angle
                        # temp = temp * 1.5  # 1.14       1550
                        temp = temp * 1.5  # 1.14      #1600
                        angle = 1500 - temp
                    if 1300 < angle < 1400:
                        temp = 1500 - angle
                        # temp = temp * 1.5  # 1.14       1550
                        temp = temp * 1  # 1.14      #1600
                        angle = 1500 - temp
                    if 1400 < angle < 1500:
                        temp = 1500 - angle
                        # temp = temp * 1.5  # 1.14       1550
                        temp = temp * 0.85  # 1.14      #1600
                        angle = 1500 - temp
                    if 1500 < angle < 1600:
                        temp = 1500 - angle
                        # temp = temp * 1.5  # 1.14       1550
                        temp = temp * 0.85  # 1.14      #1600
                        angle = 1500 - temp
                    if 1600 < angle < 1700:
                        temp = 1500 - angle
                        # temp = temp * 1.5  # 1.14       1550
                        temp = temp * 1  # 1.14      #1600
                        angle = 1500 - temp
                    if angle > 1700:
                        temp = angle - 1500
                        # temp = temp * 1.5          1550
                        temp = temp * 1.5  # 1600
                        angle = 1500 + temp
                    angle = int(angle)

                    if angle < 1100:
                        angle = 1100
                    if angle > 1900:
                        angle = 1900
                if not a.empty():  # 获取队列中的最后一个值并清空队列
                    angle = a.get()
                if not b.empty():
                    b.put(angle)
                if not v.empty():
                    vel = v.get()

                if limit_10_flag.value == 1 and not cancle_10_flag.value == 1:  # 减速
                    # vel = 1585
                    vel = 1540#1535
                    print('当前速度:Vel:',vel)
                elif cancle_10_flag.value == 1 and not paper_green_flag.value == 1:  # 恢复原速
                    # vel = 1550
                    vel = 1600
                    print('当前速度:Vel:', vel)
                elif paper_green_flag.value == 1:  # 绿灯后保持原速
                    # vel = 1550
                    vel = 1600
                    print('当前速度:Vel:', vel)

                # print(f'vel: {vel:<5}, angle: {angle:<5}')
                try:
                    ser.write(car_drive(vel, angle))
                except:
                    print(f'错误的vel: {vel:<5}, angle: {angle:<5}')

            cv2.imshow('lane', frame)
            if cv2.waitKey(1) == 27:
                ser.write(car_drive(1500, 1500))
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(0)
                break
        else:
            print('lane相机打不开,ret:', ret)
def sign(ser, e, limit_10_flag, cancle_10_flag, turn_left_flag, turn_right_flag, dangerous_flag, cross_road_flag,
         paper_red_flag, paper_green_flag, people_flag, warning_sign_flag):
    ConfidenceDegree = 0.6  # 0.52
    label_counters = {key: 0 for key in
                      ['turn_left', 'turn_right', 'cross_road', 'paper_green', 'paper_red', 'change_lanes', 'dangerous',
                       'limit_10', 'cancel_10', 'people', 'warning_sign']}
    # 单独为小人标识写的
    no_detection_counter = 0
    # 标识判断标志位
    change_lanes_flag = 0  # 变道标志
    turn_flag = 0
    # 防止time.sleep()期间目标重复识别
    last_time = time.time()  # 初始化时间戳
    frame_refresh_threshold = 0.3  # 设定时间阈值
    device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    weights = '../model/yolov5_model/best.pt'

    # 加载模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    cap = cv2.VideoCapture('/dev/cam_sign')
    print('打开sign相机')
    prev_frame_time = time.time()  # 上一帧时间（计算帧率用）
    while True:
        ret, image = cap.read()
        # 防止time.sleep()期间目标重复识别
        current_time = time.time()
        time_diff = current_time - last_time

        if ret:
            # 计算帧率
            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time)
            prev_frame_time = current_time
            fps_display = f"FPS: {fps:.2f}"
            cv2.putText(image, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
            # 判断图像更新时间差是否大于阈值
            if time_diff > frame_refresh_threshold:
                print("摄像头可能已卡住，跳过当前帧")
                last_time = current_time
                continue  # 跳过当前循环，不处理这一帧
            last_time = current_time  # 更新时间戳

            with torch.no_grad():
                img = letterbox(image, new_shape=320)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = model(img, augment=False)[0]
                pred = non_max_suppression(pred, 0.4, 0.5, classes=False, agnostic=False)
                f = pred[0].tolist()

                for i, det in enumerate(pred):
                    if len(det) == 0:  # 如果当前图片没有任何检测框
                        print("未检测到任何目标")
                        print("no_detection_counter=%d " % no_detection_counter)
                        # 判断是否之前检测到了小人模型
                        no_detection_counter += 1
                        if no_detection_counter >= 15 and people_flag.value == 1:
                            print("连续 %d 帧未检测到目标，且小人已被拿走，启动小车" % no_detection_counter)
                            print("小人模型已被拿走，恢复小车运行")
                            e.clear()
                            people_flag.value = 0  # 清除小人标志
                            send_drive_commands(1530, 1500, 0.1)  # 发送启动指令
                            print("启动！！！！！！")
                            no_detection_counter = 0
                            e.set()
                        continue  # 跳过后续逻辑，继续处理下一张图片
                    for *xyxy, conf, cls in reversed(det):
                        coor = []
                        label = names[int(cls)]
                        conf = round(float(conf), 2)
                        if conf >= ConfidenceDegree:
                            # 标识计数
                            label_counters[label] += 1
                            for key in label_counters:
                                if key != label:
                                    label_counters[key] = 0

                            for i in xyxy:
                                i = i.tolist()
                                i = int(i)
                                coor.append(i)
                            cv2.rectangle(image, (int(coor[0] * 2), int(coor[1] * 2)),
                                          (int(coor[2] * 2), int(coor[3] * 2)),
                                          (0, 255, 0), 7)
                            # 如果面积达到阈值，则更新label
                            if ((coor[2] - coor[0]) * (coor[3] - coor[1])) > 150:
                                area = (coor[2] - coor[0]) * (coor[3] - coor[1])
                                label = str(label)
                                print(f'识别到：Label: {label:<5}, Area: {area:<5},conf:{conf:<5}')
                            '''
                            # 左右转向备用方案
                            if turn_left_flag.value == 1 and area >= 580 and turn_flag == 0:
                                print("识别到左转，执行左转动作")
                                turn_flag = 1
                                e.clear()
                                send_drive_commands(1510, 1900, 0.9)
                                e.set()
                                time.sleep(5.7)  # 等待弯道走完
                                e.clear()
                                send_drive_commands(1510, 1900, 0.6)
                                e.set()
                            if turn_right_flag.value == 1 and area >= 600 and turn_flag == 0:
                                print("识别到右转，执行右转动作")
                                turn_flag = 1
                                e.clear()
                                send_drive_commands(1510, 1100, 0.9)
                                e.set()
                                time.sleep(5.7)  # 等待弯道走完
                                e.clear()
                                send_drive_commands(1510, 1100, 0.6)
                                e.set()
                            '''
                            People_Flag = False
                            # 左右转向备用方案
                            if turn_left_flag.value == 1 and area >= 580 and turn_flag == 0:
                                print("识别到左转，执行左转动作")
                                turn_flag = 1
                                # e.clear()
                                # send_drive_commands(1510, 1900, 0.9)
                                # e.set()
                                # time.sleep(5.7)  # 等待弯道走完
                                # e.clear()
                                # send_drive_commands(1510, 1900, 0.6)
                                # e.set()

                            if turn_right_flag.value == 1 and area >= 600 and turn_flag == 0:
                                print("识别到右转，执行右转动作")
                                turn_flag = 1
                                # e.set()
                                # 这一段我先注释掉
                                # time.sleep(5.3)  # 等待弯道走完
                                # e.clear()
                                # send_drive_commands(1490, 1500, 1)
                                # e.set()

                            '''
                            # 单圈完成后自动停车
                            if dangerous_flag.value == 1 and area > 600:
                                print(f"area: {area:.2f}")
                                if (label == 'turn_left' or label == 'turn_right') and area >= 1000:
                                    e.clear()
                                    print('单圈完成')
                                    send_drive_commands(1480, 1500, 0)
                                    cv2.destroyAllWindows()
                                    cap.release()
                                    sys.exit(0)
                            '''

                            # 识别到红灯或者变道标识时（即红灯赛段），降低yolo模型识别置信度，提高识别速度，便于调参以实现精准停车
                            if label == 'paper_red' or label == 'change_lanes' or label == 'turn_left' or label == 'turn_right':
                                ConfidenceDegree = 0.45   #0.45
                                # print("当前置信度阈值:", ConfidenceDegree)
                            elif label == 'cross_road':
                                ConfidenceDegree = 0.75
                            elif label == 'people':
                                ConfidenceDegree = 0.65
                            elif label == 'warning_sign':
                                ConfidenceDegree = 0.35
                            elif label == 'dangerous':
                                ConfidenceDegree = 0.35
                            else:
                                ConfidenceDegree = 0.85
                                # print("当前置信度阈值:", ConfidenceDegree)

                            # 标识判断
                            if label == 'turn_left' and turn_left_flag.value == 0 and label_counters[label] >= 6:
                                print('识别到标识——左转，area：', area)
                                turn_left_flag.value = 1
                                e.set()
                            elif label == 'turn_right' and turn_right_flag.value == 0 and label_counters[label] >= 6:
                                print('识别到标识——右转，area：', area)
                                turn_right_flag.value = 1
                                e.set()
                            elif label == 'limit_10' and area >= 850 and limit_10_flag.value == 0:
                                print('识别到标识——限速，area：', area)
                                limit_10_flag.value = 1
                                '''
                                    e.clear()
                                    send_drive_commands(1470, 1500, 0.07) # 刹车
                                    send_drive_commands(1540, 1500, 0.7)  # 往前走一点，准备开始右转
                                    send_drive_commands(1550, 1150, 1.5)  # 右转到框内  1170的angle执行不了
                                    send_drive_commands(1540, 1500, 1.5)  # 往前走一点，逃离复杂路况区域
                                    e.set()
                                '''
                            elif label == 'cancel_10' and area >= 500 and cancle_10_flag.value == 0:
                                print('识别到标识——限速取消，area：', area)
                                cancle_10_flag.value = 1
                                # e.clear()
                                # send_drive_commands(1570, 1500, 0.125)  # 这是往前走一点
                                # e.set()
                            elif label == 'paper_red' and area >= 630 and paper_red_flag.value == 0 and cancle_10_flag.value == 1:  # 不行试试1000
                                print('识别到标识——红灯，area：', area)
                                e.clear()
                                paper_red_flag.value = 1
                                send_drive_commands(1480, 1500, 0.1)
                                # send_drive(1470,1500,0.2)
                                # send_drive_commands(1510, 1900, 0.5)
                                # send_drive_commands(1480, 1500, 2)
                            elif label == 'paper_green' and paper_green_flag.value == 0 and paper_red_flag.value == 1 and cancle_10_flag.value == 1:
                                print('识别到标识——绿灯，area：', area)
                                paper_green_flag.value = 1
                                send_drive_commands(1510, 1600, 0.225)  # 这是往前走一点
                                e.set()
                                # send_drive_commands(1510, 1600, 0.425)  # 这是往左边打一点
                                # if change_lanes_flag == 0:
                                #     send_drive_commands(1600, 1500, 0.6)
                                # e.set()
                            elif label == 'change_lanes' and area >= 200 and change_lanes_flag == 0:
                                print('识别到标识——变道，area：', area)
                                change_lanes_flag = 1
                            # elif label == 'dangerous' and area >= 500 and dangerous_flag.value == 0:
                            #     print('识别到标识——危险，area：', area)
                            #     dangerous_flag.value = 1
                                # e.clear()
                                # send_drive_commands(1520, 1600, 0.03)
                                # e.set()
                                # time.sleep(2.5)
                                # e.clear()
                                # send_drive_commands(1520, 1700, 1)
                                # e.set()
                                '''
                                time.sleep(0.5)
                                e.clear()
                                send_drive_commands(1520, 1700, 0.2)
                                e.set()
                                '''
                            elif label == 'warning_sign' and area >= 1000 and warning_sign_flag.value == 0 and label_counters[label] >= 3:
                                print('识别到标识——危险，area：', area)
                                # 这段代码是在危险表示容易冲出去 不能够按着预期的车道线行走
                                # e.clear()
                                # print('要冲出赛道了，area：', area)
                                # send_drive_commands(1540, 1700, 0.4)
                                # e.set()
                                # warning_sign_flag.value = 1


                                # 以下代码是去年不能够顺应车道线走而写死的代码  仅供参考 效果并不是很好
                                # e.clear()
                                # send_drive_commands(1520, 1600, 0.03)
                                # e.set()
                                # time.sleep(2.5)
                                # e.clear()
                                # send_drive_commands(1520, 1700, 1)
                                # e.set()

                            elif label == 'cross_road' and area >= 700 and cross_road_flag.value == 0 and label_counters[label] >= 6:
                                print('识别到标识——人行道，area：', area)
                                e.clear()         # 这一段代码是在小车能够拐进人性横道的时候的逻辑 拐不进来就看418-438行代码
                                cross_road_flag.value = 1
                                send_drive_commands(1470, 1500, 0.3)
                                send_drive_commands(1480, 1500, 1.5)
                                e.set()

                            elif label == 'people' and area >= 30 and people_flag.value == 0 and label_counters[label] >= 1:
                                print('识别到标识——人，area：', area)
                                e.clear()
                                people_flag.value = 1
                                send_drive_commands(1470, 1500, 0.3)
                                send_drive_commands(1480, 1500, 1.5)
                                # send_drive_commands(1450, 1500, 0.7)
                                # ser.write(car_drive(1485, 1500))

                            elif label != 'people' and people_flag.value == 1 and label_counters[label] >= 10:
                                e.clear()
                                print('没看到小人，看到了其他标识符')
                                people_flag.value = 0
                                send_drive_commands(1520, 1500, 0.1)
                                e.set()

                            elif label == 'dangerous' and area >= 6000 and dangerous_flag.value==0 and label_counters[label] >= 2:
                                print('识别到标识——锥桶，area：', area)
                                # dangerous_flag.value = 1
                                # e.clear()         # 这一段代码是在小车能够拐进人性横道的时候的逻辑 拐不进来就看418-438行代码
                                # print('要碰上锥桶了area：', area)
                                # send_drive_commands(1540, 1650, 0.7)
                                # send_drive_commands(1540, 1500, 0.2)
                                # e.set()
                        # else:
                            # if people_flag.value == 1:
                            #     print('小人已经拿走了，恢复小车行动')
                            #     e.clear()
                            #     people_flag.value = 0
                            #     send_drive_commands(1540, 1500, 0.15)
                            #     e.set()
                        del coor

            cv2.imshow('sign', image)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(0)
                break
        else:
            print('sign相机打不开')


if __name__ == '__main__':
    e = Event()
    e.clear()
    a = Queue()  # 角度值队列
    b = Queue()  # 角度值比对队列
    v = Queue()  # 速度值队列

    limit_10_flag = Value('i', 0)
    cancle_10_flag = Value('i', 0)
    turn_left_flag = Value('i', 0)
    turn_right_flag = Value('i', 0)
    dangerous_flag = Value('i', 0)
    cross_road_flag = Value('i', 0)
    paper_red_flag = Value('i', 0)
    paper_green_flag = Value('i', 0)

    # new
    people_flag = Value('i', 0)
    warning_sign_flag = Value('i', 0)
    # dangerous_new_flag = Value('i', 0)

    # 串口初始化
    ser = serial.Serial('/dev/ttyACM0', 38400)
    time.sleep(1)
    print("串口已打开")

    sign_run = Process(target=sign, args=(
        ser, e, limit_10_flag, cancle_10_flag, turn_left_flag, turn_right_flag, dangerous_flag, cross_road_flag,
        paper_red_flag, paper_green_flag, people_flag, warning_sign_flag))
    lane_run = Process(target=lane, args=(
        ser, e, limit_10_flag, cancle_10_flag, turn_left_flag, turn_right_flag, dangerous_flag, cross_road_flag,
        paper_red_flag, paper_green_flag, people_flag, warning_sign_flag))

    sign_run.start()
    lane_run.start()
