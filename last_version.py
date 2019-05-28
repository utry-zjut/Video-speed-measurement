#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
'''
   @Author:leedom

   Created on Mon May 27 19:32:46 2019
   Description:
   tip1:整合代码,封装好
   tip2:从视频中读取
   tip3:拼接视频,凑成27s左右的视频文件
   License: (C)Copyright 2019
'''
import csv
import pandas as pd
import cv2
import os
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
def generate_Video():
    path = 'video/logo/'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24.0
    size = (368,640)
    out = cv2.VideoWriter('generateVideo.mp4', fourcc, fps, size)
    pic_name_all = os.listdir(path)
    pic_name_all.sort(key=lambda x:int(x[:-4]))
    #############  读取源数据,拼接第一步  ################
    for pic_name in pic_name_all:
        img_name = path + pic_name
        frame = cv2.imread(img_name)
        frame = cv2.resize(frame, (int(size[0]), int(size[1])))
        out.write(frame)
        print("%s/%s" % (pic_name_all.index(pic_name) + 1, len(pic_name_all)), img_name)
    #############  读取源数据一半,拼接第二步  ################
    count = 0
    for pic_name in pic_name_all:
        count += 1
        if count > len(pic_name_all) / 2:
            break
        img_name = path + pic_name
        frame = cv2.imread(img_name)
        frame = cv2.resize(frame, (int(size[0]), int(size[1])))
        out.write(frame)
        print("%s/%s" % (pic_name_all.index(pic_name) + 1, len(pic_name_all)), img_name)
    out.release()

##########################  任务一:读取视频,测试算法的性能  ##################################
def realTimeMonitor_fromVideo(name):
    #-----  程序所需变量  --------#
    pre_distance = '1'    #为了区分两个连续框中心距,'1'是为了表示第一帧
    pre_center = [-1,-1]  #中心框位置坐标
    pre_value = '---'     #第几帧,该帧的索引,以此来判断
    switch = False        #是否风车转了三分之一圈
    current_value = "---" #过渡变量,主要是为了存帧索引变量
    gap = '---'           #存储转速中间量
    pre_result = -1       #转速的存储,赋不合理初值
    five = []             #五次测算输出一次结果,用five存储
    index = 0             #以此来作为一个视频第几帧的索引
    
    cap = cv2.VideoCapture(name)   #读视频
    ret = True
    while(ret):
        index += 1
        ret,frame = cap.read()
        if ret == False:
            break
        d = Detect(frame)
 
        pre_distance, pre_center, switch,pre_value, current_value, gap, pre_result,five= d.img_process_main(index,pre_distance,pre_center,switch,pre_value,current_value, gap, pre_result, five)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cap.release() 
    cv2.destroyAllWindows()
               
class Detect:
    def __init__(self, img):      
        self.ori_img = img
        self.gray = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2HSV)
        # 获得原始图像行列
        rows, cols = self.ori_img.shape[:2]
        # 工作图像
        self.work_img = cv2.resize(
            self.ori_img, (int(cols), int(rows)))
        self.work_gray = cv2.resize(
            self.gray, (int(cols), int(rows)))
        self.work_hsv = cv2.resize(
            self.hsv, (int(cols), int(rows)))
    '''
    根据HSV上下界进行区分
    '''
    ################################ 颜色区域提取,将RGB转换到HSV空间 ###################################
    def color_area(self):
        #提取黄色区域
        low_yellow = np.array([10,43,46])
        high_yellow = np.array([70,255,255])
        mask_yellow = cv2.inRange(self.work_hsv,low_yellow,high_yellow)
       
        yellow = cv2.bitwise_and(self.work_hsv,self.work_hsv,mask = mask_yellow)

        cv2.imshow('yellow',yellow)
        cv2.waitKey(100)
        return yellow  
 
    ###################### 形态学处理,有效地将目标区域划分出来 ############################################
    def good_thresh_img(self, yellow):
        yellow = cv2.cvtColor(yellow, cv2.COLOR_HSV2BGR)
        yellow = cv2.cvtColor(yellow, cv2.COLOR_BGR2GRAY)

        # 阈值处理
        yellow,yellow_thresh = cv2.threshold(yellow, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #做一些形态学操作,去一些小物体干扰，这是个边缘检测算法
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        #进行腐蚀膨胀，处理后的图片黑白分明，效果明显更优
        img_morph = cv2.dilate(yellow_thresh, se)
        img_morph = cv2.erode(img_morph, se)
        img_morph = cv2.erode(img_morph, se, iterations=1)
        img_morph = cv2.dilate(img_morph, se, iterations=1)
        return img_morph 
    '''
    根据需求,有两种情况
    1.未检测到黄色块
    2.检测到黄色快
    '''
    ########################### 矩形四角点提取,矩形的边缘点以及宽度反馈出来了 ###############################
    def key_points_tap(self, yellow):
        yellow_heights = []
        yellow_points = []
        yellow_widths = []

        yellow_cp = yellow.copy()
        # 按结构树模式找所有轮廓
        cnts, _ = cv2.findContours(yellow_cp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("count:",len(cnts))
        # 按区域大小排序,将所有轮廓存储下来
        flag = 0
        if len(cnts) == 0:
            flag = -1       #未检测到黄色区域
        else:
            if len(cnts) >= 3:
                length = 3
            else:
                length = len(cnts)
            for i in range(length):
                cnt_one =sorted(cnts,key=cv2.contourArea, reverse=True)[i]
                box = cv2.minAreaRect(cnt_one)
                width = box[1][0]
                height = box[1][1]
                yellow_points.append(np.intc(cv2.boxPoints(box)))
                yellow_widths.append(width)
                yellow_heights.append(height)
                flag = 1          
            
        if flag == -1:
            return -1,0,0,flag
        else:
            return yellow_points,yellow_widths,yellow_heights,flag

    ############# 画出关键轮廓的最校外接矩形,该函数的返回值貌似没啥用了，主要的画框已经在内部做完了 ################
    def key_cnt_draw(self, points):
        mask = np.zeros(self.work_gray.shape, np.uint8)
        cv2.drawContours(self.work_img, [points], -1, 255, 2)
        return mask

    ############################## 目标框图像中心点提取 ################################################
    def center_point_cal(self, points):
        pt1_x, pt1_y = points[0, 0], points[0, 1]
        pt3_x, pt3_y = points[2, 0], points[2, 1]
        center_x, center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2
        return center_x, center_y
        
  
    ################################## 算出目标点和四个关键点之间的最短距离 #################################
    def distance(self, points):
        # 得到图像中心
        rows, cols = self.work_img.shape[:2]
        img_center_x, img_center_y = cols / 2, rows / 2
        distance = 0
        for i in range(0, len(points)):
            a, b = points[i, :2]
            now_distance = np.sqrt(
                np.square(a - img_center_x) + np.square(b - img_center_y))
            if i == 0:
                distance = now_distance
                key_x = a
                key_y = b
            elif now_distance < distance:
                distance = now_distance
                key_x = a
                key_y = b
            else:
                pass
        return distance, key_x, key_y

    ############################# 计算两个中心点水平之间的距离 #################################################
    def centerDistance(self,center1,center2):
        # distance = np.sqrt(np.square(center1[0] - center2[0]) + np.square(center1[1] - center2[1]))
        distance = center1[0] - center2[0]
        return distance  

    ######################################## 运行主函数 ###############################################
    def img_process_main(self,index,pre_distance,pre_center, switch,pre_value,current_value, gap,pre_result, five):  
        # 找到红色区域
        yellow = self.color_area()
        # 处理得到一个比较好的二值图
        yellow_morph = self.good_thresh_img(yellow)
        # 获取矩形框的四个关键点
        yellow_points,yellow_width, yellow_height,flag= self.key_points_tap(yellow_morph)
        key_array = []
        error = -1
        #了解点的构造,删除异常的points
        for items in yellow_points: 
            key_array.append(np.mean(items[:,0]))
        mid = np.median(key_array)
        for i in range(len(yellow_points)):
            temp = np.mean(yellow_points[i][:,0])
            if abs(temp - mid) > 20:
                error = i
        if error != -1:
            yellow_points.pop(error)
      

        if flag == -1:       
            current_distance = -1
            yellow_center = [-1,-1]
        else:
            max_x = -1
            max_y = -1
            array1 = []
            array2 = []  
            for i in range(len(yellow_points)):
                for item in yellow_points[i]:
                    array1.append(item[0])
                    array2.append(item[1])
            array1.sort()
            array2.sort()            
            array = []
            x_min = array1[0]
            y_min = array2[0]
            x_max = array1[-1]
            y_max = array2[-1]
            max1 = []
            max1.append(x_min)
            max1.append(y_min)
            array.append(max1)
            max2 = []
            max2.append(x_min)
            max2.append(y_max)
            array.append(max2)
            max4 = []
            max4.append(x_max)
            max4.append(y_max)
            array.append(max4)
            max3 = []
            max3.append(x_max)
            max3.append(y_min)
            array.append(max3)
            width = x_max - x_min
            length = y_max - y_min
            radio = width / length
            
            yellow_center = self.center_point_cal(np.array(array))
            if pre_distance == '1':  #或者这是第一帧
                current_distance = 0
            else:
                current_distance = self.centerDistance(pre_center,yellow_center)

            #----------------------  长宽比逻辑,直接跳过该帧  -----------------------------#
            if(radio < 0.8) or width < 20:
                cv2.imshow('img',self.work_img)
                return pre_distance, pre_center, switch, pre_value, current_value, gap, pre_result, five
            cv2.drawContours(self.work_img, [np.array(array)], -1,(255,0,0),2)
        
        #---------------------------  泽林观点:每5帧得到一个转速  ----------------------------------#
        if pre_distance != '1':
            result = cur_speed(pre_center,yellow_center)
            if result / 0.04 > 0.1:
                result = result / 0.04
            else :
                result = pre_result
            five.append(result)
        else :
            result = '---'

        if len(five) == 5:
            final_result = np.mean(five) 
            five = []
        else :
            final_result = pre_result

        print('zelin:index='+str(index)+'   ,result='+str(final_result))
        if final_result < 0:
            cv2.putText(self.work_img, 'rpm:---', (self.work_img.shape[1]-170,self.work_img.shape[0]-80),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),1)
        else:
            cv2.putText(self.work_img, 'rpm:'+str(final_result * 60), (self.work_img.shape[1]-170,self.work_img.shape[0]-80),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),1)
        
        #---------------------------  csp观点:统计三分之一圈得到一次转速  --------------------------------#
        pre_result = result
        pre_center = yellow_center
        pre_distance = current_distance

        if(switch):
            if(yellow_center[0] < 100):
                switch = False
        else:
            if(yellow_center[0] > 280):
                current_value = index
                flag += 1
                switch = True
                if(current_value == '---'):
                    gap = "---"
                else:
                    if(pre_value != '---'):
                        gap = (1 / ((current_value - pre_value)*0.04*3)) * 60
                    else:
                        gap = "---"
                    pre_value = current_value
        print('index:'+str(index)+'  ,gap='+str(gap))
        cv2.putText(self.work_img, 'rpm:'+str(gap), (self.work_img.shape[1]-170,self.work_img.shape[0]-50),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0),1)
        cv2.imshow('img',self.work_img)
        return pre_distance,pre_center,switch,pre_value,current_value,gap,final_result, five

######################   泽林:那边所需要提供的接口   ###################################        
def cur_speed(pre_center, current_center):
    # 前一个框的中点坐标
    # 当前中心点的坐标

    width = 184
    radius = 160
    first = width - pre_center[0]
    second = width - current_center[0]
    angle1 = math.acos(first/radius)
    angle2 = math.acos(second / radius)
    
    if angle1 > angle2:
        cha = angle1-angle2
    else:
        cha = 2 / 3 * math.pi + angle1 - angle2

    result = ( cha ) / (2 * math.pi)
    return result 

if __name__ == "__main__":
    realTimeMonitor_fromVideo('logo.mp4')
    # generate_Video()