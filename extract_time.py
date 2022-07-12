#!/usr/bin/env python3
# 動画のフレーム毎のエントロピーを求めるスクリプトです。
# エントロピーは NxN のサブ配列の平均値です。
# このスクリプトは コマンドラインから実行できる用にしています。
# Python3 script to calculate the mean entropy 
# of each frame in a video sequence. The mean
# entropy is calculated for an NxN subspace.

##########################################################
# Setting up the virtual python environment.
# これは　Linux サーバの場合です。Windows は調べてね。
# 参考：https://python.keicode.com/devenv/virtualenv.php
# (1) virtualenv pyenv
# (2) source pyenv/bin/activate
# (3) which python
# (4) pip install numpy pandas matplotlib opencv-python scikit-image
# (1) は一回すればいい。(2) は必要に応じてやる。(3) は python interpreter の場所
# (4) は module のインストール
##########################################################

# モジュールの読み込み。
# Load python modules
# from vidgear.gears import CamGear # OpenCV is just as fast.
import os, sys, math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import pytesseract
from datetime import datetime

def extract_image(frame, crop_height, crop_width):
  cropped_frame = frame[crop_height,  :, :]
  cropped_frame = cropped_frame[:, crop_width,  :]
  # grayframe = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
  hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
  # Get binary-mask
  msk = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([179, 255, 255]))
  krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
  dlt = cv2.dilate(msk, krn, iterations=1)
  thr = 255 - cv2.bitwise_and(dlt, msk)
  return thr

def get_grayframes(file):
  vcap = cv2.VideoCapture(file)
  fheight = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
  nframes = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))   
  #print(file)
  if fheight == 2160:
    crop_height = range(2040, 2160)
    crop_width = range(2300, 3500)
  elif fheight == 1520:
    crop_height = range(1432, 1520)
    crop_width = range(1600, 2440)
  
  (ret, frame1) = vcap.read()
  vcap.set(cv2.CAP_PROP_POS_FRAMES, nframes-1)
  (ret, frame2) = vcap.read()
  vcap.release()

  frame1 = extract_image(frame1, crop_height, crop_width)
  frame2 = extract_image(frame2, crop_height, crop_width)
  return frame1, frame2, nframes

def run_ocr(textimg):
  d = pytesseract.image_to_string(textimg, config="--psm 7")
  return datetime.strptime(d[0:19], '%Y/%m/%d %H:%M:%S')

basepath = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220621"
filelist = os.listdir(basepath)
included_extensions = ['AVI', 'avi']

filelist = [f for f in filelist if any(f.endswith(ext) for ext in included_extensions)]
gf1 = []
gf2 = []
nframes = []
fname = []
for f in filelist:
    vf = (os.path.join(basepath, f))
    g1, g2, nf = get_grayframes(vf)
    dt1 = run_ocr(g1)
    dt2 = run_ocr(g2)
    gf1.append(dt1)
    gf2.append(dt2)
    nframes.append(nf)
    fname.append(os.path.basename(f))


dataout = {'filename': fname,'starttime': gf1, 'endtime': gf2, 'nframes': nframes}
csvfile = os.path.basename(basepath) + "-datetimes.csv"
vf = (os.path.join(basepath, csvfile))

pd.DataFrame(data = dataout).to_csv(vf, index = False)


