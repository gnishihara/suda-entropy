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
  thr = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
  ret, thr = cv2.threshold(thr, 200, 255, cv2.THRESH_BINARY)
  thr = cv2.medianBlur(thr, 3)
  thr = cv2.dilate(thr, None, iterations = 1)
  # Get binary-mask
  thr = cv2.copyMakeBorder(thr, 30,20,20,20, borderType=cv2.BORDER_CONSTANT)
  thr = cv2.rectangle(thr, (300, 0), (360, 165), (0,0,0),-1)
  thr = cv2.rectangle(thr, (470, 0), (535, 165), (0,0,0),-1)
  thr = cv2.rectangle(thr, (855, 0), (900, 165), (0,0,0),-1)
  thr = cv2.rectangle(thr, (1020, 0), (1070, 165), (0,0,0),-1)
  thr = cv2.resize(thr, None, fx = 0.3, fy = 0.3, interpolation=cv2.INTER_LANCZOS4)
  # thr = 255 - cv2.bitwise_and(dlt, msk)
  # thr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
  # thr = cv2.bitwise_not(thr)
  #print(thr)
  return thr

def get_grayframes(file):
  vcap = cv2.VideoCapture(file)
  fheight = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
  nframes = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
  nlevels = 12  
  check = nframes - (nframes // nlevels) * nlevels
  if check != 0:
    return 0, 0, 0, False

  #print(file)
  if fheight == 2160:
    crop_height = range(2045, 2160)
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
  return frame1, frame2, nframes, True

def run_ocr(textimg): 
  d = pytesseract.image_to_string(textimg, config = "-l jpn --psm 3 -c tessedit_char_whitelist=0123456789")
  #return datetime.strptime(d[0:19], '%Y/%m/%d %H:%M:%S')
  s = f"{d[0:4]}/{d[4:6]}/{d[6:8]} {d[8:10]}:{d[10:12]}:{d[12:14]}"
  return datetime.strptime(s, '%Y/%m/%d %H:%M:%S')  

####################################################################################################################################

basepath = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220725"
filelist = os.listdir(basepath)
included_extensions = ['AVI', 'avi']
filelist = [f for f in filelist if any(f.endswith(ext) for ext in included_extensions)]
starts_with = ['suda']
filelist = [f for f in filelist if any(f.startswith(ext) for ext in starts_with)]

gf1 = []
gf2 = []
nframes = []
fname = []
for f in filelist:
    vf = (os.path.join(basepath, f))
    g1, g2, nf, check = get_grayframes(vf)
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




basepath = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220726"
filelist = os.listdir(basepath)
included_extensions = ['AVI', 'avi']

filelist = [f for f in filelist if any(f.endswith(ext) for ext in included_extensions)]
starts_with = ['suda']
filelist = [f for f in filelist if any(f.startswith(ext) for ext in starts_with)]
gf1 = []
gf2 = []
nframes = []
fname = []

for f in filelist:    
    vf = (os.path.join(basepath, f))
    g1, g2, nf, check = get_grayframes(vf)
    print(vf)
    if check:    
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



basepath = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220727"
filelist = os.listdir(basepath)
included_extensions = ['AVI', 'avi']

filelist = [f for f in filelist if any(f.endswith(ext) for ext in included_extensions)]
starts_with = ['suda']
filelist = [f for f in filelist if any(f.startswith(ext) for ext in starts_with)]
gf1 = []
gf2 = []
nframes = []
fname = []

for f in filelist:    
    vf = (os.path.join(basepath, f))
    g1, g2, nf, check = get_grayframes(vf)    
    if check:    
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


basepath = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220728"
filelist = os.listdir(basepath)
included_extensions = ['AVI', 'avi']

filelist = [f for f in filelist if any(f.endswith(ext) for ext in included_extensions)]
starts_with = ['suda']
filelist = [f for f in filelist if any(f.startswith(ext) for ext in starts_with)]
gf1 = []
gf2 = []
nframes = []
fname = []

for f in filelist:    
    vf = (os.path.join(basepath, f))
    g1, g2, nf, check = get_grayframes(vf)    
    if check:    
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





###################################################################################################################################



def extract_image(frame, crop_height, crop_width):
  cropped_frame = frame[crop_height,  :, :]
  cropped_frame = cropped_frame[:, crop_width,  :]
  # grayframe = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
  thr = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
  ret, thr = cv2.threshold(thr, 200, 255, cv2.THRESH_BINARY)
  thr = cv2.medianBlur(thr, 2)
  thr = cv2.dilate(thr, None, iterations = 2)
  # Get binary-mask
  thr = cv2.copyMakeBorder(thr, 30,20,20,20, borderType=cv2.BORDER_CONSTANT)
  thr = cv2.rectangle(thr, (300, 0), (360, 165), (0,0,0),-1)
  thr = cv2.rectangle(thr, (470, 0), (535, 165), (0,0,0),-1)
  thr = cv2.rectangle(thr, (855, 0), (900, 165), (0,0,0),-1)
  thr = cv2.rectangle(thr, (1020, 0), (1070, 165), (0,0,0),-1)
  thr = cv2.resize(thr, None, fx = 0.3, fy = 0.3, interpolation=cv2.INTER_LANCZOS4)
  # thr = 255 - cv2.bitwise_and(dlt, msk)
  # thr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
  # thr = cv2.bitwise_not(thr)
  #print(thr)
  return thr


vf = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220726/suda_02_st1_m18_220726.AVI"

g1, g2, nf, check = get_grayframes(vf)
d1 = pytesseract.image_to_string(g1, config = "--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789 -l jpn")
d2 = pytesseract.image_to_string(g2, config = "--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789 -l jpn")
print(d1)
print(d2)
s1 = f"{d1[0:4]}/{d1[4:6]}/{d1[6:8]} {d1[8:10]}:{d1[10:12]}:{d1[12:14]}"
s2 = f"{d2[0:4]}/{d2[4:6]}/{d2[6:8]} {d2[8:10]}:{d2[10:12]}:{d2[12:14]}"

datetime.strptime(s1, '%Y/%m/%d %H:%M:%S')
datetime.strptime(s2, '%Y/%m/%d %H:%M:%S')





fig, ax = plt.subplots(2)
ax[0].imshow(g1, cmap = "Greys")
ax[1].imshow(g2, cmap = "Greys")
plt.show()






