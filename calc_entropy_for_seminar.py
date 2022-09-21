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
import numpy as np
import pandas as pd
import argparse
# モジュールから特定のサブモジュールを読み込む
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import square
from multiprocessing import Pool
from matplotlib import pyplot as plt
from matplotlib import animation as anim


# Define functions
## Find all the factors of an integer
## 整数のすべての因数を求める
def find_factors(number):
  factors = set()
  for w in range(1, int(math.sqrt(number))+1):
    if number % w == 0:
      factors.add(w)
      factors.add(number // w)
  return factors

## A 4K grayscale image is a 3840×2160 sized matrix.
## Find all of the factors of a dimension.
## Then return the common factor for two dimensions.
## 4K グレースケールの画像は 3840×2160 行列です。
## 軸ごとのすべての因数を求め、共通する因数を返す。
def find_common_factors(frame):
  a = find_factors(frame.shape[0])
  b = find_factors(frame.shape[1])
  return np.array(sorted(set(a) & set(b)))

## Calculate the entropy
## エントロピーをもとめる
def entropy(x):
  length_x = len(x)
  value, counts = np.unique(x, return_counts = True)
  px = counts / length_x
  if np.count_nonzero(px) <= 1:
    return 0
  return -np.sum(np.multiply(px, np.log2(px)))

## Calcuate the entropy across the frame.
## The entropy is the average value of a 
## NxN sub-matrix. The value of N can be found
## by examining the output of find_common_factors().
## フレーム毎のエントロピーを求める。
## エントロピーは N×N 行列の平均値です。
## Nは find_common_factors() から求めてましょう。
def calculate_entropy_of_frame(gf, N = 20):
  # N = 20
  gfdims = gf.shape
  ht = gfdims[0] // N
  wt = gfdims[1] // N
  
  x = list(range(0, gfdims[0]))
  y = list(range(0, gfdims[1]))
  x = np.reshape(x, (ht, N))
  y = np.reshape(y, (wt, N))
  
  zmat = np.zeros((ht, wt))
  
  for i in range(0, ht):
    for j in range(0, wt):
      z = gf[x[i],:]
      z = z[:, y[j]]
      z = np.ravel(z)
      zmat[i, j] = entropy(z)
  return zmat

## Calculate the entropy of a video.
## CamGear version is 10% faster than OpenCV.
## 動画のエントロピーを求める。CamGear は OpenCV
## とり 10% 程度早い。
# def calculate_entropy(file, N = 20):
#   vcap = CamGear(source = file).start()
#   X = []
#   while True:
#     frame = vcap.read()
#     if frame is None:
#       break
#     grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     X.append(calculate_entropy_of_frame(grayframe, N = N))
#   X = np.stack(X)
#   vcap.stop()
#   return X
# 
## Calculate the entropy of a video.
## OpenCV version.
## 動画のエントロピーを OpenCV で求める。
def calculate_entropy_cv2(file, N = 20):
  vcap = cv2.VideoCapture(file)
  X1 = []
  X2 = []
  X = []
  while True:
    (ret, frame1) = vcap.read()
    (ret, frame2) = vcap.read()
    if not ret:
      break
    grayframe1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    grayframe2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    grayframe1 = grayframe1[0:2040, :]
    grayframe2 = grayframe2[0:2040, :]
    
    grayframe1 = grayframe1.astype(float)
    grayframe2 = grayframe2.astype(float)
    grayframe = np.subtract(grayframe1, grayframe2)
    grayframe = np.add(grayframe, 255).astype(int)
    X.append(calculate_entropy_of_frame(grayframe, N = N))
    X1.append(calculate_entropy_of_frame(grayframe1, N = N))
    X2.append(calculate_entropy_of_frame(grayframe2, N = N))
  X = np.stack(X)
  X1 = np.stack(X1)
  X2 = np.stack(X2)
  vcap.release()
  return X, X1, X2

## PyAV version
#def calculate_entropy_pyav(file, N = 40):
#  with av.open(file) as container:
#    stream = container.streams.video[0]
#    X = []
#    for frame in container.decode(stream):
#      frame = frame.to_rgb()
#      frame = frame.to_image()
#      grayframe = ImageOps.grayscale(frame)
#      grayframe = np.asarray(grayframe)
#      grayframe = grayframe[0:2040, :]
#      X.append(calculate_entropy_of_frame(grayframe, N))
#  return X

## Save the entropy matrix time-series as 
## compresssed (gzip) csv file.
## For a 6000 frame 4K video, the compressed 
## csv file is greater than 300M.
## エントロピー行列の時系列は圧縮 (gzip) した CSV ファイル
## として保存する。6000フレームの4K動画から出力した
## 圧縮CSVファイルは 300M 以上もします。
def save_data_to_csv(out, path):
  frames = out.shape[0]
  nrows = out.shape[1]
  ncols = out.shape[2]
  tmp = out.ravel()
  tmp = pd.DataFrame(tmp, columns = ["value"])
  
  i = np.repeat(np.arange(0, nrows, 1), ncols)
  i = np.tile(i, frames)
  j = np.tile(np.arange(0, ncols, 1), frames*nrows)
  tau = np.repeat(np.arange(0, frames,1), ncols*nrows)
  tmp.insert(0, "i", i)
  tmp.insert(1, "j", j)
  tmp.insert(0, "tau", tau)
  tmp.to_csv(path, index = False, compression="gzip")


def process_image(video, csvfile, N):
  out, out1, out2 = calculate_entropy_cv2(video, N)
  csvfile.replace(".csv.gz", "-delta.csv.gz")
  save_data_to_csv(out, csvfile)
  csvfile.replace(".csv.gz", "-tau.csv.gz")
  save_data_to_csv(out1, csvfile)


def parallel_process_images(x):
  process_image(x[0], x[1], N = x[2])
  
  
def calculate_runningtime(value, N, CPU):
  switcher = {
    15: 9664 * (N // CPU)/60/60,
    20: 6264 * (N // CPU)/60/60,
    24: 4931 * (N // CPU)/60/60,
    30: 3729 * (N // CPU)/60/60,
    40: 2893 * (N // CPU)/60/60,
    60: 2286 * (N // CPU)/60/60,
    120: 1905 * (N // CPU)/60/60,
  }
  return switcher.get(value, 0)

# Usage
# 使い方

# First set the file name.
video = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220525/suda_02_st01_m33_220525.AVI"
# Next read one frame.
vcap = cv2.VideoCapture(video)
grab, frame = vcap.read()
frame.shape
vcap.release()

# Determine the common factors for the dimensions
N = find_common_factors(frame)
print(N) # The largest common factor is 40
np.array(frame.shape[0]) / np.array(N[N >= 20]) # Number of sub-matricies
np.array(frame.shape[1]) / np.array(N[N >= 20]) # Number of sub-matricies


grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
grayframe = grayframe[0:2040, :]
grayframe.shape

N = 40
gf = grayframe
gfdims = gf.shape
ht = gfdims[0] // N
wt = gfdims[1] // N
  
x = list(range(0, gfdims[0]))
y = list(range(0, gfdims[1]))
x = np.reshape(x, (ht, N))
y = np.reshape(y, (wt, N))
x.shape  

zmat = np.zeros((ht, wt))
zmat.shape
i = 0
j = 1
for i in range(0, ht):
  for j in range(0, wt):
    z = gf[x[i],:]
    z.shape
    z = z[:, y[j]]
    z.shape
    z = np.ravel(z)
    zmat[i, j] = entropy(z)

x = z
length_x = len(x)
value, counts = np.unique(x, return_counts = True)
px = counts / length_x
px.shape
-np.sum(np.multiply(px, np.log2(px)))
      
      

## 処理コード

filehead = 'suda_01'
basepath = '/home/gnishihara/Lab_Data/sudar/movie/arikawa_220525/'
N = 40
CPU = 1
if not os.path.exists(basepath):
  sys.exit("Error provide a valid path.")
  
filelist = os.listdir(basepath)
csvpath = os.path.join(basepath, "csvdata")
  
if not any("AVI" in s for s in filelist):
  sys.exit("There are no avi files in this folder.")

if not os.path.exists(csvpath):
  os.makedirs(csvpath)
  
filenames = []
f = filelist[0]
vf = (os.path.join(basepath, f))
cf = (os.path.join(csvpath, f.replace("AVI", "csv.gz")))
filenames.append([vf, cf, N])

runningtime = calculate_runningtime(N, len(filelist), CPU)
print(f'{len(filelist)} files will take {runningtime:0.3f} hours to process on {CPU} cpu.')

parallel_process_images(filenames[0])

################################################################################
# Processing one frame of one file.
################################################################################

def get_frame(file, frame):
  vcap = cv2.VideoCapture(file)
  FPS = vcap.get(cv2.CAP_PROP_FPS)
  FRAMES = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
  FWIDTH = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
  FHEIGHT = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  vcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
  vcap.get(cv2.CAP_PROP_POS_FRAMES)

  print(f'Video has {FRAMES} frames at {FPS} fps. The width is {FWIDTH}, height is {FHEIGHT}.')
  (ret, frame) = vcap.read()
  grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #grayframe = frame[:, :, 3]
  grayframe = grayframe[0:2040, :]
  vcap.release()
  return grayframe


def calculate_spatial_mean(gf, N = 20):
  # N = 20
  gfdims = gf.shape
  ht = gfdims[0] // N
  wt = gfdims[1] // N
  
  x = list(range(0, gfdims[0]))
  y = list(range(0, gfdims[1]))
  x = np.reshape(x, (ht, N))
  y = np.reshape(y, (wt, N))
  
  zmat = np.zeros((ht, wt))
  
  for i in range(0, ht):
    for j in range(0, wt):
      z = gf[x[i],:]
      z = z[:, y[j]]
      z = np.ravel(z)
      zmat[i, j] = np.mean(z)
  return zmat

def make_plots(fname, oname, framenumber):
  grayframe = get_frame(fname, framenumber)  
  
  g15 = calculate_entropy_of_frame(grayframe, 15)
  g24 = calculate_entropy_of_frame(grayframe, 24)
  g30 = calculate_entropy_of_frame(grayframe, 30)
  g40 = calculate_entropy_of_frame(grayframe, 40)
  g60 = calculate_entropy_of_frame(grayframe, 60)
  # find_common_factors(g60)
  m15 = calculate_spatial_mean(g15, 8)
  m24 = calculate_spatial_mean(g24, 5)
  m30 = calculate_spatial_mean(g30, 4)
  m40 = calculate_spatial_mean(g40, 3)
  m60 = calculate_spatial_mean(g60, 2)
  
  fig, axs = plt.subplots(5,2, figsize = (9,16), dpi = 600)
  axs[0,0].imshow(m15)
  axs[1,0].imshow(m24)
  axs[2,0].imshow(m30)
  axs[3,0].imshow(m40)
  axs[4,0].imshow(m60)
  axs[0,0].annotate(f'N=15', (0, 0), ha = "left", va = "top", fontsize = 16, color = "white")
  axs[1,0].annotate(f'N=24', (0, 0), ha = "left", va = "top", fontsize = 16, color = "white")
  axs[2,0].annotate(f'N=30', (0, 0), ha = "left", va = "top", fontsize = 16, color = "white")
  axs[3,0].annotate(f'N=40', (0, 0), ha = "left", va = "top", fontsize = 16, color = "white")
  axs[4,0].annotate(f'N=60', (0, 0), ha = "left", va = "top", fontsize = 16, color = "white")
  axs[0,1].imshow(g15)
  axs[1,1].imshow(g24)
  axs[2,1].imshow(g30)
  axs[3,1].imshow(g40)
  axs[4,1].imshow(g60)
  
  plt.savefig(oname)


def make_del_plots(fname, oname, framenumber):
  grayframe = get_frame(fname, framenumber)  
  grayframe2 = get_frame(fname, framenumber+1)  
  grayframe = (grayframe2-grayframe+255) / 2
  grayframe.astype(np.uint8)

  g15 = calculate_entropy_of_frame(grayframe, 15)
  g24 = calculate_entropy_of_frame(grayframe, 24)
  g30 = calculate_entropy_of_frame(grayframe, 30)
  g40 = calculate_entropy_of_frame(grayframe, 40)
  g60 = calculate_entropy_of_frame(grayframe, 60)
  # find_common_factors(g60)
  m15 = calculate_spatial_mean(g15, 8)
  m24 = calculate_spatial_mean(g24, 5)
  m30 = calculate_spatial_mean(g30, 4)
  m40 = calculate_spatial_mean(g40, 3)
  m60 = calculate_spatial_mean(g60, 2)
  
  fig, axs = plt.subplots(5,2, figsize = (9,16), dpi = 600)
  axs[0,0].imshow(m15)
  axs[1,0].imshow(m24)
  axs[2,0].imshow(m30)
  axs[3,0].imshow(m40)
  axs[4,0].imshow(m60)
  axs[0,0].annotate(f'N=15', (0, 0), ha = "left", va = "top", fontsize = 16, color = "white")
  axs[1,0].annotate(f'N=24', (0, 0), ha = "left", va = "top", fontsize = 16, color = "white")
  axs[2,0].annotate(f'N=30', (0, 0), ha = "left", va = "top", fontsize = 16, color = "white")
  axs[3,0].annotate(f'N=40', (0, 0), ha = "left", va = "top", fontsize = 16, color = "white")
  axs[4,0].annotate(f'N=60', (0, 0), ha = "left", va = "top", fontsize = 16, color = "white")
  axs[0,1].imshow(g15)
  axs[1,1].imshow(g24)
  axs[2,1].imshow(g30)
  axs[3,1].imshow(g40)
  axs[4,1].imshow(g60)
  
  plt.savefig(oname)




file = os.path.join(basepath, filelist[20])



# vcap = cv2.VideoCapture(file)
# X = []
# while True:
#   (ret, frame) = vcap.read()
#   if not ret:
#     break
#   grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#   grayframe = grayframe[0:2040, :]  
#   X.append(grayframe)
# vcap.release()

# Y = np.average(np.array(X),  axis=0)
# Y.astype(np.uint8)



make_del_plots(file, "temp01b.png", 1000)
make_del_plots(file, "temp02b.png", 2000)
make_del_plots(file, "temp03b.png", 3000)
make_del_plots(file, "temp04b.png", 4000)
make_del_plots(file, "temp05b.png", 5000)
