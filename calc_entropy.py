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
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from multiprocessing.pool import Pool

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
  length_x = len(np.ravel(x))
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

def calculate_entropy_of_frame(gf, N):
  # First dimension is height, second dimension is width
  gfdims = gf.shape
  ht = gfdims[0] // N
  wt = gfdims[1] // N  
  xdir = np.arange(wt, gfdims[1], wt)
  ydir = np.arange(ht, gfdims[0], ht)
  zmat = []
  U = []
  W = np.array_split(gf, ydir, axis = 0)
  for w in W:
    u = np.array_split(w, xdir, axis = 1)  
    U.append(u)
  
  for u1 in U:
    for u2 in u1:
      zmat.append(entropy(u2))
  return np.reshape(zmat, (N,N))

## Calculate the grayframes
def get_grayframes(file, max_frames):
  vcap = cv2.VideoCapture(file)
  fheight = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  
  if fheight == 2160:
    crop_line = 2040
  elif fheight == 1520:
    crop_line = 1344
  
  i = 0
  X = []
  while True:
    (ret, frame) = vcap.read()    
    if not ret:
      break
    if i > max_frames:
      break
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    X.append(grayframe[0:crop_line, :])
    i = i + 1
  vcap.release()
  X = np.stack(X)
  return X

# Calculate delta-entropy
# The offset should be at least 3
def calculate_delta_entropy(gf, offset = 3):
  J = gf.shape[0] - offset
  tmp = []
  X = []
  N = 6
    
  for j in range(0, J):
    tmp = np.subtract(gf[j].astype(float),  gf[j+offset].astype(float))
    tmp = np.add(tmp, 255).astype(int)
    X.append(calculate_entropy_of_frame(tmp, N))
  X = np.stack(X)
  return X

# Calculate the entropy
def calculate_entropy(gf):
  N = 6
  
  J = gf.shape[0]
  X = []
  for j in range(0, J):    
    X.append(calculate_entropy_of_frame(gf[j], N))
  X = np.stack(X)
  return X

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

def build_paths(basepath):
  deltacsvpath = os.path.join(basepath, "delta_entropy_csvdata")
  entrocsvpath = os.path.join(basepath, "entropy_csvdata")
  staticimgpath = os.path.join(basepath, "static_images")
  dynamicimagepath = os.path.join(basepath, "dynamic_images")
  if not os.path.exists(deltacsvpath):
    os.makedirs(deltacsvpath)
  if not os.path.exists(entrocsvpath):
    os.makedirs(entrocsvpath)
  if not os.path.exists(staticimgpath):
    os.makedirs(staticimgpath)
  if not os.path.exists(dynamicimagepath):
    os.makedirs(dynamicimagepath)
  return deltacsvpath, entrocsvpath, staticimgpath, dynamicimagepath

def process_grayframes(theframes, offset = 3, nlevels = 12):
  nframes = theframes.shape[0] # Original number of frames

  check = nframes - (nframes // nlevels) * nlevels
  if check != 0:
    return 0, 0, False

  #ngroup = 12
  #nlevels = nframes // ngroup  
  nlevels = 12 # Sets the number of CPUs to use.
  ngroup = nframes // nlevels
   # Grouping the frames into batches of ngroup.
  # The indx being calculated here is for the delta-entropy, since
  # delta-entropy has a number of frames that is reduced by the offset.
  s0 = np.arange(ngroup, nframes+1, ngroup)-offset
  s0 = np.concatenate([[0], s0])
  s1 = np.arange(ngroup, nframes+1, ngroup)
  indx = np.array([s0[0:-1], s1]).reshape(2,len(s1)).T
  
  theframes2 = []
  for i in indx:
    theframes2.append(theframes[np.arange(i[0],i[1])])
  
  # The grouping of frames for entropy calculations are much simpler.
  theframes3 = []
  for i in np.arange(0, nframes).reshape(nlevels, ngroup):
    theframes3.append(theframes[i])
  
  # Processing the frames in batches is faster than calculating the entropy/frame in batches.
  # Also uses less memory, but it will still consume 200 GB!
  with Pool(processes=nlevels//2) as pool:
    X1 = pool.map(calculate_delta_entropy, theframes2)
  
  with Pool(processes=nlevels//2) as pool:
    X2 = pool.map(calculate_entropy, theframes3)
  
  # Convert the list of frames into a 3d numpy array.
  X1 = np.vstack(X1)
  X2 = np.vstack(X2)
  return X1, X2, True

##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
# CHANGE THE BASEPATH EVERYTIME
overwrite_file = False

basepath = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220725"

filelist = os.listdir(basepath)
deltacsvpath, entrocsvpath, staticimgpath, dynamicimagepath = build_paths(basepath)

included_extensions = ['AVI', 'avi']

filelist = [f for f in filelist if any(f.endswith(ext) for ext in included_extensions)]

starts_with = ['suda']

filelist = [f for f in filelist if any(f.startswith(ext) for ext in starts_with)]

## This portion will run in a for loop.
## Rather than saving to a csv.gzip file, save to a npy file since it is super fast.
for f in filelist:
  start_time = time.time()
  vf = (os.path.join(basepath, f))
  dcf = (os.path.join(deltacsvpath, f.replace(".AVI", "-delta.npy")))
  ecf = (os.path.join(entrocsvpath, f.replace(".AVI", "-entro.npy")))
  spf = (os.path.join(staticimgpath, f.replace(".AVI", ".png")))
  dpf = (os.path.join(dynamicimagepath, f.replace(".AVI", "-animation.mp4")))
  
  # Read all 6000 frames and convert to gray scale.
  # This will take a while.
  
  if os.path.exists(dcf) and os.path.exists(ecf) and not overwrite_file:
    print("Skip for now.")
    # X1 = np.load(dcf)
    # X2 = np.load(ecf)
    # Recalculate the number of frames after calculating the delta-entropy.
    # Value should be equal to nframes - offset
    # nframes, nrows, ncols = X1.shape
    # xval, Xsub = create_lineplotdata(X1)    
    # print("Create static plot.")
    # make_static_plot(X1, X2, Xsub, xval, spf)
    # print("Create dynamic plot.")
    # make_dynamic_plot(X1, X2, Xsub, xval, dpf)
  
  else:
    nlevels = 12
    vcap = cv2.VideoCapture(vf)
    total_nframes = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vcap.release()
    check = total_nframes - (total_nframes // nlevels) * nlevels
    if check == 0:
      print(f"{time.ctime()}")     
      print("Convert {} to grayframe".format(f))    
      theframes = get_grayframes(vf, total_nframes +1)    
      print("Process data.")
      X1, X2, success = process_grayframes(theframes) # X1 is delta-entropy and X2 is entropy
      
      if success:
        print("Save processed data.")
        np.save(dcf, X1)
        np.save(ecf, X2)
      else:
        print("The avi does not have the right frame count, only {}".format(theframes.shape[0]))
    else:
      print("The avi does not have the right frame count, only {}".format(theframes.shape[0]))

  end_time = time.time()
  print("--- %s minutes ---" % ((end_time - start_time)/60))





















basepath = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220726"

filelist = os.listdir(basepath)
deltacsvpath, entrocsvpath, staticimgpath, dynamicimagepath = build_paths(basepath)

included_extensions = ['AVI', 'avi']

filelist = [f for f in filelist if any(f.endswith(ext) for ext in included_extensions)]

starts_with = ['suda']

filelist = [f for f in filelist if any(f.startswith(ext) for ext in starts_with)]

## This portion will run in a for loop.
## Rather than saving to a csv.gzip file, save to a npy file since it is super fast.
for f in filelist:
  start_time = time.time()
  vf = (os.path.join(basepath, f))
  dcf = (os.path.join(deltacsvpath, f.replace(".AVI", "-delta.npy")))
  ecf = (os.path.join(entrocsvpath, f.replace(".AVI", "-entro.npy")))
  spf = (os.path.join(staticimgpath, f.replace(".AVI", ".png")))
  dpf = (os.path.join(dynamicimagepath, f.replace(".AVI", "-animation.mp4")))
  
  # Read all 6000 frames and convert to gray scale.
  # This will take a while.
  
  if os.path.exists(dcf) and os.path.exists(ecf) and not overwrite_file:
    print("Skip for now.")
    # X1 = np.load(dcf)
    # X2 = np.load(ecf)
    # Recalculate the number of frames after calculating the delta-entropy.
    # Value should be equal to nframes - offset
    # nframes, nrows, ncols = X1.shape
    # xval, Xsub = create_lineplotdata(X1)    
    # print("Create static plot.")
    # make_static_plot(X1, X2, Xsub, xval, spf)
    # print("Create dynamic plot.")
    # make_dynamic_plot(X1, X2, Xsub, xval, dpf)
  
  else:
    nlevels = 12
    vcap = cv2.VideoCapture(vf)
    total_nframes = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vcap.release()
    check = total_nframes - (total_nframes // nlevels) * nlevels
    if check == 0:
      print(f"{time.ctime()}")     
      print("Convert {} to grayframe".format(f))    
      theframes = get_grayframes(vf, total_nframes +1)    
      print("Process data.")
      X1, X2, success = process_grayframes(theframes) # X1 is delta-entropy and X2 is entropy
      
      if success:
        print("Save processed data.")
        np.save(dcf, X1)
        np.save(ecf, X2)
      else:
        print("The avi does not have the right frame count, only {}".format(theframes.shape[0]))
    else:
      print("The avi does not have the right frame count, only {}".format(theframes.shape[0]))

  end_time = time.time()
  print("--- %s minutes ---" % ((end_time - start_time)/60))
























basepath = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220727"

filelist = os.listdir(basepath)
deltacsvpath, entrocsvpath, staticimgpath, dynamicimagepath = build_paths(basepath)

included_extensions = ['AVI', 'avi']

filelist = [f for f in filelist if any(f.endswith(ext) for ext in included_extensions)]

starts_with = ['suda']

filelist = [f for f in filelist if any(f.startswith(ext) for ext in starts_with)]

## This portion will run in a for loop.
## Rather than saving to a csv.gzip file, save to a npy file since it is super fast.
for f in filelist:
  start_time = time.time()
  vf = (os.path.join(basepath, f))
  dcf = (os.path.join(deltacsvpath, f.replace(".AVI", "-delta.npy")))
  ecf = (os.path.join(entrocsvpath, f.replace(".AVI", "-entro.npy")))
  spf = (os.path.join(staticimgpath, f.replace(".AVI", ".png")))
  dpf = (os.path.join(dynamicimagepath, f.replace(".AVI", "-animation.mp4")))
  
  # Read all 6000 frames and convert to gray scale.
  # This will take a while.
  
  if os.path.exists(dcf) and os.path.exists(ecf) and not overwrite_file:
    print("Skip for now.")
    # X1 = np.load(dcf)
    # X2 = np.load(ecf)
    # Recalculate the number of frames after calculating the delta-entropy.
    # Value should be equal to nframes - offset
    # nframes, nrows, ncols = X1.shape
    # xval, Xsub = create_lineplotdata(X1)    
    # print("Create static plot.")
    # make_static_plot(X1, X2, Xsub, xval, spf)
    # print("Create dynamic plot.")
    # make_dynamic_plot(X1, X2, Xsub, xval, dpf)
  
  else:
    nlevels = 12
    vcap = cv2.VideoCapture(vf)
    total_nframes = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vcap.release()
    check = total_nframes - (total_nframes // nlevels) * nlevels
    if check == 0:
      print(f"{time.ctime()}")     
      print("Convert {} to grayframe".format(f))    
      theframes = get_grayframes(vf, total_nframes +1)    
      print("Process data.")
      X1, X2, success = process_grayframes(theframes) # X1 is delta-entropy and X2 is entropy
      
      if success:
        print("Save processed data.")
        np.save(dcf, X1)
        np.save(ecf, X2)
      else:
        print("The avi does not have the right frame count, only {}".format(theframes.shape[0]))
    else:
      print("The avi does not have the right frame count, only {}".format(theframes.shape[0]))

  end_time = time.time()
  print("--- %s minutes ---" % ((end_time - start_time)/60))
