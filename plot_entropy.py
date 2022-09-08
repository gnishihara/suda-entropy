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
# (2) source venv/bin/activate
# (3) which python
# (4) pip install numpy pandas matplotlib opencv-python scikit-image
# (1) は一回すればいい。(2) は必要に応じてやる。(3) は python interpreter の場所
# (4) は module のインストール
##########################################################

# モジュールの読み込み。
import os, sys, math
import cv2
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np
import time

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

# Calculate the median entropy for each submatrix in a frame.
def calculate_submatrix_average(gf, N1, N2):
  # N = 20
  gfdims = gf.shape
  ht = gfdims[0] // N1
  wt = gfdims[1] // N2
  
  x = list(range(0, gfdims[0]))
  y = list(range(0, gfdims[1]))
  x = np.reshape(x, (ht, N1))
  y = np.reshape(y, (wt, N2))
  
  zmat = np.zeros((ht, wt))
  
  for i in range(0, ht):
    for j in range(0, wt):
      z = gf[x[i],:]
      z = z[:, y[j]]
      z = np.ravel(z)
      zmat[i, j] = np.median(z) 
  return zmat

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

def create_lineplotdata(X1):  
  # Create the frame axis vector
  xval = np.arange(0, X1.shape[0])
  # Calculate the submatrix median values.
  if X1.shape[1] == (2040//20):
    N1 = 17
    N2 = 32
  elif X1.shape[1] == (1344//14):
    N1 = (1344//14) // 6
    N2 = (2688//14) // 6
  # N1 = 17, N2 = 32 creates a 6 x 6 matrix.
  
  Xsub = []
  for i in range(0, X1.shape[0]):
    Xsub.append(calculate_submatrix_average(X1[i], N1 = N1, N2 = N2))
  Xsub = np.stack(Xsub)
  return xval,Xsub

def make_static_plot(X1, X2, Xsub, xval, fname):
  f = Path(fname).stem + ".AVI"
  nframes = X1.shape[0]
  if X1.shape[1] == (2040//20):
    N1 = 17
    N2 = 32
  elif X1.shape[1] == (1344//14):
    N1 = (1344//14) // 6
    N2 = (2688//14) // 6
  ygridval = []
  for z in range(6+1):
    ygridval.append(z * N1)  
  xgridval = []
  for z in range(6+1):
    xgridval.append(z * N2)
  ygval = np.arange(0, N1*(6+1), N1)
  xgval = np.arange(0, N2*(6+1), N2)
  
  fig = plt.figure(figsize = (16, 16), constrained_layout=True)
  fig.suptitle(f + " (frame: " + str(10) + ")")
  subfigs = fig.subfigures(2,1, height_ratios = [1,2])
  axs0 = subfigs[0].subplots(1,2, sharex = "all", sharey = "all")
  axs1 = subfigs[1].subplots(6,6, sharex = "all", sharey = "all")
  
  axs0[0].set_xticks(xgval)
  axs0[0].set_yticks(ygval)
  axs0[1].set_xticks(xgval)
  axs0[1].set_yticks(ygval)
  
  ylim_min = np.min(Xsub).astype(int)
  ylim_max = np.max(Xsub+1).astype(int)

  # Mean entropy data to plot
  #for row in range(6):
  #    for col in range(6):
  #      entropylines, = axs1[row,col].plot([], [])
  
  # Define the x and y limits
  for row in range(6):
    for col in range(6):
      axs1[row, col].set_xlim(0, nframes)
      
  for row in range(6):
    for col in range(6):
      axs1[row, col].set_ylim(ylim_min, ylim_max)
  
  axs0[0].imshow(X1[10])
  axs0[1].imshow(X2[10])
  axs0[0].grid(color = "w", linestyle  = "-", linewidth = 1)
  
  axs0[0].set_title('delta-entropy')
  axs0[1].set_title('entropy')
  subfigs[0].supylabel('y-coordinate')
  subfigs[0].supxlabel('x-coordinate')
  subfigs[1].supylabel('delta-entropy')
  subfigs[1].supxlabel('Frame')
  
  for row in range(6):
    for col in range(6):
      axs1[row,col].plot(xval[0:nframes], Xsub[0:nframes, row, col])
  plt.savefig(fname)  
  plt.close()  

def make_dynamic_plot(X1, X2, Xsub, xval, fname):
  f = Path(fname).stem + ".AVI"
  # Calculate the submatrix median values.
  if X1.shape[1] == (2040//20):
    N1 = 17
    N2 = 32
  elif X1.shape[1] == (1344//14):
    N1 = (1344//14) // 6
    N2 = (2688//14) // 6
  ygval = np.arange(0, N1*(6+1), N1)
  xgval = np.arange(0, N2*(6+1), N2)  
  nframes, nrows, ncols = X1.shape  
  fig = plt.figure(figsize = (9, 9), frameon = True)
  subfigs = fig.subfigures(2,1, height_ratios = [1,2])
  axs0 = subfigs[0].subplots(1,2, sharex = "all", sharey = "all")
  axs1 = subfigs[1].subplots(6,6, sharex = "all", sharey = "all")
  subfigs[0].subplots_adjust(left = 0.1, top = 0.8, right = 0.9, bottom = 0.1,  hspace = 0.2, wspace = 0.2)
  subfigs[1].subplots_adjust(left = 0.1, top = 0.95, right = 0.9, bottom = 0.1, hspace = 0.2, wspace = 0.2)
  maintitle = subfigs[0].suptitle(t = "", y = 0.95)
  subfigs[0].supylabel('y-coordinate')
  subfigs[0].supxlabel('x-coordinate')
  subfigs[1].supylabel('delta-entropy')
  subfigs[1].supxlabel('Frame')
  axs00 = axs0[0].imshow(X1[0], interpolation = "none")
  axs01 = axs0[1].imshow(X2[0], interpolation = "none")
  axs0[0].grid(color = "w", linestyle  = "-", linewidth = 1)
  axs00.set_data(np.zeros(X1[0].shape))
  axs01.set_data(np.zeros(X2[0].shape))
  
  # Define the x and y limits
  axs0[0].set_xticks(xgval)
  axs0[0].set_yticks(ygval)
  axs0[1].set_xticks(xgval)
  axs0[1].set_yticks(ygval)
  axs0[0].set_title('delta-entropy')
  axs0[1].set_title('entropy')
  
  ylim_min = np.min(Xsub).astype(int)
  ylim_max = np.max(Xsub+1).astype(int)

  for row in range(6):
    for col in range(6):
      axs1[row, col].set_xlim(0, nframes)
  for row in range(6):
    for col in range(6):
      axs1[row, col].set_ylim(ylim_min, ylim_max)
  
  AXS = []
  for row in range(6):
    for col in range(6):
      AXS.append(axs1[row,col].plot(0, 0, color = "blue", linewidth = 0.5))
  
  for row in range(6):
      for col in range(6):      
        xvalues = xval[0:0]
        yvalues = Xsub[0:0, row, col]
        axs1[row,col].plot(xvalues, yvalues, color = "blue", linewidth = 0.5)
  
  def init_plot():
    axs0[0].cla()
    axs0[1].cla()
    for row in range(6):
      for col in range(6):
        axs1[row,col].cla()
  
  def update_plot(i):
    maintitle.set_text("{} (frame: {})".format(f, i)) 
    axs00.set_data(X1[i])
    axs01.set_data(X2[i])  
    # Animate lines
    posi = 0
    for row in range(6):
      for col in range(6):
        xvalues = xval[0:i]
        yvalues = Xsub[0:i, row, col]
        AXS[posi][0].set_data(xvalues,yvalues)
        posi = posi + 1     
    return axs00, axs01, AXS, maintitle
  
  anim = FuncAnimation(fig, func = update_plot, frames = np.arange(0, nframes), repeat = False)  
  anim.save(fname, writer =  "ffmpeg", fps = 30)  
  plt.close()

##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
# Change if needed. ##########################################################################################
overwrite_file = False
basepath = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220622"
##############################################################################################################

filelist = os.listdir(basepath)
deltacsvpath, entrocsvpath, staticimagepath, dynamicimagepath = build_paths(basepath)
filelist1 = os.listdir(deltacsvpath)
filelist2 = os.listdir(entrocsvpath)
included_extensions = ['npy']
filelist1 = [f for f in filelist1 if any(f.endswith(ext) for ext in included_extensions)]
filelist2 = [f for f in filelist2 if any(f.endswith(ext) for ext in included_extensions)]

# f1 = filelist1[1]  # For testing only.
# f2 = filelist2[1]  # For testing only.

## Create the dynamic plots ##################################################################################
for f1, f2 in zip(filelist1, filelist2):
  start_time = time.time()
  dcf = (os.path.join(deltacsvpath, f1)) 
  ecf = (os.path.join(entrocsvpath, f2)) 
  spf = os.path.join(staticimagepath, (Path(f1).stem + ".png").replace("-delta", ""))
  dpf = os.path.join(dynamicimagepath, (Path(f1).stem + ".mp4").replace("-delta", "-animation"))
  
  # Read all 6000 frames and convert to gray scale.
  # This will take a while.
  
  if os.path.exists(dcf) and os.path.exists(ecf) and not os.path.exists(dpf) or overwrite_file:
    X1 = np.load(dcf)
    X2 = np.load(ecf)
    # Recalculate the number of frames after calculating the delta-entropy.
    # Value should be equal to nframes - offset
    xval, Xsub = create_lineplotdata(X1)
    print(f"{time.ctime()}")
    print(f"Create dynamic plot using the files {f1} and {f2}.")
    make_dynamic_plot(X1, X2, Xsub, xval, dpf)
    end_time = time.time()
    print(f"========== Processing time: {(end_time - start_time)/60: 0.5f} minutes ==========")
  else:
    print(f"Skip files {f1} and {f2}.")


# f1 = filelist1[20]
# f2 = filelist2[20]
# start_time = time.time()
# dcf = (os.path.join(deltacsvpath, f1)) 
# ecf = (os.path.join(entrocsvpath, f2)) 
# spf = os.path.join(staticimagepath, (Path(f1).stem + ".png").replace("-delta", ""))
# dpf = os.path.join(dynamicimagepath, (Path(f1).stem + ".mp4").replace("-delta", "-animation"))
#  
# X1 = np.load(dcf)
# X2 = np.load(ecf)
# # Recalculate the number of frames after calculating the delta-entropy.
# # Value should be equal to nframes - offset
# xval, Xsub = create_lineplotdata(X1)
# if not os.path.exists(spf) or overwrite_file:
#   print("Create static plot.")
#   make_static_plot(X1, X2, Xsub, xval, spf)
# if not os.path.exists(dpf) or overwrite_file:
#   print("Create dynamic plot.")
#   make_dynamic_plot(X1, X2, Xsub, xval, dpf)
# end_time = time.time()
# print(f"========== Processing time: {(end_time - start_time)/60: 0.5f} minutes ==========")


