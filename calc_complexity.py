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
from enum import unique
import os, sys, math
import cv2
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from pygam import LinearGAM, s, f
import scipy
from scipy.stats import linregress

def find_factors(number):
  factors = set()
  for w in range(1, int(math.sqrt(number))+1):
    if number % w == 0:
      factors.add(w)
      factors.add(number // w)
  return factors

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


def find_common_factors(frame):
  a = find_factors(frame.shape[0])
  b = find_factors(frame.shape[1])
  return np.array(sorted(set(a) & set(b)))



# This algorithm is similar to the nearest-neighbor algorightm.
def reduce_size(tf, nsize):    
  gfdims = tf.shape
  msize = int(gfdims[1] / gfdims[0] * nsize)
  i = np.arange(nsize, gfdims[0], nsize)
  j = np.arange(msize, gfdims[1], msize)
  #i = gfdims[0] // nsize
  #j = gfdims[1] // msize
  W = np.array_split(tf, i, axis = 0)  
  zmat = []
  for w in W:
    u = np.array_split(w, j, axis = 1)
    zmat.append(u)
  return zmat, nsize, msize

# Calculate the variance after splitting the image into a 
# 2x2 matrix.
def calculate_variance(tf, nsize):
    L = 2
    V = []
    z1 = []
    z3 = []
    
    zmat, nsize, msize = reduce_size(tf, nsize)
    osize1 = len(zmat)
    osize2 = osize1 // 2
    for x in zmat:
        for y in x:
            z1.append(np.median(y))
    z2 = np.reshape(z1, (osize1, osize1))
    W = np.array_split(z2, osize2, axis = 0)
    for w in W:
      u = np.array_split(w, osize2, axis = 1)
      z3.append(u)
    for x in z3:
        for y in x:
            V.append(np.var(y))    
    V = L**2 * np.sum(V) / nsize / msize
    S = L * tf.shape[0] / nsize
    return V, S



# Calculate delta-entropy
# The offset should be at least 3
def calculate_delta_gray(gf, offset = 3):
  J = gf.shape[0] - offset
  tmp = []
  X = [] 
    
  for j in range(0, J):
    tmp = np.subtract(gf[j].astype(float),  gf[j+offset].astype(float))
    tmp = np.abs(tmp).astype(int)
    X.append(tmp)
  X = np.stack(X)
  return X

# For calculations of BW image complexity, see:
# Citation: Zanette DH (2018) Quantifying the complexity of black-and-white images. PLoS ONE 13(11): e0207879. https://doi.org/10.1371/journal.pone.0207879
def calculate_Q(S, V):
    V = np.asarray(V)
    S = np.asarray(S)
    goodvals = V > 0
    V = V[goodvals]
    S = S[goodvals]
    gamout = LinearGAM(s(0, n_splines = 20)).fit(np.log(S), np.log(V))
    dtau = 1/1000000
    Smax = np.max(S)
    Smin = np.min(S)
    X0 = np.linspace(Smin, Smax, num = 100)
    X1 = X0 + dtau
    ds = X0[1] - X0[0]
    y0 = gamout.predict(X0)
    y1 = gamout.predict(X1)       
    dvds = (y1 - y0) / dtau
    #qs = 1 - (dvds**2)/4
    #qs = qs[qs>=0]
    #return np.sum(qs*ds) / (Smax - Smin)
    qs = dvds**2
    return np.sum(qs*ds)



############################################################################################

basepath = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220622"
filelist = os.listdir(basepath)
included_extensions = ['AVI', 'avi']

filelist = [f for f in filelist if any(f.endswith(ext) for ext in included_extensions)]
f1 = filelist[10]
vf = (os.path.join(basepath, f1))

vcap = cv2.VideoCapture(vf)
total_nframes = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
vcap.release()
theframes = get_grayframes(vf, total_nframes)

# total_nframes = 100
#tf2 = calculate_delta_gray(theframes)
tf2 = theframes
common = np.arange(32, tf2.shape[1] // 2, step = 64)
#common = np.arange(32, 1000, step = 64)

vall =[]
sall = []
for tf in tf2:
  vmat = []
  smat = []
  for x in common:    
      a, b = calculate_variance(tf, x)
      vmat.append(a)
      smat.append(b)    
  vall.append(vmat)
  sall.append(smat)

vall = np.asarray(vall)
sall = np.asarray(sall)
Q = []
for i in range(0, total_nframes-2):    
    Q.append(calculate_Q(sall[i], vall[i]))


vval = np.ravel(vall)
sval = np.ravel(sall)
sval = sval[vval>0]
vval = vval[vval>0]

fname = "test.png"
m0 = linregress((np.log(sval)), (np.log(vval)))
mx = np.log(np.unique(sval))
my = m0.intercept + m0.slope * mx
mt = f"Slope {m0.slope: 0.2f}"

fig, ax = plt.subplots(2, figsize = (10,10))
ax[0].scatter(np.log(sall), np.log(vall))
ax[0].plot(mx, my, "r")
ax[0].set_title(mt)
ax[1].plot(range(0, len(Q)), Q)
ax[0].set(xlabel = "S", ylabel = "V")
ax[1].set(xlabel = "tau", ylabel = "Q")
plt.savefig(fname)  
plt.close()
