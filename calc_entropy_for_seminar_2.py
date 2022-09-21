import os, sys, math
import cv2
import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt

def entropy(x):
  length_x = len(x)
  value, counts = np.unique(x, return_counts = True)
  px = counts / length_x
  if np.count_nonzero(px) <= 1:
    return 0
  return -np.sum(np.multiply(px, np.log2(px)))


# First set the file name.
video = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220525/suda_02_st01_m33_220525.AVI"
# Next read one frame.



vcap = cv2.VideoCapture(video)
framen = 100 # 11 番目のフレーム（コマ）
FPS = vcap.get(cv2.CAP_PROP_FPS)
FRAMES = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
FWIDTH = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
FHEIGHT = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
vcap.set(cv2.CAP_PROP_POS_FRAMES, framen)
vcap.get(cv2.CAP_PROP_POS_FRAMES)
grab, frame = vcap.read()

frame.shape
vcap.release()

N = 40

grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
grayframe = grayframe[0:2040, :]
grayframe.shape


gfdims = grayframe.shape
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
    z = grayframe[x[i],:]
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

zmat 

fig, axs = plt.subplots(2,1, figsize = (9,16), dpi = 100)
axs[0].imshow(grayframe)
axs[1].imshow(zmat)
plt.savefig("test.png")
