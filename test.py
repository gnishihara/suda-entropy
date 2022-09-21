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
video = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220525/suda_01_st01_m20_220525.AVI"
# Next read one frame.


# framen1 = 2001 # 1002 番目のフレーム（コマ）
# framen2 = 2003 # 1003 番目のフレーム（コマ）
framen1 = 101 # 1002 番目のフレーム（コマ）
framen2 = 105 # 1003 番目のフレーム（コマ）

vcap = cv2.VideoCapture(video)
vcap.set(cv2.CAP_PROP_POS_FRAMES, framen1)
vcap.get(cv2.CAP_PROP_POS_FRAMES)
grab, frame1 = vcap.read()
vcap.set(cv2.CAP_PROP_POS_FRAMES, framen2)
vcap.get(cv2.CAP_PROP_POS_FRAMES)
grab, frame2 = vcap.read()
vcap.release()

N = 40

def find_factors(number):
  factors = set()
  for w in range(1, int(math.sqrt(number))+1):
    if number % w == 0:
      factors.add(w)
      # factors = factors + w
      factors.add(number // w)
  return factors


a = find_factors(2040)
b = find_factors(3840)
set(a) & set(b)



grayframe1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
grayframe1 = grayframe1[0:2040, :]
grayframe2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
grayframe2 = grayframe2[0:2040, :]

grayframe1 = grayframe1.astype(float)
grayframe2 = grayframe2.astype(float)
# grayframe = grayframe1 - grayframe2
grayframe = np.subtract(grayframe1, grayframe2).astype(int)

grayframe.min()
grayframe.max()

gfdims = grayframe.shape
ht = gfdims[0] // N
wt = gfdims[1] // N
  
x = list(range(0, gfdims[0]))
y = list(range(0, gfdims[1]))
x = np.reshape(x, (ht, N))
y = np.reshape(y, (wt, N))

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


fig, axs = plt.subplots(2,2, figsize = (16,9), dpi = 100)
axs[0,0].imshow(grayframe1)
axs[0,1].imshow(grayframe2)
axs[1,0].imshow(grayframe)
axs[1,1].imshow(zmat)
plt.savefig("test.png")
