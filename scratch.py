# Playing around with skiimage to see if it is faster.
# There is not much difference.
import os, sys, math
import numpy as np
import pandas as pd
import cv2
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import square
import matplotlib.pyplot as plt
import av
from PIL import Image, ImageOps
import timeit

# 3840 x 2040

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

a = find_factors(3840)
b = find_factors(2040)
np.array(sorted(set(a) & set(b)))

3840 / np.array(sorted(set(a) & set(b)))
2040 / np.array(sorted(set(a) & set(b)))

def myentropy(x):
  length_x = len(x)
  value, counts = np.unique(x, return_counts = True)
  px = counts / length_x
  if np.count_nonzero(px) <= 1:
    return 0
  return -np.sum(np.multiply(px, np.log2(px)))

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
      zmat[i, j] = myentropy(z)
  return zmat

a = find_factors(3160)
b = find_factors(2040)
np.array(sorted(set(a) & set(b)))

video = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220525/suda_02_st01_m33_220525.AVI"

def calculate_entropy_cv2(file, N = 40):
  vcap = cv2.VideoCapture(file)
  X = []
  i = 0
  while True:
    (ret, frame) = vcap.read()
    if not ret:
      break
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayframe = grayframe[0:2040, :]
    X.append(calculate_entropy_of_frame(grayframe, N = N))
    i = i+1
    
    if (i > 100):
        break
  X = np.stack(X)
  vcap.release()
  return X


def calculate_entropy_pyav(file, N = 40):
  with av.open(file) as container:
    outcont = av.open("test.mp4", mode = "w")
    outstream = outcont.add_stream("mpeg4", rate = 20)
    outstream.width = 96*2
    outstream.height = 51*2
    outstream.pix_fmt = "yuv420p"
    stream = container.streams.video[0]
    X = []
    i = 0
    for frame in container.decode(stream):
      frame = frame.to_rgb()
      frame = frame.to_image()
      grayframe = ImageOps.grayscale(frame)
      grayframe = np.asarray(grayframe)
      grayframe = grayframe[0:2040, :]
      eframe = calculate_entropy_of_frame(grayframe, N)
      X.append(eframe)
      eframe = eframe / np.max(eframe)
      eframe = np.round(eframe * 255).astype(np.uint8)
      eframe = np.clip(eframe, 0, 255)
      img = Image.fromarray(eframe)
      outframe = av.VideoFrame.from_image(img)
      for packet in outstream.encode(outframe):
        outcont.mux(packet)
      
      i = i+1
      print(i)
      if (i > 100):
        break
    
    for packet in outstream.encode():
      outcont.mux(packet)
    outcont.close()
  return X


N = [15, 20, 24, 30, 40, 60, 120]
tau = []
for n in N:
  tau2 = timeit.default_timer()
  out = calculate_entropy_cv2(video, n)  
  tau3 = timeit.default_timer()
  tau.append((tau3 - tau2) / 100)

6000 * tau

test = np.ravel(tau)
6000 * test




round(50.2,3)

