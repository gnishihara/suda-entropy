import numpy as np
import matplotlib.pyplot as plt
import os
import re

##############################################################################################################
os.listdir("/home/gnishihara/Lab_Data/sudar/movie")

basepath = "/home/gnishihara/Lab_Data/sudar/movie/arikawa_220621"
deltacsvpath = os.path.join(basepath, "delta_entropy_csvdata")
entrocsvpath = os.path.join(basepath, "entropy_csvdata")

filelist1 = os.listdir(deltacsvpath)
filelist2 = os.listdir(entrocsvpath)
included_extensions = ['npy']
filelist1 = [f for f in filelist1 if any(f.endswith(ext) for ext in included_extensions) and re.match]
filelist2 = [f for f in filelist2 if any(f.endswith(ext) for ext in included_extensions)]

figname = "test.png"




st = int(re.findall("(?<=st)[0-9]+" ,filelist1[1])[0])
gr = int(re.findall("(?<=m)[0-9]+" ,filelist1[1])[0])
dt = int(re.findall("[0-9]{6,8}" ,filelist1[1])[0])

delta = []
station = np.zeros(len(filelist2), dtype= np.int64) 
measurement = np.zeros(len(filelist2), dtype= np.int64) 
group =  np.zeros(len(filelist2), dtype = np.int64) 


i = 0
for f in filelist2:
    if re.match("suda_01", f):
      out = os.path.join(entrocsvpath, f)
      station[i] = int(re.findall("(?<=st)[0-9]+" ,f)[0])
      measurement[i] = int(re.findall("(?<=m)[0-9]+" ,f)[0])
      group[i] = int(re.findall("(?<=_)[0-9]+" ,f)[0])
      print(f"{f} {i} {station[i]} {group[i]} {measurement[i]}")    
      delta.append(np.load(out))
      i += 1

len(delta)
measurement[0:len(delta)]
Y1 = []
for f in delta:
    Y1.append(np.median(f))
plt.figure()
plt.plot(measurement, Y1)
plt.show()