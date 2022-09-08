# Calculating the entropy of an image
# 2022 June 8


library(tidyverse)
library(imager)
library(furrr)
options(mc.cores = 20)
plan("multisession")

entropy = function(x) {
  x = as.vector(x)
  x = x[x>0]
  -sum(x * log2(x))
}


calc_entropy_matrix = function(img, N = 10) {
  img = as.matrix(img)
  n = dim(img)[1]
  m = dim(img)[2]
  x = matrix(1:n, ncol = N, byrow = T)
  y = matrix(1:m, ncol = N, byrow = T)
  z = matrix(nrow = nrow(x), ncol = nrow(y))
  xn = nrow(x)
  yn = nrow(y)
  
  for(i in 1:xn) {
    for(j in 1:yn) {
      omg = img[x[i,], y[j, ]]
      z[i, j] = entropy(omg)
    }
  }
  z
}

avifiles = dir("~/Lab_Data/sudar/movie/arikawa_220525", full = TRUE, pattern = "suda_*.*.AVI")

K = length(avifiles)

pngfolder = "temp"
if(!dir.exists(pngfolder)) {
  dir.create(pngfolder)
}

rdsfolder = "rds"
if(!dir.exists(rdsfolder)) {
  dir.create(rdsfolder)
}

## Run this in bash
outfile = str_c(pngfolder, "/%04d.png")

list_of_png = dir(pngfolder, full = T)
if(length(list_of_png) > 0) {
  file.remove(list_of_png)
}

list_of_rds = dir(rdsfolder, full = T)
if(length(list_of_rds) > 0) {
  file.remove(list_of_rds)
}


for(k in 1:K) {
  list_of_png = dir(pngfolder, full = T)
  if(length(list_of_png) > 0) {
    file.remove(list_of_png)
  }
  
  list_of_rds = dir(rdsfolder, full = T)
  if(length(list_of_rds) > 0) {
    file.remove(list_of_rds)
  }
  
  # Extract images from video using ffmpeg.
  cmd = str_c("ffmpeg -i ", avifiles[k], " -vf fps=1 -vsync 0 ", outfile, " -y && sleep 5")
  system(cmd)
  pngfiles = dir(pngfolder, full = T, pattern = "png")
  
  # Calculate entropy and store in rds file.
  for(i in 1:(length(pngfiles)-1)) {
    x1 = pngfiles[i]   |> load.image() |> grayscale() |> as.matrix()
    x2 = pngfiles[i+1] |> load.image() |> grayscale() |> as.matrix()
    
    z = x1 - x2
    z = calc_entropy_matrix(z)
    fname = sprintf(str_c(rdsfolder, "/%04d.rds"), i)
    z |> write_rds(fname)
  }
  
  # Read rds files and combine.
  rdsfiles = dir("rds", pattern = "rds", full = T)
  
  dset  = tibble(rds = rdsfiles) |> mutate(z = map(rds, read_rds))
  
  dset = dset |> mutate(z = future_map(z, \(x) {
    x |> as_tibble(.name_repair = ~sprintf("v%03d", 1:length(.))) |> 
      mutate(rows = 1:n()) |> 
      pivot_longer(starts_with("v")) |> 
      mutate(cols = str_extract(name, "[0-9]+")) |> 
      mutate(cols = as.numeric(cols))
  }))
  
  rdsoutfile = basename(avifiles[k]) |> str_replace("\\.AVI", "-delta-entropy.rds")
  dset |> write_rds(file = rdsoutfile)
  
  dset = dset |> unnest(z) |> 
    group_by(rds) |> 
    summarise(value = median(value)) |> 
    mutate(x = str_extract(rds, "[0-9]+") |> as.numeric()) 
  
  title = basename(avifiles[k])
  xlabel = "Time (s)"
  ylabel = "Median entropy (bits)"
  plotout = dset |> 
    ggplot() + 
    geom_point(aes(x = x, y = value)) +
    scale_x_continuous(xlabel) +
    scale_y_continuous(ylabel) +
    labs(title = title)
  
  plotfile = str_replace(title, "\\.AVI", "-delta-entropy-plot.png")
  height = 100
  ggsave(filename = plotfile, plotout,
         width = sqrt(2)*height, height = height, units = "mm")
}


