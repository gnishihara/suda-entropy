library(tidyverse)
library(magick)


pngfiles = dir("~/Lab_Data/sudar/movie/arikawa_220525/entropy_plots/", full.names = T)

pngfiles01 = str_subset(pngfiles, "suda_01")
pngfiles02 = str_subset(pngfiles, "suda_02")

img = image_read(pngfiles01)

K = length(pngfiles01)
Q = K %/% 5
R = K %/% Q
wh = img[1] |> image_info() |> select(width, height)
scale = 0.25
offset = 5
geom = str_c(floor(scale * wh[1]), "x", floor(scale * wh[2]), "+",offset, "+", offset)
tile = str_c(Q, "x", R)
iout = image_montage(img, geometry = geom, tile = tile, bg = "grey80")
iout |> image_write("all-entropy-plots-suda_01.png", format = "png")

img = image_read(pngfiles02)
K = length(pngfiles02)
Q = K %/% 5
R = K %/% Q
wh = img[1] |> image_info() |> select(width, height)
scale = 0.25
offset = 5
geom = str_c(floor(scale * wh[1]), "x", floor(scale * wh[2]), "+",offset, "+", offset)
tile = str_c(Q, "x", R)
iout = image_montage(img, geometry = geom, tile = tile, bg = "grey80")
iout |> image_write("all-entropy-plots-suda_02.png", format = "png")

