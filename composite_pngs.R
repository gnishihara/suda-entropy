library(tidyverse)
library(magick)


images = dir(".", pattern = "suda.*.png", full = TRUE)
# iset = tibble(images) |> mutate(images = map(images, image_read))
iset = image_read(images)

K = length(images)

Q = K %/% 5
R = K %/% Q

wh = iset[1] |> image_info() |> select(width, height)
scale = 0.25
offset = 5
geom = str_c(floor(scale * wh[1]), "x", floor(scale * wh[2]),
             "+",offset, "+", offset)
tile = str_c(Q, "x", R)
iout = image_montage(iset, geometry = geom, tile = tile, bg = "grey80")
iout |> image_write("entropy-plots.png", format = "png")

