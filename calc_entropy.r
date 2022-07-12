# エントロピーデータの解析


# パッケージの読み込み
library(tidyverse)
library(patchwork)
library(showtext)

# Python の numpy モジュールを読み込む
library(reticulate)
np = import("numpy")


# 作図の準備
font_add(family = "notosans", regular = "NotoSansCJKjp-Regular.otf")
theme_gray(base_size = 10, base_family = "notosans") |> theme_set()
showtext_auto()

# 関数の定義



save_plot = function(pdfname, plot = last_plot(), w = 600, h = 600, u = "mm")  {
  pngname = str_replace(pdfname, "pdf", "png")  
  ggsave(pdfname, width = w, height = h, units = u, plot = plot)
  magick::image_read_pdf(pdfname, density = 300) |> 
    magick::image_write(pngname)
  
}

calculate_mean = function(X) {
  # エントロピーデータは配列の状態で処理した方が早い
  # tibble化したら、この処理は時間かから。
  # apply() の場合は、5秒から10秒でおわる。
  submatrix_mean = function(x, zmat, ymat, xmat) {
    mean(zmat[x[1], ymat[x[2], ], xmat[x[3], ]])
  }
  tau = seq(1, dim(X)[1]) # コマ軸 (時間)
  y   = seq(1, dim(X)[2]) # y 軸
  x   = seq(1, dim(X)[3]) # x 軸
  # 画像は 6 x 6 に区分する
  xmat = matrix(x, nrow = 6, byrow = T)
  ymat = matrix(y, nrow = 6, byrow = T)
  egrid = expand.grid(tau, 1:nrow(xmat), 1:nrow(ymat))
  cbind(egrid, 
        value = apply(X = egrid, MARGIN = 1, FUN = submatrix_mean, 
                      zmat = X, ymat = ymat, xmat = xmat))
}

get_data = function(fnames) {
  dset = np$load(fnames) # numpy の load() を使って、 npy ファイルを読み込む
  calculate_mean(dset) |> as_tibble()
}

# フォルダを設定して、ファイル・ポスを読み込む
folder = "~/Lab_Data/sudar/movie/arikawa_220621/delta_entropy_csvdata/"
fnames = dir(folder, pattern = "npy", full = TRUE)

# extract_time.py で作った CSV ファイルを読み込む
# AVIの時間情報が入っています。
datetimes = read_csv("~/Lab_Data/sudar/movie/arikawa_220621/arikawa_220621-datetimes.csv")

# データの読み込み・以外と時間がかかる。
dset = tibble(fnames) |> 
  mutate(bnames = basename(fnames)) |> 
  mutate(camera = str_extract(bnames, "0[1-4]"),
         st  = str_extract(bnames, "(?<=st)[0-9]{2}"),
         m  = str_extract(bnames, "(?<=m)[0-9]{2}"),
         ymd = str_extract(bnames, "[0-9]{6}")) |> 
  mutate(data = map(fnames, get_data))

dset = dset |> mutate(bnames = basename(fnames))
dset2 = dset |> select(-fnames)
dset2 = dset2 |> mutate(bnames = str_extract(bnames, "suda*.*[0-9]"))
datetimes = datetimes |> mutate(filename = str_extract(filename, "suda*.*[0-9]"))


# 時間データとエントロピーデータを結合する。
dset3 = left_join(dset2, datetimes, by = c("bnames" = "filename"))
  
# ここでは、Var1 (コマ軸) に対する平均値と標準偏差をもとめる。
dset3 = dset3 |> 
  mutate(mean1 = map(data, \(x) {
    x |> 
      group_by(Var1) |> 
      summarise(mean = mean(value),
                sd = sd(value),
                n = length(value), 
                .groups = "drop")
  }))

# ここでは、Var2 と Var3 に対する平均値と標準偏差をもとめる。
# Var2 は y 軸、Var3 は x 軸
dset3 = dset3 |> 
  mutate(mean2 = map(data, \(x) {
    x |> 
      group_by(Var2, Var3) |> 
      summarise(mean = mean(value),
                sd = sd(value),
                n = length(value), 
                .groups = "drop")
  }))


dset3 |> select(-data) |> 
  filter(camera == "02") |> 
  unnest(mean1) |> 
  ggplot()+ 
  geom_line(aes(x = Var1, y = mean/sd)) +
  scale_y_log10() +
  facet_wrap(vars(m))


dset3 |> select(-data) |> 
  filter(camera == "02") |> 
  unnest(mean2) |> 
  ggplot() + 
  geom_tile(aes(x = Var3, y = Var2, fill = 100*sd/mean)) + 
  facet_wrap(vars(m))



# 画像の区分毎の変動係数
cols = viridis::cividis(6)
cam1a = dset3 |> select(-data) |> 
  filter(camera == "01") |> 
  unnest(mean1) |> 
  filter(m != "01") |> print() |> 
  ggplot() + 
  geom_point(aes(x = starttime,  y = mean), size = 0.5, col = cols[2]) + 
  geom_smooth(aes(x = starttime, y = mean), col = cols[5],
              method = "gam",
              formula = y~s(x, k = 10, bs = "ps"), se = F) + 
  scale_x_datetime(date_labels = "%H",
                   limits = lubridate::ymd_hm(c("2022-06-21 14:00",
                                     "2022-06-21 20:00"))) +
  labs(x = "時間", y = "平均値") +
  facet_grid(rows = vars(Var3),
             cols = vars(Var2)) +
  labs(title = "Camera 01 (220621)") +
  theme(panel.grid = element_blank())

cam2a = cam1a %+% 
  (dset4 |> select(-data) |> 
     filter(camera == "02") |> unnest(mean1) |> 
     filter(m != "01")) +
  labs(title = "Camera 02 (220621)") 

cam3a = cam1a %+% (dset4 |> select(-data) |> 
  filter(camera == "03") |> 
  unnest(mean1) |> 
  filter(m != "01")) +
  labs(title = "Camera 03 (220621)") 

cam4a = cam1a %+% (dset4 |> select(-data) |> 
  filter(camera == "04") |> 
  unnest(mean1) |> 
  filter(m != "01")) +
  labs(title = "Camera 04 (220621)") 

pout = cam1a + cam2a + cam3a + cam4a + plot_layout(ncol = 2, nrow = 2)
save_plot("2206201-mean.pdf", w = 400, h = 300, plot = pout)


cam1b = dset4 |> select(-data) |> 
  filter(camera == "01") |> 
  unnest(mean1) |> 
  filter(m != "01") |> print() |> 
  ggplot() + 
  geom_point(aes(x = starttime,  y = sd^2), size = 0.5, col = cols[2]) + 
  geom_smooth(aes(x = starttime, y = sd^2), col = cols[5],
              method = "gam",
              formula = y~s(x, k = 10, bs = "ps"), se = F) + 
  scale_x_datetime(date_labels = "%H",
                   limits = lubridate::ymd_hm(c("2022-06-21 14:00",
                                                "2022-06-21 20:00"))) +
  labs(x = "時間", y = "分散") +
  facet_grid(rows = vars(Var3),
             cols = vars(Var2)) +
  labs(title = "Camera 01 (220621)") +
  theme(panel.grid = element_blank())

cam2b = cam1b %+% (dset4 |> select(-data) |> 
                   filter(camera == "02") |> 
                   unnest(mean1) |> 
                   filter(m != "01")) +
  labs(title = "Camera 02 (220621)") 

cam3b = cam1b %+% (dset4 |> select(-data) |> 
                   filter(camera == "03") |> 
                   unnest(mean1) |> 
                   filter(m != "01")) +
  labs(title = "Camera 03 (220621)") 

cam4b = cam1b %+% (dset4 |> select(-data) |> 
                   filter(camera == "04") |> 
                   unnest(mean1) |> 
                   filter(m != "01")) +
  labs(title = "Camera 04 (220621)") 

pout = cam1b + cam2b + cam3b + cam4b + plot_layout(ncol = 2, nrow = 2)
save_plot("2206201-variance.pdf", w = 400, h = 300, plot = pout)

pout = 
cam1a + cam1b +
cam2a + cam2b +
cam3a + cam3b +
cam4a + cam4b + plot_layout(ncol = 2, nrow = 4) +
  plot_annotation(title = "Δエントロピー・左：平均値・右：分散")

save_plot("2206201-timeseries.pdf", w = 300, h = 500, plot = pout)



# Do a PCA on the mean vs sd^2, then a linear regression
# of PC1 and PC2 against time and camera.

library(vegan)
rda(















