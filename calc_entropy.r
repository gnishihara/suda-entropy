# エントロピーデータの解析
# 2022-09-07

# パッケージの読み込み
library(tidyverse)
library(ggvegan)
library(ggpubr)
library(patchwork)
library(showtext)
library(lubridate)
library(vegan)
library(gnnlab)

# Python の numpy モジュールを読み込む
library(reticulate)
np = import("numpy")
################################################################################
# 作図の準備
font_add(family = "notosans", regular = "NotoSansCJKjp-Regular.otf")
# theme_gray(base_size = 10, base_family = "notosans") |> theme_set()
theme_pubr(base_size = 10, base_family = "notosans") |> theme_set()
showtext_auto()
################################################################################
# 関数の定義
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



################################################################################
# 画像の区分毎の変動係数

make_plot_timeseries = function(dset3) {
  
  cols = viridis::cividis(6)
  YMD = dset3 |> pull(ymd) |> unique()
  CAMERA = "01"
  TITLE = str_glue("{CAMERA} ({YMD})")
  cam1a = dset3 |> select(-data) |> 
    filter(camera == CAMERA) |> 
    unnest(mean2) |> 
    filter(m != "01") |> print() |> 
    ggplot() + 
    geom_point(aes(x = starttime,  y = mean), size = 0.5, col = cols[2]) + 
    geom_smooth(aes(x = starttime, y = mean), col = cols[5],
                method = "gam",
                formula = y~s(x, k = 10, bs = "ps"), se = F) + 
    scale_x_datetime(date_labels = "%H") +
    labs(x = "時間", y = "平均値") +
    facet_grid(rows = vars(Var3),
               cols = vars(Var2)) +
    labs(title = TITLE) +
    theme(panel.grid = element_blank())
  
  CAMERA = "02"
  TITLE = str_glue("{CAMERA} ({YMD})")
  cam2a = cam1a %+% 
    (dset3 |> select(-data) |> 
       filter(camera == CAMERA) |> unnest(mean2) |> 
       filter(m != "01")) +
    labs(title = TITLE) 
  
  CAMERA = "03"
  TITLE = str_glue("{CAMERA} ({YMD})")
  cam3a = cam1a %+% (dset3 |> select(-data) |> 
                       filter(camera == CAMERA) |> 
                       unnest(mean2) |> 
                       filter(m != "01")) +
    labs(title = TITLE) 
  
  CAMERA = "04"
  TITLE = str_glue("{CAMERA} ({YMD})")
  cam4a = cam1a %+% (dset3 |> select(-data) |> 
                       filter(camera == CAMERA) |> 
                       unnest(mean2) |> 
                       filter(m != "01")) +
    labs(title = TITLE) 
  
  pout = cam1a + cam2a + cam3a + cam4a + plot_layout(ncol = 2, nrow = 2)
  
  pdfname = str_glue("{YMD}-mean.pdf")
  save_plot(pdfname = pdfname, plot = pout, width = 400, height = 300)
  
  CAMERA = "01"
  TITLE = str_glue("{CAMERA} ({YMD})")
  cam1b = dset3 |> select(-data) |> filter(camera == CAMERA) |> 
    unnest(mean2) |> filter(m != "01") |> 
    ggplot() + 
    geom_point(aes(x = starttime,  y = sd^2), size = 0.5, col = cols[2]) + 
    geom_smooth(aes(x = starttime, y = sd^2), col = cols[5],
                method = "gam",
                formula = y~s(x, k = 10, bs = "ps"), se = F) + 
    scale_x_datetime(date_labels = "%H") +
    labs(x = "時間", y = "分散") +
    facet_grid(rows = vars(Var3),
               cols = vars(Var2)) +
    labs(title = TITLE) +
    theme(panel.grid = element_blank())
  
  CAMERA = "02"
  TITLE = str_glue("{CAMERA} ({YMD})")
  cam2b = cam1b %+% (dset3 |> select(-data) |> 
                       filter(camera == CAMERA) |> 
                       unnest(mean2) |> 
                       filter(m != "01")) +
    labs(title = TITLE) 
  
  CAMERA = "03"
  TITLE = str_glue("{CAMERA} ({YMD})")
  cam3b = cam1b %+% (dset3 |> select(-data) |> 
                       filter(camera == CAMERA) |> 
                       unnest(mean2) |> 
                       filter(m != "01")) +
    labs(title = TITLE) 
  
  CAMERA = "04"
  TITLE = str_glue("{CAMERA} ({YMD})")
  cam4b = cam1b %+% (dset3 |> select(-data) |> 
                       filter(camera == CAMERA) |> 
                       unnest(mean2) |> 
                       filter(m != "01")) +
    labs(title = TITLE) 
  
  pout = cam1b + cam2b + cam3b + cam4b + plot_layout(ncol = 2, nrow = 2)
  pdfname = str_glue("{YMD}-variance.pdf")
  save_plot(pdfname, width = 400, height = 300, plot = pout)
  
  pout = 
    cam1a + cam1b +
    cam2a + cam2b +
    cam3a + cam3b +
    cam4a + cam4b + plot_layout(ncol = 2, nrow = 4) +
    plot_annotation(title = "Δエントロピー・左：平均値・右：分散")
  
  pdfname = str_glue("{YMD}-timeseries.pdf")
  save_plot(pdfname, width = 300, height = 500, plot = pout)
}
################################################################################
# Do a RDA on the mean vs sd^2, then a linear regression
# of RDA1 and RDA2 against time and camera.
make_plot_rda = function(dset3) {
  cols = viridis::cividis(6)
  YMD = dset3 |> pull(ymd) |> unique()
  dset4 = dset3 |> select(camera, st, m, starttime, nframes, mean2) |> unnest(mean2) |> 
    mutate(camera = str_c("Cam ", camera)) |> 
    mutate(camera = factor(camera, 
                           labels = c("Cam~01",
                                      "Cam~02",
                                      "Cam~03",
                                      "Cam~04")))
  dset4 = dset4 |>
    mutate(hours = hour(starttime) + minute(starttime)/60) |> 
    filter(m != "01") |> # 最初の動画を外す
    filter(between(Var2, 2,5)) |> # 画像の外側のセルを外す
    filter(between(Var3, 2,5)) 
  
  X = dset4 |> select(mean, sd) |> as.matrix()
  rdaout0 = rda(X ~ 1, data = dset4, scale = TRUE)
  rdaout1 = rda(X ~ camera, data = dset4, scale = TRUE)
  rdaout2 = rda(X ~ camera + hours, data = dset4, scale = TRUE)
  rdaout  = rda(X ~ camera * hours, data = dset4, scale = TRUE)
  # camera と hours をモデルに追加すると有意な効果があった
  # データ数が多いから当然？
  anova(rdaout0, rdaout1, rdaout2, rdaout)
  
  z = anova(rdaout) |> as_tibble()
  
  fval = sprintf("F['(%d, %d)']*' = '*%0.2f*', '*R[adj]^{2}*' = '*%0.4f", z$Df[1], z$Df[2], z$F[1], RsquareAdj(rdaout)$adj.r.squared) |> print()
  
  sites   = fortify(rdaout, axes = 1:2, scaling = 2) |> filter(str_detect(Score, "sites")) |> mutate(hours = dset4$hours, camera = dset4$camera)
  species = fortify(rdaout, axes = 1:2, scaling = 2) |> filter(str_detect(Score, "species"))
  biplot  = fortify(rdaout, axes = 1:2, scaling = 2) |> filter(str_detect(Score, "biplot"))
  
  
  p1 = ggplot() + 
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey75") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey75") +
    geom_text(aes(x = -2, y = 2, label = fval), parse = T, vjust = 1, hjust = 0) +
    geom_point(aes(x = RDA1, y = RDA2), data = sites, alpha = 0.5, color = "grey50") +
    geom_segment(aes(x = 0, y = 0, 
                     xend = 5 * RDA1, 
                     yend = 5 * RDA2, color = Label), data = biplot |> filter(str_detect(Label, "hou")), 
                 size = 1,
                 arrow = arrow(15, unit(1.5, "mm"))) +
    geom_segment(aes(x = 0, y = 0, 
                     xend =0.5 * RDA1, 
                     yend =0.5 * RDA2, color = Label), data = species, 
                 size = 1,
                 arrow = arrow(15, unit(1.5, "mm"))) +
    # annotate("text", x = 1.5, y = -1.5, hjust = 1, label = "Correlation biplot", family = "notosans") +
    scale_color_viridis_d(end = 0.95, labels = scales::parse_format()) + 
    scale_fill_viridis_d( end = 0.95, labels = scales::parse_format()) + 
    coord_fixed() +
    scale_x_continuous(parse(text = "RDA[1]"), limits = c(-2, 2)) +
    scale_y_continuous(parse(text = "RDA[2]"), limits = c(-2, 2)) +
    theme(legend.position = c(0,0),
          legend.justification = c(0,0))
  
  
  p2 = sites |> 
    pivot_longer(c(RDA1, RDA2)) |>
    group_by(name, hours, camera) |> 
    summarise(across(value, list(mean = mean, sd = sd, se = se, n = length)),
              .groups = "drop") |> 
    ggplot() + 
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey75") +
    geom_pointrange(aes(x = hours, 
                        y     = value_mean,
                        ymin  = value_mean - value_se,
                        ymax  = value_mean + value_se,
                        color = name),
                    show.legend = F) +
    geom_smooth(aes(x = hours, y = value_mean, 
                    color = name),
                method = "gam",
                formula = y~s(x, bs = "ps"),
                show.legend = F) +
    scale_color_viridis_d("", end = 0.8) +
    facet_wrap(vars(camera, name), ncol = 2,
               scales = "free_y")
  
  
  p3 = dset4 |> 
    select(hours, camera, mean, sd) |> 
    pivot_longer(c(mean, sd)) |>
    group_by(name, hours, camera) |> 
    summarise(across(value, list(mean = mean, sd = sd, se = se, n = length)),
              .groups = "drop") |> 
    ggplot() + 
    geom_pointrange(aes(x = hours, 
                        y     = value_mean,
                        ymin  = value_mean - value_se,
                        ymax  = value_mean + value_se,
                        color = name),
                    show.legend = F) +
    geom_smooth(aes(x = hours, y = value_mean, 
                    color = name),
                method = "gam",
                formula = y~s(x, bs = "ps"),
                show.legend = F) +
    scale_color_viridis_d("", end = 0.8) +
    facet_wrap(vars(camera, name), ncol = 2,
               scales = "free_y")
  
  
  design = "
12
33
33
"
  
  pout = p2 + p3 + p1 + plot_layout(design = design)
  pdfname = str_glue("{YMD}-rda.pdf")
  save_plot(pdfname, width = 240, height = 400, plot = pout)
}

process_data = function(fnames) {

  # データの読み込み・以外と時間がかかる。
  dset = tibble(fnames) |> mutate(bnames = basename(fnames)) |> 
    mutate(data = map(fnames, get_data))
  dset = dset |>  mutate(camera = str_extract(bnames, "0[1-4]"),
                         st  = str_extract(bnames, "(?<=st)[0-9]{1,2}"),
                         m  = str_extract(bnames, "(?<=m)[0-9]{2}"),
                         ymd = str_extract(bnames, "[0-9]{6}"))
  dset = dset |> mutate(bnames = basename(fnames))
  dset2 = dset |> select(-fnames)
  dset3 = dset2 |> mutate(bnames = str_extract(bnames, "suda*.*[0-9]"))
  # 時間データとエントロピーデータを結合する。
  # dset3 = left_join(dset2, datetimes, by = c("bnames" = "filename"))
  # ここでは、Var1 (コマ軸) に対する平均値と標準偏差をもとめる。
  dset3 = dset3 |> 
    mutate(mean1 = map(data, \(x) {
      x |> 
        group_by(Var1) |> 
        summarise(mean = mean(value), sd = sd(value), n = length(value),  .groups = "drop")
    }))
  
  # ここでは、Var2 と Var3 に対する平均値と標準偏差をもとめる。
  # Var2 は y 軸、Var3 は x 軸
  dset3 = dset3 |> 
    mutate(mean2 = map(data, \(x) {
      x |> 
        group_by(Var2, Var3) |> 
        summarise(mean = mean(value), sd = sd(value), n = length(value),  .groups = "drop")
    }))
  YMD = dset3 |> pull(ymd) |> unique()
  filename = str_glue("{YMD}-dataset.rds")
  write_rds(dset3, filename, compress = "gz")
  dset3
}



################################################################################
# フォルダを設定して、ファイル・ポスを読み込む


folder = "~/Lab_Data/sudar/movie"

dout = tibble(folder = dir(folder, full = TRUE)) |> 
  filter(str_detect(folder, "2207")) |> 
  mutate(datetimes = map(folder, function(x) {
    YMD = x |> str_extract("[0-9]{6}")
    fname = str_glue("{x}/arikawa_{YMD}-datetimes.csv")
    read_csv(fname) |> distinct() |> print()
  })) |> 
  mutate(folder = dir(folder, full = TRUE, pattern = "delta")) |> 
  mutate(fnames = map(folder, dir, pattern = "npy", full = TRUE)) 
  

dout = dout |> 
  mutate(data = map2(datetimes, fnames, function(d, f) {
    f = tibble(chk = basename(f) |> str_extract("suda_*.*[0-9]"),
               fnames = f)
    d = d |> 
      mutate(chk = basename(filename) |> str_extract("suda_*.*[0-9]")) 
     
    inner_join(d, f, by = "chk") 
  })) |> 
  select(data) |> 
  unnest(data)

dout2 = dout |> mutate(data = map(fnames, process_data))


dout2 |> select(starttime, data) |> 
  unnest(data) |> 
  unnest(data) |> 
  select(starttime, value, camera, st) |> 
  mutate(starttime = floor_date(starttime, "hours")) |> 
  mutate(h = hour(starttime)) |> 
  group_by(camera, st, h) |> 
  summarise(val = mean(value),
            sd = sd(value)) |> 
  ggplot() + 
  geom_point(aes(x = h, y = val, color = camera)) +
  scale_x_continuous(limits = c(0, 24),
                     breaks = seq(0, 24, by = 4))

folder = "~/Lab_Data/sudar/movie/arikawa_220725/delta_entropy_csvdata/"


fnames = dir(folder, pattern = "npy", full = TRUE)
# extract_time.py で作った CSV ファイルを読み込む
# AVIの時間情報が入っています。
datetimes = read_csv("~/Lab_Data/sudar/movie/arikawa_220725/arikawa_220725-datetimes.csv")
datetimes = datetimes |> mutate(filename = str_extract(filename, "suda*.*[0-9]"))
dset3 = process_data(fnames, datetimes)
make_plot_timeseries(dset3)
make_plot_rda(dset3)



################################################################################
# フォルダを設定して、ファイル・ポスを読み込む
folder = "~/Lab_Data/sudar/movie/arikawa_220726/delta_entropy_csvdata/"
fnames = dir(folder, pattern = "npy", full = TRUE)
# extract_time.py で作った CSV ファイルを読み込む
# AVIの時間情報が入っています。
datetimes = read_csv("~/Lab_Data/sudar/movie/arikawa_220726/arikawa_220726-datetimes.csv")
datetimes = datetimes |> mutate(filename = str_extract(filename, "suda*.*[0-9]"))
dset3 = process_data(fnames, datetimes)
make_plot_timeseries(dset3)
make_plot_rda(dset3)



################################################################################
# フォルダを設定して、ファイル・ポスを読み込む
folder = "~/Lab_Data/sudar/movie/arikawa_220727/delta_entropy_csvdata/"
fnames = dir(folder, pattern = "npy", full = TRUE)
# extract_time.py で作った CSV ファイルを読み込む
# AVIの時間情報が入っています。
datetimes = read_csv("~/Lab_Data/sudar/movie/arikawa_220727/arikawa_220727-datetimes.csv")
datetimes = datetimes |> mutate(filename = str_extract(filename, "suda*.*[0-9]"))
dset3 = process_data(fnames, datetimes)
make_plot_timeseries(dset3)
make_plot_rda(dset3)



################################################################################
# フォルダを設定して、ファイル・ポスを読み込む
folder = "~/Lab_Data/sudar/movie/arikawa_220728/delta_entropy_csvdata/"
fnames = dir(folder, pattern = "npy", full = TRUE)
# extract_time.py で作った CSV ファイルを読み込む
# AVIの時間情報が入っています。
datetimes = read_csv("~/Lab_Data/sudar/movie/arikawa_220728/arikawa_220728-datetimes.csv")
datetimes = datetimes |> mutate(filename = str_extract(filename, "suda*.*[0-9]"))
dset3 = process_data(fnames, datetimes)
make_plot_timeseries(dset3)
make_plot_rda(dset3)


