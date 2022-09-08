library(tidyverse)
library(magick)
library(foreach)
library(furrr)
library(gganimate)

options(mc.cores = 20)
plan("multisession")

entropy = function(x, max = 255) {
  x = as.vector(x)
  x = x[x>0] / max
  -sum(x * log2(x))
}

calc_entropy_matrix = function(img, N = 10) {
  img_dimension = dim(img)
  x = matrix(1:img_dimension[1], ncol = N, byrow = T)
  y = matrix(1:img_dimension[2], ncol = N, byrow = T)
  w = expand.grid(i = 1:dim(x)[1], j = 1:dim(y)[1])
  z2 = apply(w, 1, \(a) {img[x[a[1],], y[a[2],]] |> entropy()})
  matrix(z2, ncol = dim(y)[1], nrow = dim(x)[1])
  # # Need to find a way to do this with apply().
  # for(i in 1:I) {
  #   for(j in 1:J) {
  #     img_out = img[x[i,], y[j, ]]
  #     z[i, j] = entropy(img_out)
  #   }
  # }
  # z
}


create_movie = function(avifile, folder = NULL) {
  bname = basename(avifile)
  bname = str_remove(bname, "\\.AVI")
  stopifnot("Missing avi folder name." = !is.null(folder))
  # Build and clean the folders
  entropyfolder = str_c("/home/Lab_Data/sudar/movie/", folder, "/csv_data/")
  if(!dir.exists(entropyfolder)) {
    dir.create(entropyfolder)
  }
  entropyplotsfolder = str_c("/home/Lab_Data/sudar/movie/", folder, "/entropy_plots/")
  if(!dir.exists(entropyplotsfolder)) {
    dir.create(entropyplotsfolder)
  }
  
  mp4folder = str_c("/home/Lab_Data/sudar/movie/", folder, "/entropy/")
  if(!dir.exists(mp4folder)) {
    dir.create(mp4folder)
  }
  
  # Temporary folders
  ggfolder1 = "gganimate1"
  if(!dir.exists(ggfolder1)) {
    dir.create(ggfolder1)
  }
  ggfolder2 = "gganimate2"
  if(!dir.exists(ggfolder2)) {
    dir.create(ggfolder2)
  }
  matrixfolder = "tmp_txt"
  if(!dir.exists(matrixfolder)) {
    dir.create(matrixfolder)
  }
  pngfolder1 = "tmp_png_folder1"
  if(!dir.exists(pngfolder1)) {
    dir.create(pngfolder1)
  }
  
  pngfolder2 = "tmp_png_folder2"
  if(!dir.exists(pngfolder2)) {
    dir.create(pngfolder2)
  }
  
  list_of_png1 = dir(pngfolder1, full = T)
  if(length(list_of_png1) > 0) {
    file.remove(list_of_png1)
  }
  
  list_of_png2 = dir(pngfolder2, full = T)
  if(length(list_of_png2) > 0) {
    file.remove(list_of_png2)
  }
  
  list_of_mat = dir(matrixfolder, full = T)
  if(length(list_of_mat) > 0) {
    file.remove(list_of_mat)
  }
  
  list_of_rds = dir(ggfolder1, full = T)
  if(length(list_of_rds) > 0) {
    file.remove(list_of_rds)
  }
  list_of_rds = dir(ggfolder2, full = T)
  if(length(list_of_rds) > 0) {
    file.remove(list_of_rds)
  }
  
  ##############################################################################
  
  # Extract images from video using ffmpeg.
  # This takes abot 4 minutes
  outfile = str_c(pngfolder1, "/%04d.ppm")
  FPS = 1 # Use 1 frame per second.
  FPS = str_c("fps=", FPS)
  cmd = str_c("ffmpeg -i ", avifile, " -vf ", FPS, " -vsync 0 ", outfile, " -y && sleep 5")
  # cmd = str_c("ffmpeg -i ", avifile, " -vf 'fps=1,scale=-1:600' -vsync 0 ", outfile, " -y && sleep 5")
  system(cmd)
  
  pngfiles1 = dir(pngfolder1, full = T, pattern = "ppm")
  
  # Get image dimensions
  iset = image_read(pngfiles1[1])
  wh = iset |> image_info() |> select(width, height)
  wh = as.numeric(wh)
  
  # Calculate entropy and store in rds file.
  # This takes about 10 minutes
  
  make_entropy_mat = function(pngfiles, i) {
    z = pngfiles |> image_read() |> image_data(channels = "gray") |> as.integer() |> drop()
    z = calc_entropy_matrix(z)
    fname = sprintf(str_c(matrixfolder, "/entropy-%04d.txt"), i)
    MASS::write.matrix(z, fname, sep = "\t")
  }
  
  make_delta_entropy_mat = function(pngfiles1, pngfiles2, i) {
    z1 = pngfiles1 |> image_read() |> image_data(channels = "gray") |> as.integer() |> drop()
    z2 = pngfiles2 |> image_read() |> image_data(channels = "gray") |> as.integer() |> drop()
    z = z2 - z1
    z = calc_entropy_matrix(z)
    fname = sprintf(str_c(matrixfolder, "/delta-entropy-%04d.txt"), i)
    MASS::write.matrix(z, fname, sep = "\t")
  }
  
  tibble(pngfiles1) |> 
    mutate(i = 1:n()) |> 
    mutate(future_map2(pngfiles1, i, make_entropy_mat))
  
  tibble(pngfiles1) |> 
    mutate(pngfiles2 = lead(pngfiles1)) |> 
    slice_head(n = -1) |> 
    mutate(i = 1:n()) |> 
    mutate(future_pmap(list(pngfiles1, pngfiles2, i), make_delta_entropy_mat))
  
  # Read rds files and combine.
  # Save to a gz file because a csv file is 2 GB.
  matfiles = dir(matrixfolder, pattern = "^entropy-", full =T)
  dset  = tibble(matfiles) |> mutate(z = future_map(matfiles, read_table, col_names = FALSE))
  dset0 = dset |> mutate(z = future_map(z, \(x) {
    x |> as_tibble(.name_repair = ~sprintf("v%04d", 1:length(.))) |> 
      mutate(rows = 1:n()) |> 
      pivot_longer(starts_with("v")) |> 
      mutate(cols = str_extract(name, "[0-9]+")) |> 
      mutate(cols = as.numeric(cols))
  }))
  
  csvoutfile = str_c(entropyfolder,  bname, "_entropy.csv.gz")
  dset0 |> unnest(z) |> write_csv(file = csvoutfile)
  
  matfiles = dir(matrixfolder, pattern = "^delta-entropy-", full =T)
  eset  = tibble(matfiles) |> mutate(z = future_map(matfiles, read_table, col_names = FALSE))
  eset0 = eset |> mutate(z = future_map(z, \(x) {
    x |> as_tibble(.name_repair = ~sprintf("v%04d", 1:length(.))) |> 
      mutate(rows = 1:n()) |> 
      pivot_longer(starts_with("v")) |> 
      mutate(cols = str_extract(name, "[0-9]+")) |> 
      mutate(cols = as.numeric(cols))
  }))
  
  csvoutfile = str_c(entropyfolder, bname, "_delta_entropy.csv.gz")
  eset0 |> unnest(z) |> write_csv(file = csvoutfile)
  
  
  ##############################################################################
  dset = dset0 |> unnest(z) |> 
    group_by(matfiles) |> 
    summarise(value = median(value)) |> 
    mutate(x = str_extract(matfiles, "[0-9]+") |> as.numeric()) 
  eset = eset0 |> unnest(z) |> 
    group_by(matfiles) |> 
    summarise(value = median(value)) |> 
    mutate(x = str_extract(matfiles, "[0-9]+") |> as.numeric()) 
  
  col = viridis::viridis(10)
  
  title = basename(avifile)
  xlabel = "Time (s)"
  ylabel = "Median entropy (bits)"
  plotout = ggplot() + 
    geom_line(aes(x = x, y = value), color = col[1], dset, alpha = 0.5) +
    geom_line(aes(x = x, y = value), color = col[8], data = eset, alpha = 0.5) +
    scale_x_continuous(xlabel) +
    scale_y_continuous(ylabel) +
    labs(title = title, caption = "Purple is entropy and green is delta-entropy.") +
    theme_grey(10) +
    theme(panel.grid = element_blank())
  
  plotfile = str_c(entropyplotsfolder, bname, "_entropy-plot.png")
  height = 100
  ggsave(filename = plotfile, plotout, width = sqrt(2)*height, height = height, units = "mm")
  ###############################################################################
  # delta-entropy
  
  fout1 %<-% {
    
    ylabel = "Median delta-entropy (bits)"
    plotout1 = 
      eset |> mutate(label = sprintf("CV: %0.2f", value)) |> 
      mutate(x2 = x) |> 
      ggplot() + 
      geom_line(aes(x = x, y = value), size = 1, color = col[1]) +
      geom_point(aes(x = x, y = value), size = 5, color = col[3], alpha = 0.5) +
      geom_text(aes(x = 10, y = 80, label= label), vjust = 1, hjust = 0, size = 10, color = col[5]) + 
      scale_x_continuous(xlabel) +
      scale_y_continuous(ylabel) +
      transition_reveal(x) +
      labs(title = str_c(title, " (Time: {frame_along} s)")) +
      theme_grey(24) +
      theme(panel.grid = element_blank())
    
    animate(plotout1, nframes = (dim(dset)[1]-1), device = "png",
            width = wh[1]/2, height = wh[2]/4,  units = "px",
            renderer = file_renderer(ggfolder1, prefix = "dataplot", overwrite = TRUE))
  }
  
  
  fout2 %<-% {
    
    ylabel = "Median entropy (bits)"
    plotout2 = 
      dset |> 
      mutate(label = sprintf("CV: %0.2f", value)) |> 
      mutate(x2 = x) |> 
      ggplot() + 
      geom_line(aes(x = x, y = value), size = 1, color = col[1]) +
      geom_point(aes(x = x, y = value), size = 5, color = col[3], alpha = 0.5) +
      geom_text(aes(x = 600, y = 80, label= label), vjust = 1, hjust = 1, size = 10, color = col[5]) + 
      scale_x_continuous(xlabel) +
      scale_y_continuous(ylabel) +
      transition_reveal(x) +
      labs(title = str_c(title, " (Time: {frame_along} s)")) +
      theme_grey(24) +
      theme(panel.grid = element_blank())
    
    animate(plotout2, nframes = dim(dset)[1], device = "png",
            width = wh[1]/2, height = wh[2]/4,  units = "px",
            renderer = file_renderer(ggfolder2, prefix = "dataplot", overwrite = TRUE))
  }
  
  while(all(!resolved(list(fout1, fout2)))) { }
  
  plotpng1 = dir(ggfolder1, full = T)
  plotpng2 = dir(ggfolder2, full = T)
  geom = str_c("x", wh[2]/2)
  length(plotpng1)
  length(plotpng2)
  length(pngfiles1)
  
  # This takes 20 minutes or so.
  my.cluster <- parallel::makeForkCluster(nnodes = 16)
  doParallel::registerDoParallel(cl = my.cluster, cores = 16)
  
  I = min(c(length(plotpng1), length(plotpng2), length(pngfiles1)))
  foreachout = foreach(i = 1:I, .packages = "magick") %dopar% {
    p1 = image_read(plotpng1[i])
    p2 = image_read(plotpng2[i])
    p3 = image_read(pngfiles1[i]) |> image_resize("x1080")
    p12 = image_append(c(p1,p2), stack = TRUE)
    outfile2 = sprintf("%s/%05i.png", pngfolder2, i)
    image_append(c(p12,p3), stack = FALSE) |> image_write(path = outfile2)
    TRUE
  }
  parallel::stopCluster(cl = my.cluster)
  
  # Build mp4 of figures and video.
  outfile2 = dir(pngfolder2, full = TRUE)
  movfile = str_c(bname, "-composite-plots.mp4")
  movfile = str_c(mp4folder, movfile)
  
  cmd = str_c("ffmpeg -framerate 15 -i ", pngfolder2, "/%05d.png -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white' -c:v libx264 -an -pix_fmt yuv420p ", movfile, " -y && sleep 5")
  system(cmd)
}

avifiles = dir("~/Lab_Data/sudar/movie/arikawa_220525", 
               full = TRUE, pattern = "suda_01_*.*.AVI")

for(i in 1:length(avifiles)) {
  create_movie(avifiles[i])
}

avifiles = dir("~/Lab_Data/sudar/movie/arikawa_220525", 
               full = TRUE, pattern = "suda_02_*.*.AVI")

for(i in 1:length(avifiles)) {
  create_movie(avifiles[i])
}
