# Run this once in R. 
# install.packages("reticulate") # Reticulate should already be installed.

# Do this in R
Sys.setenv(RETICULATE_PYTHON = "pyenv/bin/python")
reticulate::py_config()


library(tidyverse)

df = read_csv("test.csv")

df |> 
  group_by(tau) |> 
  summarise(value = mean(value)) |> 
  ggplot() + 
  geom_path(aes(x = tau, y = value))


df |> 
  arrange(tau,i, j) |> 
  filter(near(tau, 10)) |>
  tail()

df |> 
  filter(near(tau, 5)) |> 
  ggplot() + 
  geom_tile(aes(x = i, y = j, fill = value))



