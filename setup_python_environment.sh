#!/bin/bash
# Prepare python environment

# Once the pyenv is setup, this does not need to be done again.
# virtualenv pyenv 
source pyenv/bin/activate
which python
# 
# pip install numpy pandas matplotlib opencv-python
# pip install imutils
# 
# install.packages("reticulate")
# 
# Do this in R
# # Sys.setenv(RETICULATE_PYTHON = "pyenv/bin/python")
# reticulate::py_config()

# Reference
# https://support.rstudio.com/hc/en-us/articles/360023654474-Installing-and-Configuring-Python-with-RStudio