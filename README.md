# Steering Angle Prediction using Deep Learning

1. Download or clone this repository

2. Navigate to the repo's location on your hard disk using Anaconda Prompt (using cd)

3. Run **python steer.py**

Steer.py has been implemented using pywin32 library for screen capture

The model, steer.h5 (28.2 MB) has been trained on Keras framework and optimized for inference on GPU

The input image to the model is the same size as the screen capture but in gray scale

The screen capture parameters (**line 41**) as default are:
screen = grab_screen(region=(0, 40, 1000, 600))
