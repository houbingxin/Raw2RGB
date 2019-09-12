#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import keras
import numpy as np
import os, glob
import PIL

from keras.models import load_model

import cv2
import os, sys
from PIL import Image

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print('Elapsed time is'+ str(time.time() - startTime_for_tictoc) + 'seconds')
    else:
        print('Toc: start time not set')

cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

#
def getData(input_path):


    X = []
    X_name = []

    X_list= glob.glob(os.path.join(input_path, '*.png'))

    for i in range(0,len(X_list)):

        x_name = X_list[i].split('.')[0]
        x_name = x_name.split('\\')[1]

        x = cv2.imread(X_list[i], cv2.IMREAD_UNCHANGED)

        X.append(x)
        X_name.append(x_name)

    XX = np.asarray(X)
    XX_name = np.array(X_name)

    return XX, XX_name

input_path = os.path.join('TestingPhoneRaw')
data,name = getData(input_path)

num_in_row = 1
num_in_col = 2
mdl_path = 'C:/Users/houbi/Documents/research/Raw2RGB/FgSegNet_S/test_b10_1000.h5'

model = load_model(mdl_path,compile=False)
tic()
probs = model.predict(data, batch_size=1, verbose=1)
toc()

for frame_idx in range(0,len(probs)): # display frame index

    x = probs[frame_idx]

    x = (x - np.min(x)) / np.ptp(x)

    x = (x * 255).astype(np.uint8)
    a = 'C:/Users/houbi/Documents/research/Raw2RGB/TestingPhone_RGB/'+ name[frame_idx]+'.png'

    j = Image.fromarray(x)
    j.save(a, compress_level=0)

