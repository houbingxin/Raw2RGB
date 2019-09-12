#%%
import numpy as np
import os, glob
from keras.models import load_model
import cv2
from PIL import Image
import os, sys
# This code is for AIM2019 Raw2RGB, Full Resolution Raw image to RGB
# Input is (*,*,4) Raw image (PNG)
# Output is (*,*,3) RGB image (PNG)
# Author: Bingxin Hou
# 09/11/2019

cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

def generate_patches(image_data,sub_patch_dim):

    patch_height = sub_patch_dim[0]
    patch_width = sub_patch_dim[1]
    x_spots =  list(range(0,image_data.shape[0] , patch_height))
    y_spots =  list(range(0, image_data.shape[1] , patch_width ))

    image_patches = []
    all_patches = []
    position =[]
    h=image_data.shape[0]
    w=image_data.shape[1]
    for x in x_spots:
        for y in y_spots:
            if ((x+patch_height)>image_data.shape[0]-1):
                x=image_data.shape[0]-1-patch_height
            if ((y+patch_width)>image_data.shape[1]-1):
                y=image_data.shape[1]-1-patch_width
            image_patches = image_data[x: x+patch_height,y: y+patch_width]
            all_patches.append(image_patches)
            position.append([x,y])
    all_patches = np.asarray(all_patches)
    position = np.array(position)
    return all_patches, position, h, w

def getData(X_list):


    y = cv2.imread(X_list, cv2.IMREAD_UNCHANGED)

    sub_patch_dim = [224, 224]
    y_patch, position1, h, w = generate_patches(y, sub_patch_dim)

    return y_patch,  position1, h, w


input_path = os.path.join('FullResTestingPhoneRaw') # you need to set the input folder
#%%

mdl_path = 'C:/Users/houbi/Documents/research/Raw2RGB/FgSegNet_S/test_b10_1000.h5' # you need to set the model path

for imageN in range(0,10):
    X_list = glob.glob(os.path.join(input_path, '*.png'))

    data, position, h, w = getData(X_list[imageN])
    position = np.asarray(position)


    model = load_model(mdl_path,compile=False)
    probs_patch = model.predict(data, batch_size=1, verbose=1)

    #### patches stitch to image ##############

    imagebank = np.zeros((h,w,3))
    list =[]
    i = int(np.ceil(h/224))
    j = int(np.ceil(w/224))
    ii=0


    for index in range(int((1-1) * i*j), int((1-1)*i*j+i*j)):
       [startx,starty] = position[ii]
       imagebank[startx:startx+224,starty:starty+224] = probs_patch[ii]
       ii=ii+1
    x = imagebank
    x = (x - np.min(x)) / np.ptp(x)

    x = (x * 255).astype(np.uint8)
    a = 'C:/Users/houbi/Documents/research/Raw2RGB/FullResTestingPhone_RGB/'+str(imageN)+'.png'  # you need to set the result image saving folder
    j = Image.fromarray(x)
    j.save(a, compress_level=0)
    del list

