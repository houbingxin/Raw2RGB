# Raw2RGB

Environment:
Python 3.6
openCV
Tensorflow
Keras

The code is able to transform raw image to rgb image.

The result images are saved in GoogleDrive
test_prediction.py:  is for (224,224,4) input images, output images are (224,224,3) PNG files.
Full_test_predictioin.py:  is for full-resolution (m,n,4) input images. output images are (m,n,3) PNG files.

To run the code, need to set three things inside the code: 1. input folder path; 2. model path; 3. output folder path.
