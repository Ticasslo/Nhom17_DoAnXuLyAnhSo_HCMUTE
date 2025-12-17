import google.protobuf # 4.25.8
print("protobuf:", google.protobuf.__version__)

import numpy as np
print("numpy:", np.__version__)

import PIL
print("Pillow:", PIL.__version__)

import mediapipe as mp
print("mediapipe:", mp.__version__)

import tensorflow as tf
print("tensorflow:", tf.__version__)
print("keras:", tf.keras.__version__)

import pandas as pd
print("pandas:", pd.__version__)

print("OK")

# pip install numpy==1.26.4
# pip install pillow==12.0.0
# pip install mediapipe==0.10.21 # KHÔNG CẦN CÀI OPENCV-PYTHON VÌ MEDIAPIPE CÀI CHUNG RỒI
# pip install matplotlib==3.10.8
# pip install tensorflow==2.16.1
# pip install pandas
# pip install paho-mqtt
# pip uninstall -y jax jaxlib # BỎ ĐI JAX VÀ JAXLIB VÌ MEDIAPIPE CÀI CHUNG RỒI