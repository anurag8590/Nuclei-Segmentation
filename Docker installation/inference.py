import tensorflow as tf
from tensorflow.keras.models import load_model
import keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
import os
import cv2
import numpy as np
import sys


MODEL_PATH = 'unet.h5'

model = load_model(MODEL_PATH)


# input_path = 'INPUT/'
# output_path = 'OUTPUT/'

def inference(input_path,output_path):

    # if os.path.exists(input_path):

    #     if os.path.exists(output_path):
    #         files = os.listdir(output_path)
    #         for file in files:
    #             file_path = os.path.join(output_path, file)
    #             os.remove(file_path)
    #     else:
    #         os.makedirs(output_path)
    
    # else:
    #     print("Input path is not defined.")
    #     return 

    
    input_files = os.listdir(input_path)

    for file in input_files:

        image = os.path.join(input_path,file)

        img = cv2.imread(image)
        img = np.expand_dims(img,axis=0)

        pred = model.predict(img)
        pred = (pred > 0.5).astype(np.uint8)
        pred = np.squeeze(pred,axis=0)
        pred_binary = (pred * 255).astype(np.uint8)

        
        out_path = os.path.join(output_path + '/'+ 'predicted_mask.png')
        cv2.imwrite(out_path, pred_binary)
        
        print(f"Output saved to {output_path}")


if __name__ == "__main__": 

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    inference(input_path,output_path)
