
import os
import numpy as np

def create_folder(dirName):
    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")

def normalize(images):
    m = np.max(images)
    mi = np.min(images)
    images = (images - mi) / (m - mi)
    return images
