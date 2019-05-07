
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pdb
import pickle


DATADIR = "/Users/dalelee/desktop/IW_melanoma/dcgan/benign" # ISIC-images/UDA-1", dcgan/malignant, benign

path = os.path.join(DATADIR,"")
# for img in os.listdir(path):  # iterate over each image per dogs and cats
# 	pdb.set_trace()
# 	if (img != '.DS_Store'):
# 		img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
# 		plt.imshow(img_array, cmap='gray')  # graph it
# 		plt.show()  # display!

# 		IMG_SIZE = 100
# 		new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# 		plt.imshow(new_array, cmap='gray')
# 		plt.show()

training_data = []
IMG_SIZE = 56 # 28
for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
    try:
        img_array = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE)  # convert to array, cv2.IMREAD_GRAYSCALE cv2.IMREAD_COLOR
        # pdb.set_trace()
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
        # new_array = img_array # instead of resize
        # plt.imshow(new_array, cmap='gray') # remove cmap='gray' if rgb
        # plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
        # plt.show()
        # training_data.append(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
        training_data.append(new_array)  # add this to our training_data
    except Exception as e:
    	print(e)

X = []

for features in training_data:
    X.append(features)

# print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # 1, 3
pickle_out = open("benign_final.pickle","wb") # benign.pickle, X.pickle, malignant_56x56_rgb
pickle.dump(X, pickle_out)
pickle_out.close()


