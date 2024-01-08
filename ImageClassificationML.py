import os
import numpy as np
import keras
from keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import data as tf_data

import matplotlib.pyplot as plt


num_skipped = 0

folder_List = ["/Users/andresangel/Desktop/kagglecatsanddogs_5340/PetImages/Cat","/Users/andresangel/Desktop/kagglecatsanddogs_5340/PetImages/Dog"]
folder_Counters = [0,0]

for folder_index, folder_path in enumerate(folder_List):
    for _,_, files in os.walk(folder_path):
        folder_Counters[folder_index] += len(files)

NumOfCatImages = folder_Counters[0]
NumOfDogImages = folder_Counters[1]

#Loop through respective files in each folder
for folder_Name in ("Cat","Dog"):
    folder_Path = os.path.join("/Users/andresangel/Desktop/kagglecatsanddogs_5340/PetImages", folder_Name)

    #Loops through folder

    for fileName in os.listdir(folder_Path):
        
        #joins the current file in loop with filepath EXAMPLE: "PetImages/Cat/0.jpg"
        filePath = os.path.join(folder_Path,fileName)
        try:
            #Open file in Reading in binary format
            fileObject = open(filePath, "rb")
            
            # Checks the first 10 bytes without opening it and sees if "JFIF" is in there
            # Common for checking if file is in JPEG Format

            is_JFIF = b"JFIF" in fileObject.peek(10)
        finally:
            fileObject.close()

        if not is_JFIF:
            num_skipped += 1
            os.remove(filePath)

    print("Dog Files in total: " + str(NumOfDogImages))
    print("Cat Files in total: " + str(NumOfCatImages))      

print(f"Deleted {num_skipped} files due to corruption or unformatted content")




