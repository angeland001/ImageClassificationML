import os
import numpy as np
import keras
from keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import data as tf_data

import matplotlib.pyplot as plt


num_skipped = 0

#Loop through respective files in each folder
for folder_Name in ("Cat","Dog"):
    folder_Path = os.path.join(r"C:\Users\angel\Desktop\kagglecatsanddogs_5340\PetImages", folder_Name)

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

     

print(f"Deleted {num_skipped} files due to corruption or unformatted content")


imageSize = (180,180)
batchSize = 128

#Seperate Training set and Testing Validation
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    r"C:\Users\angel\Desktop\kagglecatsanddogs_5340\PetImages",
    # Optional float between 0 and 1, fraction of data to reserve for validation.
    # 20 Percent for Testing/Labels
    validation_split = 0.2,
    
    # When subset="both", the utility returns a tuple of two datasets 
    # (the training and validation datasets respectively).
    
    subset = "both",
    seed=1337,
    image_size = imageSize,
    batch_size = batchSize,
)

plt.figure(figsize=(8,8))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
        plt.show()