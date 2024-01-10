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
columns = 5
rows = 2

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
        
        
#Apply transformations to training images to help expose
#Model to different aspects of training data while slowing down overfitting

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
        
    return images

#visualize the new augmented images
for images, _ in train_ds.take(1):
    for i in range(1, columns*rows + 1):
        augmented_images= data_augmentation(images)
        ax = plt.subplot(rows,columns,i)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")
    plt.show()    
    
#will standarize values to be in the [0,1] by using the rescaling
#layer at the start of the model

#Preprocess Data

#map function applies the function to every element in array (in this case training data)

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=imageSize + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)

epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

img = keras.utils.load_img("PetImages/Cat/6779.jpg", target_size=imageSize)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")