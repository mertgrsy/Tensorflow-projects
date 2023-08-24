# # Download zip file of pizza_steak images
# !wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random, os
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras import Sequential
import tensorflow as tf

def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)

  print(f"Image shape: {img.shape}") # show the shape of the image
  return img

target_size = (224, 224)
batch_size = 32
input_shape = (224, 224, 3)

# Visualize data (requires function 'view_random_image' above)
plt.figure()
plt.subplot(1, 2, 1)
steak_img = view_random_image("data/pizza_steak/train/", "steak")
plt.subplot(1, 2, 2)
pizza_img = view_random_image("data/pizza_steak/train/", "pizza")
plt.show()

# Define training and test directory paths
train_dir = "data/pizza_steak/train/"
test_dir = "data/pizza_steak/test/"

# Create train and test data generators and rescale the data 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

# Turn it into batches
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size=target_size,
                                               class_mode='binary',
                                               batch_size=batch_size)

test_data = test_datagen.flow_from_directory(directory=test_dir,
                                             target_size=target_size,
                                             class_mode='binary',
                                             batch_size=batch_size)
# Create the model_1
model_1 = Sequential([
  Conv2D(filters=10, 
         kernel_size=3, 
         strides=1,
         padding='valid',
         activation='relu', 
         input_shape=(224, 224, 3)), # input layer (specify input shape)
  Conv2D(10, 3, activation='relu'),
  Conv2D(10, 3, activation='relu'),
  Dropout(0.25),
  Flatten(),
  Dense(1, activation='sigmoid') # output layer (specify output shape)
])
model_1.summary()

# Compile the model
model_1.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

# Fit the model
history = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))
# Plot the training curves
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(10, 7))
plt.show()