import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import time
import math
import json

base_dir = 'N:\\Halite'
train_dir = os.path.join(base_dir, 'TRAIN')

train_fnames = os.listdir(train_dir)
print(train_fnames[:10])

# Our input feature map is 200x200x3: 200x200 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(10, 10, 3))

# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(img_input)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(16, activation='relu')(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(5, activation='sigmoid')(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully
# connected layer + sigmoid output layer
model = Model(img_input, output)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(10, 10),  # All images will be resized to 200x200
        batch_size=10,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

tensor_board = tf.keras.callbacks.TensorBoard(log_dir="N:\\Halite\\Logs\\{}".format(time.time()))
model_save = tf.keras.callbacks.ModelCheckpoint(
    'N:\\Halite\\Models\\v1Checkpoint.h5',
    monitor='acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

label_map = (train_generator.class_indices)

lol = json.dumps(label_map)

#classes = train_generator.class_indices
#with open('E:\\Projects\\COD Head Spotter\\Models\\Classes.json', 'w') as outfile:
    #json.dump(classes, outfile)
#print(classes)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=math.floor(train_generator.samples / train_generator.batch_size),   # 2000 images = batch_size * steps
      epochs=100,
      callbacks=[tensor_board, model_save],
      verbose=1)



model.save('N:\\Halite\\Models\\v1.h5')