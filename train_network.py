import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import time
import math
import json
from collections import Counter
from sklearn.utils import class_weight
import numpy as np

base_dir = 'N:\\Halite'
model_dir = 'Models'
tensorboard_dir = 'Logs'
model_name = 'v3'
train_dir = os.path.join(base_dir, 'TRAIN')
target_image_size = (10, 10)

train_datagen = ImageDataGenerator(validation_split=0.1)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_image_size,
        batch_size=100,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_image_size,
        batch_size=100,
        class_mode='categorical',
        subset='validation')

#counter = Counter(train_generator.classes)
#max_val = float(max(counter.values()))
#class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}
#class_weights= {}
#pre_class_weights = class_weight.compute_class_weight(
              # 'balanced',
               # np.unique(train_generator.classes),
               # train_generator.classes)

#for x in range(0, train_generator.num_classes):
    #class_weights[x] = pre_class_weights[x]

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(target_image_size[0], target_image_size[1], 3)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(
                #learning_rate=0.0000001
            ),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tensor_board = tf.keras.callbacks.TensorBoard(log_dir=f"{base_dir}\\{tensorboard_dir}\\{time.time()}")
model_save = tf.keras.callbacks.ModelCheckpoint(
        f'{base_dir}\\{model_dir}\\{model_name}_Checkpoint.h5',
        monitor='accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

classes = train_generator.class_indices
with open(f'{base_dir}\\{model_dir}\\{model_name}_Classes.json', 'w') as outfile:
        json.dump(classes, outfile)
print(classes)

# 2000 images = batch_size * steps
history = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        validation_steps=math.floor(validation_generator.samples / validation_generator.batch_size),
        steps_per_epoch=math.floor(train_generator.samples / train_generator.batch_size),
        epochs=800,
        callbacks=[tensor_board, model_save],
        verbose=1,
        #class_weight=class_weights
)

model.save(f'{base_dir}\\{model_dir}\\{model_name}_Complete.h5')