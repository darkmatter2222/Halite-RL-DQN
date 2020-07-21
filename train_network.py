import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import time
import math
import json
from collections import Counter

base_dir = 'N:\\Halite'
model_dir = 'Models'
tensorboard_dir = 'Logs'
model_name = 'v3'
train_dir = os.path.join(base_dir, 'TRAIN')
target_image_size = (10, 10)

train_datagen = ImageDataGenerator(validation_split=0.25)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_image_size,
        batch_size=10,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_image_size,
        batch_size=10,
        class_mode='categorical',
        subset='validation')

counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(target_image_size[0], target_image_size[1], 3)),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    #tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_generator.num_classes, activation='sigmoid')
])

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.0001),
              loss='categorical_crossentropy',
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
        class_weight=class_weights)

model.save(f'{base_dir}\\{model_dir}\\{model_name}_Complete.h5')