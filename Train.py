import database_interface
import numpy as np
import json
import tensorflow as tf
import logging
import time
import os

logging.getLogger().setLevel(logging.INFO)
database = database_interface.database

logging.info('Download Data')
raw_db_data = database.get(database)
logging.info('Download Data Complete')

logging.info('Extract Data')
raw_list_data = []
raw_list_labels = []
LABELS = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3, "NOTHING": 4}
for record in raw_db_data:
    raw_list_data.append(json.loads(record[2]))
    raw_list_labels.append(LABELS[record[3]])
logging.info('Extract Data Complete')

logging.info('Slice Data')

nf_of = int(len(raw_list_data) * .80)
training_data = raw_list_data[0: nf_of]
validation_data = raw_list_data[nf_of:]
training_labels = raw_list_labels[0: nf_of]
validation_labels = raw_list_labels[nf_of:]

#training_data = raw_list_data[0: int(len(raw_list_data) * .95)]
#validation_data = raw_list_data[int(len(raw_list_data) * .95):]
#training_labels = raw_list_data[0: int(len(raw_list_labels) * .95)]
#validation_labels = raw_list_data[int(len(raw_list_labels) * .95):]
training_data = training_data
training_labels = training_labels
validation_data = validation_data
validation_labels = validation_labels
logging.info('Slice Data Complete')

logging.info('Normalize Data')
normalized_training_data = []
normalized_training_data = tf.keras.utils.normalize(training_data)
normalized_validation_data = []
normalized_validation_data = tf.keras.utils.normalize(validation_data)
logging.info('Normalize Data Complete')

logging.info('Compress/Shuffle Data')
#training_dataset = tf.data.Dataset.from_tensor_slices((normalized_training_data, np.array(training_labels)))
#training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
#training_dataset = training_dataset.repeat(20)
#training_dataset = training_dataset.batch(1024)

#validation_dataset = tf.data.Dataset.from_tensor_slices((normalized_validation_data, np.array(validation_labels)))
logging.info('Compress/Shuffle Data Complete')

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(5, 10, 10)),
    #tf.keras.layers.Dense(32, activation='relu'),
    #tf.keras.layers.Conv2D(50, 50, padding='same', activation='relu', input_shape=(5, 10, 10)),
    #tf.keras.layers.MaxPooling2D(),
    #tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(5)
])

tensor_board = tf.keras.callbacks.TensorBoard(log_dir=os.path.realpath('..')+"\\Logs\\{}".format(time.time()))

model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanAbsoluteError(),
                metrics=['accuracy'])

model.fit(x=normalized_training_data, y=np.array(training_labels), validation_data=(normalized_validation_data, np.array(validation_labels)),
          epochs=100000,
          batch_size=2048,
          callbacks=[tensor_board],
          verbose=1)

model.save("T1.h5")
