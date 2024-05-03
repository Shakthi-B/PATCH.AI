import torch
import torchvision.models as models
import numpy as np
import pandas as pd
import os,io
import logging
import sys
import tensorflow as tf
from keras import layers
import geopandas as gpd
from shapely.geometry import Point


class LogStream(io.StringIO):
    def write(self, msg):
        logging.info(msg)


stdout_logger = LogStream()
stderr_logger = LogStream()

sys.stdout = stdout_logger
sys.stderr = stderr_logger
logging.basicConfig(filename='test_log.txt', level=logging.INFO, encoding='utf-8')


file_path = r'F:/SEM-8/GIS/Dataset/trainLabels.csv'
df = pd.read_csv(file_path)
diagnosis_dict_binary = {
    0: 'Forest',
    1: 'Background'
}

base_dir = r'F:\\SEM-8\\GIS\\IoT Jcomp\\Dataset'

# train_dir = os.path.join(base_dir, 'train')
# #val_dir = os.path.join(base_dir, 'val')
# test_dir = os.path.join(base_dir, 'test')

# src_dir = r'F:/SEM-8/GIS/Dataset/train'

# train_batches = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255).flow_from_directory(train_dir, target_size=(224,224), shuffle = True)
# #val_batches = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255).flow_from_directory(val_dir, target_size=(224,224), shuffle = True)
# test_batches = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255).flow_from_directory(test_dir, target_size=(224,224), shuffle = False)

shapefile_path = "gis.shp"
gdf = gpd.read_file(shapefile_path)

class_labels = gdf['class']

points = gdf.geometry

coordinates = np.column_stack((points.x, points.y))

model = tf.keras.Sequential([
    layers.Conv2D(8, (3,3), padding="valid", input_shape=(224,224,3), activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.BatchNormalization(),
    
    layers.Conv2D(16, (3,3), padding="valid", activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.BatchNormalization(),
    
    layers.Conv2D(32, (4,4), padding="valid", activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.BatchNormalization(),
    
    layers.Conv2D(64, (4,4), padding="valid", activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.BatchNormalization(),
 
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dropout(0.15),
    layers.Dense(2, activation = 'softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Fit the model
history = model.fit(train_batches,
                    epochs=15,
                    validation_data=test_batches)

# with open('training_log.txt', 'a') as log_file:
#     log_file.write(stdout_logger.getvalue())
#     log_file.write(stderr_logger.getvalue())

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

print("Model_Saved")


# #Load Json
# #Load the model architecture from the JSON file
# with open("model.json", "r") as json_file:
#     loaded_model_json = json_file.read()

# loaded_model = tf.keras.models.model_from_json(loaded_model_json)


# # Load the weights into the model
# with open("model_weights.bin", "rb") as bin_file:
#     for layer in loaded_model.layers:
#         if isinstance(layer, tf.keras.layers.BatchNormalization):
#             # For BatchNormalization layers, load gamma and beta
#             gamma_beta = np.fromfile(bin_file, dtype=np.float32, count=2 * layer.input[-1].shape[-1])
#             gamma = gamma_beta[:layer.input[-1].shape[-1]]
#             beta = gamma_beta[layer.input[-1].shape[-1]:]
#             moving_mean = np.fromfile(bin_file, dtype=np.float32, count=layer.input[-1].shape[-1])
#             moving_variance = np.fromfile(bin_file, dtype=np.float32, count=layer.input[-1].shape[-1])

#             layer.set_weights([gamma, beta, moving_mean, moving_variance])
#         else:
#             # For other layers, load weights as usual
#             layer_weights = [np.fromfile(bin_file, dtype=np.float32, count=np.prod(param.shape)).reshape(param.shape)
#                              for param in layer.trainable_variables]
#             layer.set_weights(layer_weights)

# # Convert the model
# converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
# tflite_model = converter.convert()

# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)

# loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#                      loss=tf.keras.losses.BinaryCrossentropy(),
#                      metrics=['accuracy'])

# loss, acc = loaded_model.evaluate(test_batches, verbose=1)
# print("Loss: ", loss)
# print("Accuracy: ", acc)


model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load("your_trained_model_weights.pth"))

# Set the model to evaluation mode
model.eval()

# Input tensor shape (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 224, 224)  # Assuming input size is (224, 224) and 3 channels (RGB)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "landcover.onnx", verbose=True)