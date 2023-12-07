
import numpy as np
from tqdm import tqdm 
import pandas as pd

import numpy as np
from scipy.ndimage import zoom
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from model import *

#%%
class CNNDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path1, path2, df, shape=(256, 256), batch_size=1, num_classes=None, shuffle=True):
        self.path1 = path1
        self.path2 = path2
        self.shape = shape
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indices) // self.batch_size)

    def __getitem__(self, index):
        indexes = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in indexes]
        X_original, X_sinogram, y = self.__get_data(batch)
        return [X_original, X_sinogram], y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X_original = np.zeros((len(batch), *self.shape), dtype='float32')
        X_sinogram = np.zeros((len(batch), *self.shape), dtype='float32')
        y = np.zeros((len(batch), self.num_classes), dtype='float32')

        for i, id in enumerate(batch):
            try:
                # Load original image data
                original_image = np.load(self.path1 + '/' + self.df.loc[id, 'Image'] + '.npy')
                X_original[i, :, :] = original_image[:,:,0];

                # Load sinogram image data
                sinogram_image = np.load(self.path2 + '/' + self.df.loc[id, 'Image'] + '.npy')
                a = sinogram_image[:,:,0]
                resized_image = cv2.resize(a, (256, 256))
                X_sinogram[i, :, :] = resized_image

                # Labels
                y[i, :] = self.df.loc[id, ['any']].to_numpy(dtype='float32')
            except Exception as e:
                # If an error occurs, display the image_id and the error message
                print(f"Error loading image with image_id: {self.df.loc[id, 'Image']}, Error: {str(e)}")

        return X_original, X_sinogram, y
    
class CNNDataGeneratorTest(tf.keras.utils.Sequence):
    def __init__(self, path1, path2, df, shape=(256, 256), batch_size=1, num_classes=1, shuffle=True):
        self.path1 = path1
        self.path2 = path2
        self.shape = shape
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indices) // self.batch_size)

    def __getitem__(self, index):
        indexes = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in indexes]
        X_original, X_sinogram, y, image_ids = self.__get_data(batch)
        return [X_original, X_sinogram], y, image_ids  # Now also returns image IDs

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X_original = np.zeros((self.batch_size, *self.shape), dtype='float32')
        X_sinogram = np.zeros((self.batch_size, *self.shape), dtype='float32')
        y = np.zeros((self.batch_size, self.num_classes), dtype='float32')

        image_ids = []  # List to hold image IDs

        for i, id in enumerate(batch):
            try:
                # Load original image data
                original_image = np.load(self.path1 + '/' + self.df.loc[id, 'Image'] + '.npy')
                X_original[i, :, :] = original_image[:,:,0]

                # Load sinogram image data
                sinogram_image = np.load(self.path2 + '/' + self.df.loc[id, 'Image'] + '.npy')
                a = sinogram_image[:,:,0]
                resized_image = cv2.resize(a, (256, 256))
                X_sinogram[i, :, :] = resized_image

                # Labels
                y[i, :] = self.df.loc[id, ['any']].to_numpy(dtype = 'float32') 

                # Append image ID to the list
                image_ids.append(self.df.loc[id, 'Image'])

            except Exception as e:
                print(f"Error loading image with image_id: {self.df.loc[id, 'Image']}, Error: {str(e)}")

        return X_original, X_sinogram, y, image_ids  # Return image IDs along with the data
#%%
df = pd.read_csv("data/test250.csv")
unique_patients = df['PatientID'].unique()
selected_patients = unique_patients[:100]
selected_data = df[df['PatientID'].isin(selected_patients)]


test_yn= pd.read_csv("data/test250.csv")

test_dataset = CNNDataGenerator(df=test_yn, 
                                path1="data/CT_3W_npy", 
                                path2="data/sinograms_3W_train", shape=(256, 256), batch_size=1, num_classes=1, shuffle=False)

model = create_and_compile_model(input_shape1=(256, 256, 1), input_shape2=(256, 256, 1), learning_rate=0.001)
model.summary()

model.load_weights("best_model_weights.h5")
model.evaluate(test_dataset)
#%%
# # data_generator = CNNDataGenerator(path1, path2, df, batch_size=32)
# # for X, y, image_ids in test_dataset:
# #     predictions = model.predict(X)
# #     print(image_ids,predictions)

# # import pandas as pd

# Assuming 'test_dataset' is your data generator and 'model' is your trained model
results = []

for X, y, image_ids in tqdm(test_dataset):
    predictions = model.predict(X)
    for id, pred in zip(image_ids, predictions):
        results.append([id, *pred])  # Flatten prediction array if needed

# Convert results to a DataFrame
results_df = pd.DataFrame(results, columns=['ImageID', 'Prediction1'])#, 'Prediction2', 'Prediction3', 'Prediction4', 'Prediction5', 'Prediction6'])

# Save to CSV
results_df.to_csv('SinoCT_predictions.csv', index=False)
