import numpy as np
from scipy.ndimage import zoom
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class CNNDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path1, df, shape=(256, 256), batch_size=1, num_classes=None, shuffle=True):
        self.path1 = path1
        # self.path2 = path2
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
        # X_original, X_sinogram, y = self.__get_data(batch)
        # return [X_original, X_sinogram], y
        X_original, y = self.__get_data(batch)
        return [X_original], y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X_original = np.zeros((len(batch), *self.shape), dtype='float32')
        # X_sinogram = np.zeros((len(batch), *self.shape), dtype='float32')
        y = np.zeros((len(batch), self.num_classes), dtype='float32')

        for i, id in enumerate(batch):
            try:
                # Load original image data
                original_image = np.load(self.path1 + '/' + self.df.loc[id, 'Image'] + '.npy')
                X_original[i, :, :] = original_image[:,:,0];

                # Load sinogram image data
                # sinogram_image = np.load(self.path2 + '/' + self.df.loc[id, 'Image'] + '.npy')
                # a = sinogram_image[:,:,0]
                # resized_image = cv2.resize(a, (256, 256))
                # X_sinogram[i, :, :] = resized_image

                # Labels
                y[i, :] = self.df.loc[id, ['any']].to_numpy(dtype='float32')
            except Exception as e:
                # If an error occurs, display the image_id and the error message
                print(f"Error loading image with image_id: {self.df.loc[id, 'Image']}, Error: {str(e)}")

        # return X_original, X_sinogram, y    
        return X_original, y    

