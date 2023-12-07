import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.utils import Sequence
from tf_explain.core.grad_cam import GradCAM
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Define your data generators

class CNNDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path1, df, shape=(256, 256), batch_size=1, num_classes=None, shuffle=True):
        self.path1 = path1
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
        X_original, y = self.__get_data(batch)
        return [X_original], y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X_original = np.zeros((len(batch), *self.shape), dtype='float32')
        y = np.zeros((len(batch), self.num_classes), dtype='float32')

        for i, id in enumerate(batch):
            try:
                original_image = np.load(self.path1 + '/' + self.df.loc[id, 'Image'] + '.npy')
                X_original[i, :, :] = original_image[:, :, 0]

                y[i, :] = self.df.loc[id, ['any']].to_numpy(dtype='float32')
            except Exception as e:
                print(f"Error loading image with image_id: {self.df.loc[id, 'Image']}, Error: {str(e)}")

        return X_original, y

class CNNDataGeneratorTest(tf.keras.utils.Sequence):
    def __init__(self, path2, df, shape=(256, 256), batch_size=1, num_classes=None, shuffle=True):
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
        X_sinogram, y, image_ids = self.__get_data(batch)
        return [X_sinogram], y, image_ids

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X_sinogram = np.zeros((len(batch), *self.shape), dtype='float32')
        y = np.zeros((len(batch), self.num_classes), dtype='float32')
        image_ids = []

        for i, id in enumerate(batch):
            try:
                sinogram_image = np.load(self.path2 + '/' + self.df.loc[id, 'Image'] + '.npy')
                a = sinogram_image[:, :, 0]
                resized_image = cv2.resize(a, (256, 256))
                X_sinogram[i, :, :] = resized_image

                y[i, :] = self.df.loc[id, ['any']].to_numpy(dtype='float32')

                image_ids.append(self.df.loc[id, 'Image'])

            except Exception as e:
                print(f"Error loading image with image_id: {self.df.loc[id, 'Image']}, Error: {str(e)}")

        return X_sinogram, y, image_ids

# Define your model creation and compilation function

def create_and_compile_model(input_shape, learning_rate=0.001):
    image_input = Input(shape=input_shape, name='image_input')
    image_backbone = EfficientNetB0(weights=None, include_top=False, input_tensor=image_input, input_shape=input_shape)

    for layer in image_backbone.layers:
        layer._name = 'image_' + layer.name

    image_features = GlobalAveragePooling2D()(image_backbone.output)

    x = Dense(256, activation='relu')(image_features)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)

    output = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[image_input], outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC(curve='ROC'),
                           tf.keras.metrics.AUC(curve='PR')])

    return model

# Load the pre-trained model weights
model = create_and_compile_model(input_shape2=(256, 256, 1), learning_rate=0.001)
model.load_weights("best_model_weights.h5")  #load the model_weights for prediction

# Initialize GradCAM
grad_cam = GradCAM()

# Function to visualize Grad-CAM
def visualize_grad_cam(model, image_path, class_index, layer_name):
    # Load and preprocess the image from image_path
    img = np.load(image_path + 'ID_0da09bb5c.npy')
    img = img_to_array(img[:, :, 0])

    # Generate heatmap using GradCAM
    data = ([img], None)
    heatmap = grad_cam.explain(data, model, class_index, layer_name)
    heatmap = (heatmap).astype("uint8")

    # Process heatmap and overlay on the original image
    heatmap = (heatmap * 255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_img = img
    superimposed_img = heatmap * 0.4 + original_img

    # Save or display the Grad-CAM visualization
    # cv2.imwrite('grad_cam.jpg', superimposed_img)
    plt.imshow('Grad-CAM', superimposed_img)

# Define your evaluation loop
results = []

for X, y, image_ids in tqdm(test_dataset):
    predictions = model.predict(X)
    for id, pred in zip(image_ids, predictions):
        results.append([id, *pred])

# Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(results, columns=['ImageID', 'Prediction1'])
results_df.to_csv('CT_predictions.csv', index=False)  #user must include right path to save the file

# Calculate evaluation metrics
result_metrics = {}
result_metrics['Accuracy'] = []
result_metrics['Sensitivity'] = []
result_metrics['Specificity'] = []
result_metrics['AUC'] = []
result_metrics['Precision'] = []

Precision = tf.keras.metrics.Precision()
auc = tf.keras.metrics.AUC()
classes = test_yn.columns[1]
metrics = ["Accuracy", "Sensitivity", "Specificity", "AUC", "Precision"]

for c in range(1):
    Y_preds = y_preds[:, c]
    Y_trues = y_trues[:, c]
    conf_matrix = tf.math.confusion_matrix(Y_trues, Y_preds, num_classes=2).numpy()

    result_metrics['Sensitivity'].append((conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])) * 100)
    result_metrics['Specificity'].append((conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])) * 100)
    result_metrics['Accuracy'].append((conf_matrix[0, 0] + conf_matrix[1, 1]) / (np.sum(conf_matrix)) * 100)
    
    auc.reset_states()
    auc.update_state(Y_trues, Y_preds)
    Precision.reset_states()
    Precision.update_state(Y_trues, Y_preds)
    
    result_metrics['AUC'].append(auc.result().numpy() * 100)
    result_metrics['Precision'].append(Precision.result().numpy() * 100)

result_metrics["class_strength"] = [test_yn[c].sum() for c in classes]
metric_df = pd.DataFrame(result_metrics, index=classes)
metric_df.index.name = "Class"
print(metric_df)

# Visualize Grad-CAM for specific images
# visualize_grad_cam(model, image_path, class_index, layer_name)

# Main code execution
if __name__ == "__main__":
    # Your main code execution logic here
