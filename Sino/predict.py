import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization
from tf_explain.core.grad_cam import GradCAM
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from keras import backend as K
from model import EfficientNetB0  # Assuming you have your EfficientNetB0 model defined in 'model.py'

# Load your test dataset (assuming you have already loaded it)
test_yn = pd.read_csv("data/test250.csv")
test_dataset = CNNDataGeneratorTest(df=test_yn, path2="data/sinograms_3W_train", shape=(256, 256), batch_size=1, num_classes=1, shuffle=False)

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
model.load_weights("best_model_weights_dgx.h5")

# Initialize GradCAM
grad_cam = GradCAM()

# Function to visualize Grad-CAM
def visualize_grad_cam(model, image_path, class_index, layer_name):
    img = np.load(image_path)

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
    plt.imshow(superimposed_img)

# Define your evaluation loop
results = []

for X, y, image_ids in tqdm(test_dataset):
    predictions = model.predict(X)
    for id, pred in zip(image_ids, predictions):
        results.append([id, *pred])

# Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(results, columns=['ImageID', 'Prediction1'])
results_df.to_csv('sino_predictions.csv', index=False)

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
