
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 21:43:30 2023

@author: ee18d504
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB1
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate, BatchNormalization    
# Define input shape for each modality
input_shape1 = (256, 256, 1)
input_shape2 = (256, 256, 1)

def create_and_compile_model(input_shape1, input_shape2, learning_rate=0.001):
    # Create EfficientNetB0 backbone for the image modality
    image_input = Input(shape=input_shape1, name='image_input')
    image_backbone = EfficientNetB0(weights=None, include_top=False, input_tensor=image_input, input_shape=input_shape1)

    # Rename all layers inside the image backbone
    for layer in image_backbone.layers:
        layer._name = 'image_' + layer.name  # Add 'image_' prefix to layer names

    # Create EfficientNetB0 backbone for the sinogram modality
    sinogram_input = Input(shape=input_shape2, name='sinogram_input')
    sinogram_backbone = EfficientNetB0(weights=None, include_top=False, input_tensor=sinogram_input, input_shape=input_shape2)

    # Rename all layers inside the sinogram backbone
    for layer in sinogram_backbone.layers:
        layer._name = 'sinogram_' + layer.name  # Add 'sinogram_' prefix to layer names

    # Extract features from the image backbone
    image_features = GlobalAveragePooling2D()(image_backbone.output)

    # Extract features from the sinogram backbone
    sinogram_features = GlobalAveragePooling2D()(sinogram_backbone.output)

    # Concatenate the features
    concatenated_features = Concatenate()([image_features, sinogram_features])

    # Add additional layers if needed
    x = Dense(256, activation='relu')(concatenated_features)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    
    output = Dense(1, activation='sigmoid')(x)  # For binary classification

    # Create the final model
    model = tf.keras.Model(inputs=[image_input, sinogram_input], outputs=output)

    # Compile the model with appropriate loss, optimizer, and learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                                                                       tf.keras.metrics.SpecificityAtSensitivity(0.5)])
    model.compile(optimizer=optimizer,
              loss='binary_crossentropy',  # or another appropriate loss
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       # F1Score(),
                       tf.keras.metrics.AUC(curve='ROC'),
                       tf.keras.metrics.AUC(curve='PR')])

    return model

# # Example usage:
model = create_and_compile_model(input_shape1=(256, 256, 1), input_shape2=(256, 256, 1), learning_rate=0.001)
model.summary()
