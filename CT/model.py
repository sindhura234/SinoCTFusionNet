import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, BatchNormalization

# Define input shape for the sinogram modality
input_shape2 = (256, 256, 1)

def create_and_compile_model(input_shape, learning_rate=0.001):
    # Create EfficientNetB0 backbone for the image modality
    image_input = Input(shape=input_shape, name='image_input')
    image_backbone = EfficientNetB0(weights=None, include_top=False, input_tensor=image_input, input_shape=input_shape)

    # Rename all layers inside the image backbone
    for layer in image_backbone.layers:
        layer._name = 'image_' + layer.name  # Add 'image_' prefix to layer names

    # Extract features from the image backbone
    image_features = GlobalAveragePooling2D()(image_backbone.output)

    # Add additional layers if needed
    x = Dense(256, activation='relu')(image_features)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    
    output = Dense(1, activation='sigmoid')(x)  # For binary classification

    # Create the final model
    model = tf.keras.Model(inputs=[image_input], outputs=output)

    # Compile the model with appropriate loss, optimizer, and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
              loss='binary_crossentropy',  # or another appropriate loss
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       # F1Score(),
                       tf.keras.metrics.AUC(curve='ROC'),
                       tf.keras.metrics.AUC(curve='PR')])

    return model

# Example usage:
model = create_and_compile_model(input_shape=(256, 256, 1), learning_rate=0.001)
model.summary()
