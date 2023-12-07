
import os
import argparse
import tensorflow as tf
from model import create_and_compile_model
from data_generator import CNNDataGenerator,CNNDataGeneratorTest
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau


def set_visible_gpus(gpu_devices):
    # Set the CUDA_VISIBLE_DEVICES environment variable to the selected GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate your model.")
    
    # Add command-line arguments
    parser.add_argument("--data_dir", type=str, default="../data", help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="./output_sinoCT", help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Select 'train' or 'test' mode")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated list of GPU devices to use")
    parser.add_argument("--load_weights_path", type=str, default=None, help="Path to load pre-trained model weights")
    parser.add_argument("--initial_learning_rate", type=float, default=0.001, help="Initial learning rate")
    # parser.add_argument("--traincsv", type=str, default=None, help="Path to csv files")
    # parser.add_argument("--validcsv", type=str, default=None, help="Path to csvfiles")
    return parser.parse_args()

def train_model(args, train_dataset, valid_dataset):
    # Define input shape for each modality
    input_shape = (256, 256, 1)
    
    # Create and compile the model
    model = create_and_compile_model(input_shape, input_shape, learning_rate=args.initial_learning_rate)  # Set an initial learning rate
    
    # Define paths for checkpoints, logs, and CSV file
    checkpoint_path = os.path.join(args.output_dir, "checkpoints")
    tensorboard_log_dir = os.path.join(args.output_dir, "logs")
    csv_filename = os.path.join(args.output_dir, "training_metrics.csv")
    
    # tensorboard_log_dir = os.path.join(args.output_dir, "logs_tensorboard")
    # os.makedirs(tensorboard_log_dir, exist_ok=True)

    # Create the directories if they don't exist
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    # Save checkpoints with timestamps and epoch numbers
    checkpoint_callback = ModelCheckpoint(
        os.path.join(checkpoint_path, 'best_model_weights.h5'),
        monitor='val_loss',
        save_best_only=True,  # Save all weights
        save_weights_only=True,
        verbose=1,
        mode='min'
    )
   
    tensorboard_callback = TensorBoard(
    log_dir=tensorboard_log_dir,
    histogram_freq=1,  # How often to compute histograms (set to 1 for every epoch)
    write_images=True,  # Save model architecture as an image
    write_graph=True  # Write the computation graph to a file
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=8, 
                                   mode='min', verbose=1)
    
    # Create a CSVLogger to log training metrics to a CSV file
    csv_logger = CSVLogger(csv_filename, separator=',', append=True)
    
    callbacks = [checkpoint_callback, csv_logger, early_stopping, tensorboard_callback]
    
    # Train the model
    history = model.fit(train_dataset, validation_data=valid_dataset, 
                        epochs=args.epochs, #steps_per_epoch=steps_per_epoch, validation_steps = validation_steps, 
                        callbacks=callbacks)

    # Save the trained model
    model.save(os.path.join(args.output_dir, "final_model.h5"))
    
    # Save training history to a text file
    with open(os.path.join(args.output_dir, "training_history.txt"), "w") as history_file:
        history_file.write(str(history.history))

def test_model(args, test_dataset):
    # Define input shape for each modality
    input_shape = (256, 256, 1)
    
    # Create and compile the model
    model = create_and_compile_model(input_shape,input_shape,learning_rate=args.initial_learning_rate)
    
    # Load pre-trained weights if provided
    if args.load_weights_path:
        model.load_weights(args.load_weights_path)
        # model.load_weights(os.path.join(args.output_dir, "final_model.h5"))
        print("Loaded the weights successfully")
    
    # Evaluate the model on the test dataset
    test_loss, test_accuracy, test_auc, test_sensitivity, test_specificity = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test AUC: {test_auc}")
    print(f"Test Sensitivity: {test_sensitivity}")
    print(f"Test Specificity: {test_specificity}")
    
    # Create a dictionary to store the test results
    test_results = {
        "Test Loss": [test_loss],
        "Test Accuracy": [test_accuracy],
        "Test AUC": [test_auc],
        "Test Sensitivity": [test_sensitivity],
        "Test Specificity": [test_specificity]
    }
    
    # Convert the dictionary into a Pandas DataFrame
    test_results_df = pd.DataFrame(test_results)
    
    # Specify the path for the CSV file
    result_csv_path = os.path.join(args.output_dir, "test_results.csv")
    
    # Save the DataFrame to the CSV file
    test_results_df.to_csv(result_csv_path, index=False)

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set the CUDA_VISIBLE_DEVICES environment variable to limit GPU visibility
    set_visible_gpus(args.gpus)
    
    # Check if there are GPUs available
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        print("No GPUs available. Please check your GPU configuration.")
        return
    
    # Create a MirroredStrategy with all available GPUs
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:" + str(i) for i in range(len(physical_devices))])
    
    with strategy.scope():
        train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
        valid = pd.read_csv(os.path.join(args.data_dir, "valid.csv"))
        test_yn = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        
        train_dataset = CNNDataGenerator(df=train, path1=os.path.join(args.data_dir, "CT_3W_npy"), path2=os.path.join(args.data_dir, "sinograms_3W_train"), shape=(256, 256), batch_size=args.batch_size, num_classes=1, shuffle=True)
        valid_dataset = CNNDataGenerator(df=valid, path1=os.path.join(args.data_dir, "CT_3W_npy"), path2=os.path.join(args.data_dir, "sinograms_3W_train"), shape=(256, 256), batch_size=args.batch_size, num_classes=1, shuffle=True)
        test_dataset = CNNDataGenerator(df=test_yn, path1=os.path.join(args.data_dir, "CT_3W_npy"), path2=os.path.join(args.data_dir, "sinograms_3W_train"), shape=(256, 256), batch_size=args.batch_size, num_classes=1, shuffle=False)
        
        if args.mode == "train":
            train_model(args, train_dataset, valid_dataset)
        else:
            test_model(args, test_dataset)

if __name__ == "__main__":
    main()