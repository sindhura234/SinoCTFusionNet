import os
import argparse
import tensorflow as tf
from model import create_and_compile_model
from data_generator import CNNDataGenerator
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger

def set_visible_gpus(gpu_devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate your model.")
    
    parser.add_argument("--data_dir", type=str, default="../data", help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="./output_CT", help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Select 'train' or 'test' mode")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated list of GPU devices to use")
    parser.add_argument("--load_weights_path", type=str, default=None, help="Path to load pre-trained model weights")
    parser.add_argument("--initial_learning_rate", type=float, default=0.001, help="Initial learning rate")
    
    return parser.parse_args()

def create_callbacks(args):
    checkpoint_path = os.path.join(args.output_dir, "checkpoints")
    tensorboard_log_dir = os.path.join(args.output_dir, "logs")
    csv_filename = os.path.join(args.output_dir, "training_metrics.csv")
    
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        os.path.join(checkpoint_path, 'best_model_weights.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        mode='min'
    )
    
    tensorboard_callback = TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_images=True,
        write_graph=True
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
    
    csv_logger = CSVLogger(csv_filename, separator=',', append=True)
    
    return [checkpoint_callback, csv_logger, early_stopping, tensorboard_callback]

def load_data(args):
    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    valid = pd.read_csv(os.path.join(args.data_dir, "valid.csv"))
    test_yn = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    return train, valid, test_yn

def main():
    args = parse_args()
    
    set_visible_gpus(args.gpus)
    
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        print("No GPUs available. Please check your GPU configuration.")
        return
    
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:" + str(i) for i in range(len(physical_devices))])
    
    with strategy.scope():
        train, valid, test_yn = load_data(args)
        
        train_dataset = CNNDataGenerator(df=train, path1=os.path.join(args.data_dir, "CT_3W_npy"), shape=(256, 256), batch_size=args.batch_size, num_classes=1, shuffle=True)
        valid_dataset = CNNDataGenerator(df=valid, path1=os.path.join(args.data_dir, "CT_3W_npy"), shape=(256, 256), batch_size=args.batch_size, num_classes=1, shuffle=True)
        test_dataset = CNNDataGenerator(df=test_yn, path1=os.path.join(args.data_dir, "CT_3W_npy"), shape=(256, 256), batch_size=args.batch_size, num_classes=1, shuffle=False)
        
        if args.mode == "train":
            model = create_and_compile_model((256, 256, 1), learning_rate=args.initial_learning_rate)
            callbacks = create_callbacks(args)
            history = model.fit(train_dataset, validation_data=valid_dataset, epochs=args.epochs, callbacks=callbacks)

            model.save(os.path.join(args.output_dir, "final_model.h5"))

            with open(os.path.join(args.output_dir, "training_history.txt"), "w") as history_file:
                history_file.write(str(history.history))
        else:
            model = create_and_compile_model((256, 256, 1), learning_rate=args.initial_learning_rate)
            if args.load_weights_path:
                model.load_weights(args.load_weights_path)
                print("Loaded the weights successfully")

            test_model(args, test_dataset)

if __name__ == "__main__":
    main()
