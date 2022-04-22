# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import layers

import numpy as np
import random
import os
import re
import json
import argparse
import glob
import boto3

HEIGHT = 224
WIDTH  = 224
DEPTH = 3
NUM_CLASSES = 8
LAST_FROZEN_LAYER = 20

TF_VERSION = tf.version.VERSION
print('TF version: {}'.format(tf.__version__))
print('Keras version: {}'.format(tensorflow.keras.__version__))

def get_filenames(input_data_dir):
    return glob.glob('{}/*.tfrecords'.format(input_data_dir))

def _parse_image_function(example):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.image.resize(image, [WIDTH, HEIGHT])
    
    image = preprocess_input(image)
    
    prob = random.uniform(0, 1)
    if prob < .25:
        image=tf.image.rot90(image,k=2)
#         image = tf.image.random_flip_left_right(image)
    elif prob < .5:
        image=tf.image.rot90(image,k=3)
#         image = tf.image.random_flip_up_down(image)
    elif prob < .75:
        image = tf.image.random_saturation(image, 0.6, 1.6)
        image = tf.image.random_hue(image, 0.08)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.7, 1.3)
    else:
        image=tf.image.rot90(image,k=1)
        
    label = tf.cast(features['label'], tf.int32)
    
    label = tf.one_hot(label, NUM_CLASSES)

    return image, label

def read_dataset2(channel, args):
    mode = args.data_config[channel]["TrainingInputMode"]
    
    print("Running {} in {} mode".format(channel, mode))

    if mode == "Pipe":
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(channel=channel, record_format="TFRecord")
    else:
        if channel == 'train':
            filenames = get_filenames(args.train)
        else:
            filenames = get_filenames(args.valid)
        
        print(filenames)
        dataset = tf.data.TFRecordDataset(filenames)

    # Repeat infinitely.
    dataset = dataset.repeat()
    dataset = dataset.prefetch(args.batch_size)

    # Parse records.
    dataset = dataset.map(_parse_image_function, num_parallel_calls=10)

    # Potentially shuffle records.
    if channel == "train":
        # Ensure that the capacity is sufficiently large to provide good random shuffling.
        buffer_size = int(1000 * 0.4) + 3 * args.batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch it up.
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

def read_dataset(channel, args):
    
    mode = args.data_config[channel]['TrainingInputMode']

    
    if mode.lower() == 'pipe':
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(channel=channel, record_format='TFRecord')
    else:
        if channel == 'train':
            filenames = get_filenames(args.train)
        else:
            filenames = get_filenames(args.valid)
            
        print(filenames)
        dataset = tf.data.TFRecordDataset(filenames)
        
    dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(500)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)

    
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

## Load the weights from the latest checkpoint
def load_weights_from_latest_checkpoint(model, checkpoint_path):
    file_list = os.listdir(checkpoint_path)
    print('GPU # {} :: Checking for checkpoint files...')#.format(sdp.rank()))
    if len(file_list) > 0:
        print('GPU # {} :: Checkpoint files found.') #.format(sdp.rank()))
        print('GPU # {} :: Loading the weights from the latest model checkpoint...')#.format(sdp.rank()))
        model.load_weights(tf.train.latest_checkpoint(args.checkpoint_path))
#         logger.info('GPU # {} :: Completed loading weights from the latest model checkpoint.'.format(sdp.rank()))
    else:
         print('GPU # {} :: Checkpoint files not found.')#.format(sdp.rank()))
    
    return model

def save_model_artifacts(model, model_dir):
    print(f'Saving model to {model_dir}...')
    # Note that this method of saving does produce a warning about not containing the train and evaluate graphs.
    # The resulting saved model works fine for inference. It will simply not support incremental training. If that
    # if needed, one can use model checkpoints and save those.
    
    model.save(f'{model_dir}/1', save_format='tf')

    print('...DONE saving model!')

## Transfer learning an existing network
def make_model(dropout, num_fully_connected_layers, initial_lr, num_classes):

    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', 
                      include_top=False, 
                      input_shape=(HEIGHT, WIDTH, DEPTH))
    
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    
    x = Flatten()(x)
    
    fully_connected_layers = []
    for i in range(num_fully_connected_layers):
        fully_connected_layers.append(1024)
 
    for fc in fully_connected_layers:
        x = Dense(fc, activation='relu')(x) 
        if (dropout != 0.0):
            x = Dropout(dropout)(x)
    
    predictions = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=[base_model.input], outputs=predictions)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=initial_lr), 
                  metrics=['accuracy'])
    
    return model

## Train the model
def train_model(model, train_dataset, valid_dataset, lr, batch_size, epochs, call_back=False):
    
    print('Beginning training...')
    
    num_train_images = 333
    num_val_images = 96

    num_hosts   = len(args.hosts) 
    train_steps = num_train_images // batch_size // num_hosts
    val_steps   = num_val_images   // batch_size // num_hosts

    print('Batch size: {}, Train Steps: {}, Val Steps: {}'.format(batch_size, train_steps, val_steps))


    model.compile(optimizer=SGD(lr=lr, 
                                momentum=0.9, 
                                decay=lr / epochs), 
                  loss='categorical_crossentropy', metrics=['accuracy'])

    if call_back:
        callbacks = []

        callbacks.append(ModelCheckpoint(args.checkpoint_path + '/checkpoint-{epoch}.ckpt',
                                         save_weights_only=False,
                                         monitor='val_accuracy',
                                         verbose=2))
    

        model.fit(train_dataset,
                  epochs=epochs,
                  workers=8,
                  steps_per_epoch=train_steps,
                  validation_data=valid_dataset, 
                  validation_steps=val_steps,
                  shuffle=True,
                  callbacks=callbacks,
                  verbose=2)
    else:
        model.fit(train_dataset,
                  epochs=epochs,
                  workers=8,
                  steps_per_epoch=train_steps,
                  validation_data=valid_dataset, 
                  validation_steps=val_steps,
                  shuffle=True,
                  verbose=2)
    
    return model


## Parse and load the command-line arguments sent to the script
## These will be sent by SageMaker when it launches the training container
def parse_args():
    
    print('Parsing command-line arguments...')
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--initial_epochs', type=int, default=5)
    parser.add_argument('--fine_tuning_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--initial_lr', type=float, default=0.0001)
    parser.add_argument('--fine_tuning_lr', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_fully_connected_layers', type=int, default=1)
    parser.add_argument('--s3_checkpoint_path', type=str, default='')
    
    # Data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--valid', type=str, default=os.environ.get('SM_CHANNEL_VALID'))
    
    # Model output directory
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))  
    
    # Checkpoint info
#     parser.add_argument('--checkpoint_enabled', type=str, default='True')
#     parser.add_argument('--checkpoint_load_previous', type=str, default='True')
    parser.add_argument('--checkpoint_path', type=str, default='/opt/ml/checkpoints')
    
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    
    # data configuration
    parser.add_argument(
        '--data-config',
        type=json.loads,
        default=os.environ.get('SM_INPUT_DATA_CONFIG')
    )
    
    print('Completed parsing command-line arguments.')
    
    return parser.parse_known_args() 


if __name__=='__main__':
    print('Executing the main() function...')
    # Parse command-line arguments    
    args, _ = parse_args()
    print('args: {}'.format(args))   
    
    print('Load TFRecord files ...')
    
    # load the data from TFRecord Files
    train_dataset = read_dataset('train', args)
    valid_dataset = read_dataset('valid', args)
        
    # create a baseline model from MobileNetV2
    model = make_model(dropout=args.dropout, 
                       num_fully_connected_layers = args.num_fully_connected_layers,
                       initial_lr = args.initial_lr,
                       num_classes=NUM_CLASSES)
    
    # If there are no checkpoints found, must be starting from scratch, so load the pretrained model
    checkpoints_avail = os.path.isdir(args.checkpoint_path) and os.listdir(args.checkpoint_path)
    
    if checkpoints_avail:
        model = load_weights_from_latest_checkpoint(model)
        
    else:
        # Train for a few epochs
        model = train_model(model, 
                            train_dataset, 
                            valid_dataset,
                            args.initial_lr,
                            args.batch_size,
                            args.initial_epochs)

        # Now fine tune the last set of layers in the model
        for layer in model.layers[LAST_FROZEN_LAYER:]:
            layer.trainable = True
        
        
    model = train_model(model, 
                        train_dataset, 
                        valid_dataset,
                        args.fine_tuning_lr,
                        args.batch_size,
                        args.fine_tuning_epochs,
                        call_back=True)
   
    
    print('Save the base model ...')
    save_model_artifacts(model, os.environ.get('SM_MODEL_DIR')) #args.model_dir)
    
    print('Complete the model training ...')