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

TF_VERSION = tf.version.VERSION
print('TF version: {}'.format(tf.__version__))
print('Keras version: {}'.format(tensorflow.keras.__version__))

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
LAST_FROZEN_LAYER = 20

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

import numpy as np
import os
import re
import json
import argparse
import glob
import boto3

HEIGHT = 224
WIDTH  = 224

class PurgeCheckpointsCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, args):
        """ Save params in constructor
        """
        self.args = args
        
    def on_epoch_end(self, epoch, logs=None):
        files = sorted([f for f in os.listdir(self.args.checkpoint_path) if f.endswith('.' + 'h5')])
        keep  = 3
        print(f'    End of epoch {epoch + 1}. Removing old checkpoints...')
        for i in range(0, len(files) - keep - 1):
            print(f'    local: {files[i]}')
            os.remove(f'{self.args.checkpoint_path}/{files[i]}')
            
            s3_uri_parts = self.args.s3_checkpoint_path.split('/')
            bucket = s3_uri_parts[2]
            key = '/'.join(s3_uri_parts[3:]) + '/' + files[i]

            print(f'    s3 bucket: {bucket}, key: {key}')
            s3_client = boto3.client('s3')
            s3_client.delete_object(Bucket=bucket, Key=key)
        print(f'      Done\n')

def load_checkpoint_model(checkpoint_path):
    files = sorted([f for f in os.listdir(checkpoint_path) if f.endswith('.' + 'h5')])
    epoch_numbers = [re.search('(?<=\.)(.*[0-9])(?=\.)',f).group() for f in files]
      
    max_epoch_number = max(epoch_numbers)
    max_epoch_index = epoch_numbers.index(max_epoch_number)
    max_epoch_filename = files[max_epoch_index]
    
    print('\nList of available checkpoints:')
    print('------------------------------------')
    [print(f) for f in files]
    print('------------------------------------')
    print(f'Checkpoint file for latest epoch: {max_epoch_filename}')
    print(f'Resuming training from epoch: {max_epoch_number}')
    print('------------------------------------')
    
    resume_model = load_model(f'{checkpoint_path}/{max_epoch_filename}')
    return resume_model, int(max_epoch_number)

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    # Freeze all base layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x) 
        if (dropout != 0.0):
            x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax', name='output')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

def create_data_generators(args):
    train_datagen =  ImageDataGenerator(
              preprocessing_function=preprocess_input,
              rotation_range=70,
              brightness_range=(0.6, 1.0),
              width_shift_range=0.3,
              height_shift_range=0.3,
              shear_range=0.3,
              zoom_range=0.3,
              horizontal_flip=True,
              vertical_flip=False)
    val_datagen  = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory('/opt/ml/input/data/train',
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=args.batch_size)
    test_gen = train_datagen.flow_from_directory('/opt/ml/input/data/test',
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=args.batch_size)
    val_gen = train_datagen.flow_from_directory('/opt/ml/input/data/validation',
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=args.batch_size)
    return train_gen, test_gen, val_gen

def save_model_artifacts(model, model_dir):
    print(f'Saving model to {model_dir}...')
    # Note that this method of saving does produce a warning about not containing the train and evaluate graphs.
    # The resulting saved model works fine for inference. It will simply not support incremental training. If that
    # is needed, one can use model checkpoints and save those.
    print('Model directory files BEFORE save: {}'.format(glob.glob(f'{model_dir}/*/*')))
    if tf.version.VERSION[0] == '2':
        model.save(f'{model_dir}/1', save_format='tf')
    else:
        tf.contrib.saved_model.save_keras_model(model, f'{model_dir}/1')
    print('Model directory files AFTER save: {}'.format(glob.glob(f'{model_dir}/*/*')))
    print('...DONE saving model!')

    # Need to copy these files to the code directory, else the SageMaker endpoint will not use them.
    print('Copying inference source files...')

    if not os.path.exists(f'{model_dir}/code'):
        os.system(f'mkdir {model_dir}/code')
    os.system(f'cp inference.py {model_dir}/code')
    os.system(f'cp requirements.txt {model_dir}/code')
    print('Files after copying custom inference handler files: {}'.format(glob.glob(f'{model_dir}/code/*')))

def main(args):
    sm_training_env_json = json.loads(os.environ.get('SM_TRAINING_ENV'))
    is_master = sm_training_env_json['is_master']
    print('is_master {}'.format(is_master))
    
    # Create data generators for feeding training and evaluation based on data provided to us
    # by the SageMaker TensorFlow container
    train_gen, test_gen, val_gen = create_data_generators(args)

    base_model = MobileNetV2(weights='imagenet', 
                          include_top=False, 
                          input_shape=(HEIGHT, WIDTH, 3))

    # Here we extend the base model with additional fully connected layers, dropout for avoiding
    # overfitting to the training dataset, and a classification layer
    fully_connected_layers = []
    for i in range(args.num_fully_connected_layers):
        fully_connected_layers.append(1024)

    num_classes = len(glob.glob('/opt/ml/input/data/train/*'))
    model = build_finetune_model(base_model, 
                                  dropout=args.dropout, 
                                  fc_layers=fully_connected_layers, 
                                  num_classes=num_classes)

    opt = RMSprop(lr=args.initial_lr)
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

    print('\nBeginning training...')
    
    NUM_EPOCHS  = args.fine_tuning_epochs
    
    num_train_images = len(train_gen.filepaths)
    num_val_images   = len(val_gen.filepaths)

    num_hosts   = len(args.hosts) 
    train_steps = num_train_images // args.batch_size // num_hosts
    val_steps   = num_val_images   // args.batch_size // num_hosts

    print('Batch size: {}, Train Steps: {}, Val Steps: {}'.format(args.batch_size, train_steps, val_steps))

    # If there are no checkpoints found, must be starting from scratch, so load the pretrained model
    checkpoints_avail = os.path.isdir(args.checkpoint_path) and os.listdir(args.checkpoint_path)
    if not checkpoints_avail:
        if not os.path.isdir(args.checkpoint_path):
            os.mkdir(args.checkpoint_path)
        
        # Train for a few epochs
        model.fit_generator(train_gen, epochs=args.initial_epochs, workers=8, 
                               steps_per_epoch=train_steps, 
                               validation_data=val_gen, validation_steps=val_steps,
                               shuffle=True) 

        # Now fine tune the last set of layers in the model
        for layer in model.layers[LAST_FROZEN_LAYER:]:
            layer.trainable = True

        initial_epoch_number = 0
        
    # Otherwise, start from the latest checkpoint
    else:    
        model, initial_epoch_number = load_checkpoint_model(args.checkpoint_path)

    fine_tuning_lr = args.fine_tuning_lr
    model.compile(optimizer=SGD(lr=fine_tuning_lr, momentum=0.9, decay=fine_tuning_lr / NUM_EPOCHS), 
                  loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = []
    if not (args.s3_checkpoint_path == ''):
        checkpoint_names = 'img-classifier.{epoch:03d}.h5'
        checkpoint_callback = ModelCheckpoint(filepath=f'{args.checkpoint_path}/{checkpoint_names}',
                                              save_weights_only=False,
                                              monitor='val_accuracy')
        callbacks.append(checkpoint_callback)
        callbacks.append(PurgeCheckpointsCallback(args))

    history = model.fit_generator(train_gen, epochs=NUM_EPOCHS, workers=8, 
                           steps_per_epoch=train_steps, 
                           initial_epoch=initial_epoch_number,
                           validation_data=val_gen, validation_steps=val_steps,
                           shuffle=True, callbacks=callbacks)
    print('Model has been fit.')

    # Save the model if we are executing on the master host
    if is_master:
        print('Saving model, since we are master host')
        save_model_artifacts(model, os.environ.get('SM_MODEL_DIR'))
    else:
        print('NOT saving model, will leave that up to master host')

    checkpoint_files = [f for f in os.listdir(args.checkpoint_path) if f.endswith('.' + 'h5')]  
    print(f'\nCheckpoints: {sorted(checkpoint_files)}')

    print('\nExiting training script.\n')
    
if __name__=='__main__':
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
    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--checkpoint_path', type=str, default='/opt/ml/checkpoints')

    args, _ = parser.parse_known_args()
    print('args: {}'.format(args))
    
    main(args)