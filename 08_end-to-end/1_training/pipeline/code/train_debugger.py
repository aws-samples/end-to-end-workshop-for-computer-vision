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

#sagemaker debugger
import smdebug
import smdebug.tensorflow as smd

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
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
import time

HEIGHT = 224
WIDTH  = 224
DEPTH = 3
NUM_CLASSES = 8
LAST_FROZEN_LAYER = 20

TF_VERSION = tf.version.VERSION
print('TF version: {}'.format(tf.__version__))
print('Keras version: {}'.format(tensorflow.keras.__version__))

## Initialize the SMDebugger for the Tensorflow framework
def init_smd():
    print('Initializing the SMDebugger for the Tensorflow framework...')
    # Use KerasHook - the configuration file will be copied to /opt/ml/input/config/debughookconfig.json
    # automatically by SageMaker when it launches the training container
    hook = smd.KerasHook.create_from_json_file()
    return hook

def get_filenames(input_data_dir):
    return glob.glob('{}/*.tfrecords'.format(input_data_dir))

def _parse_image_function(example, augmentation = False):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.image.resize(image, [WIDTH, HEIGHT])
    
    image = preprocess_input(image)
    
    if augmentation:
        prob = random.uniform(0, 1)
        if prob < .25:
    #         image=tf.image.rot90(image,k=2)
            image = tf.image.random_flip_left_right(image)
        elif prob < .5:
    #         image=tf.image.rot90(image,k=3)
            image = tf.image.random_flip_up_down(image)
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
    
    repeat = 1
    if channel == 'train':
        dataset = dataset.map(lambda x: _parse_image_function(x, augmentation = True),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        repeat = 10
    else:
        dataset = dataset.map(lambda x: _parse_image_function(x, augmentation = False),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    dataset = dataset.shuffle(500)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    
    dataset = dataset.repeat(repeat)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
    
def save_model_artifacts(model, model_dir):
    print(f'Saving model to {model_dir}...')
    # Note that this method of saving does produce a warning about not containing the train and evaluate graphs.
    # The resulting saved model works fine for inference. It will simply not support incremental training. If that
    # if needed, one can use model checkpoints and save those.
    
    model.save(f'{model_dir}/1', save_format='tf')

    print('...DONE saving model!')

## Define the training step
@tf.function
def training_step(hook, model, x_batch_train, y_batch_train, optimizer, loss_fn, train_acc_metric):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation
    # SMDebugger: Wrap the gradient tape to retrieve gradient tensors
    #with hook.wrap_tape(tf.GradientTape(persistent=True)) as tape:
    with hook.wrap_tape(tf.GradientTape()) as tape:
        # Run the forward pass of the layer
        logits = model(x_batch_train, training=True)
        # Compute the loss value
        loss_value = loss_fn(y_batch_train, logits)
    # Retrieve the gradients of the trainable variables with respect to the loss
    grads = tape.gradient(loss_value, model.trainable_weights)
    # SMDebugger: Save the gradients
    #hook.save_tensor('gradients', grads, 'gradients')
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # Update training metric
    train_acc_metric.update_state(y_batch_train, logits)
    
    return loss_value

## Define the validation step
@tf.function
def validation_step(model, x_batch_val, y_batch_val, val_acc_metric):
    val_logits = model(x_batch_val, training=False)
  
    val_acc_metric.update_state(y_batch_val, val_logits)
    
    
## Perform validation
def perform_validation(model, val_dataset, val_acc_metric):
    print('Performing validation...')
    for x_batch_val, y_batch_val in val_dataset:
        validation_step(model, x_batch_val, y_batch_val, val_acc_metric)
    print('Completed performing validation.')
    return val_acc_metric.result()

def make_model(dropout, fc_layers, num_classes):
    
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', 
                      include_top=False, 
                      input_shape=(HEIGHT, WIDTH, DEPTH))
    
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    x = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=[base_model.input], outputs=[x])
    
    return model

## Train the model
def train_model(model, train_dataset, valid_dataset, lr, batch_size, epochs, checkpoints, hook):

    print('Beginning training...')
    history = []

    print(f'Batch size: {batch_size}, Number of Epoch: {epochs}, Learning Rate: {lr}')
    
    # SMDebugger: Save basic details
    hook.save_scalar('batch_size', batch_size, sm_metric=True)
    hook.save_scalar('number_of_epochs', epochs, sm_metric=True)
    
    optimizer = SGD(lr=lr, momentum=0.9, decay=lr / epochs)
    
    optimizer = hook.wrap_optimizer(optimizer)
    
    # Instantiate the loss function
    loss_fn = CategoricalCrossentropy(from_logits=True)
    # Prepare the metrics
    train_acc_metric = CategoricalAccuracy()
    val_acc_metric = CategoricalAccuracy()
    
    model.compile(optimizer=optimizer,
                  loss=loss_fn, 
                  metrics=[train_acc_metric])
 
    hook.set_mode(smd.modes.TRAIN)
    training_start_time = time.time()
    
    for epoch in range(epochs):
        print(f'Starting epoch {int(epoch) + 1}...')
        
        # SMDebugger: Save the epoch number
        hook.save_scalar('epoch_number', int(epoch) + 1, sm_metric=True)
        epoch_start_time = time.time()
        
        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            print(f'Running training step {int(step) + 1}...')
            # SMDebugger: Save the step number
            hook.save_scalar('step_number', int(step) + 1, sm_metric=True)
            
            loss_value = training_step(hook, model, x_batch_train, y_batch_train, optimizer, loss_fn, train_acc_metric)
            print(f'loss: {loss_value} ...')     
        
        # Perform validation and save metrics at the end of each epoch
        history.append([int(epoch) + 1, 
                        train_acc_metric.result(),
                        perform_validation(model, valid_dataset, val_acc_metric)])
        
        print(f"model accuracy: {train_acc_metric.result().numpy()} and val_accuracy: {val_acc_metric.result().numpy()} ...")
        # Reset metrics
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()
        
        # Save the model as a checkpoint
        if checkpoints.lower() == 'true':
            ModelCheckpoint(f"{args.checkpoint_path}/checkpoint-{epoch}.ckpt",
                            save_weights_only=False,
                            monitor='val_accuracy',
                            erbose=2)
            
        epoch_end_time = time.time()
        print("Epoch duration = %.2f second(s)" % (epoch_end_time - epoch_start_time))

    training_end_time = time.time()
    print('Training duration = %.2f second(s)' % (training_end_time - training_start_time))
    
    print_training_result(history)
    print('Completed training the model.')
    
    return model

## Print training result
def print_training_result(history):
    output_table_string_list = []
    output_table_string_list.append('\n')
    output_table_string_list.append("{:<10} {:<25} {:<25}".format('Epoch', 'Accuracy', 'Validation Accuracy'))
    output_table_string_list.append('\n')
    size = len(history)
    for index in range(size):
        record = history[index]
        output_table_string_list.append("{:<10} {:<25} {:<25}".format(record[0], record[1], record[2]))
        output_table_string_list.append('\n')
    output_table_string_list.append('\n')
    print(''.join(output_table_string_list))

## Parse and load the command-line arguments sent to the script
## These will be sent by SageMaker when it launches the training container
def parse_args():
    
    print('Parsing command-line arguments...')
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_fully_connected_layers', type=int, default=1)
    parser.add_argument('--s3_checkpoint_path', type=str, default='')
    
    # Data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--valid', type=str, default=os.environ.get('SM_CHANNEL_VALID'))
    
    # Checkpoint info
    parser.add_argument('--checkpoint_enabled', type=str, default='True')
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
        
    # sagemaker framework parameters
    parser.add_argument(
        '--fw-params',
        type=json.loads,
        default=os.environ.get('SM_FRAMEWORK_PARAMS')
    )
    
    print('Completed parsing command-line arguments.')
    
    return parser.parse_known_args() 


if __name__=='__main__':
    print('Executing the main() function...')
    # Parse command-line arguments    
    args, _ = parse_args()
    
    # Initialize the SMDebugger for the Tensorflow framework
    hook = init_smd()
    
    print('Load TFRecord files ...')
    
    # load the data from TFRecord Files
    train_dataset = read_dataset('train', args)
    valid_dataset = read_dataset('valid', args)

    fully_connected_layers = []
    for i in range(args.num_fully_connected_layers):
        fully_connected_layers.append(1024)
        
    # create a baseline model from MobileNetV2
    model = make_model(dropout=args.dropout, 
                       fc_layers = fully_connected_layers,
                       num_classes=NUM_CLASSES)   
        
    model = train_model(model,
                        train_dataset, 
                        valid_dataset,
                        args.lr,
                        args.batch_size,
                        args.epochs,
                        args.checkpoint_enabled,
                        hook)
   
    print('Save the base model ...')
    save_model_artifacts(model, os.environ.get('SM_MODEL_DIR'))

    hook.close()
    print('Complete the model training ...')