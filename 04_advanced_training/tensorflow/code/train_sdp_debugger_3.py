"""
Copyright 2022 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""
import argparse
import numpy as np
import os
import json
import logging
import random
import glob
import time
import tensorflow as tf

import smdebug
import smdebug.tensorflow as smd
import smdistributed.dataparallel
import smdistributed.dataparallel.tensorflow as sdp
import tensorflow.config.experimental as exp

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, 
                                     add, GlobalAveragePooling2D,BatchNormalization,
                                     Dense, Dropout, Activation, SeparableConv2D)

from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, CategoricalAccuracy

# Declare constants
HEIGHT, WIDTH, DEPTH = 224, 224, 3

# Create the logger
logger = logging.getLogger(__name__)
logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.INFO)))


if os.environ['SM_CURRENT_HOST'] == "algo-1":
    # write TF_CONFIG for node-1
    os.environ['TF_CONFIG'] = json.dumps({
        "cluster": {
            "master":["algo-1:2222"],
            "worker": ["algo-1:2222", "algo-2:2222"]
        },
        "environment": "cloud",
        "task": {"index": 0, "type": "worker"}
 })
elif os.environ['SM_CURRENT_HOST'] == "algo-2":
    # write TF_CONFIG for node-2
    os.environ['TF_CONFIG'] = json.dumps({
        "cluster": {
            "master":["algo-1:2222"],
            "worker": ["algo-1:2222", "algo-2:2222"]
        },
        "environment": "cloud",
        "task": {"index": 1, "type": "worker"}
 })

## Parse and load the command-line arguments sent to the script
## These will be sent by SageMaker when it launches the training container
def parse_args():
    logger.info('Parsing command-line arguments...')
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--decay', type=float, default=1e-6)
    parser.add_argument('--num_fully_connected_layers', type=int, default=1)

    # Data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--valid', type=str, default=os.environ.get('SM_CHANNEL_VALID'))
    # Model output directory
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    # Checkpoint info
    parser.add_argument('--checkpoint_enabled', type=str, default='True')
#     parser.add_argument('--checkpoint_load_previous', type=str, default='True')
    parser.add_argument('--checkpoint_path', type=str, default='/opt/ml/checkpoints/')
    
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
    
    logger.info('Completed parsing command-line arguments.')
    return parser.parse_known_args()

## Initialize the SMDataParallel environment
def init_sdp():
    logger.info('Initializing the SMDataParallel environment...')
    tf.random.set_seed(42)
    sdp.init()
    logger.debug('Getting GPU list...')
    gpus = exp.list_physical_devices('GPU')
    logger.debug('Number of GPUs = {}'.format(len(gpus)))
    logger.debug('Completed getting GPU list.')
    logger.debug('Enabling memory growth on all GPUs...')
    for gpu in gpus:
        exp.set_memory_growth(gpu, True)
    logger.debug('Completed enabling memory growth on all GPUs.')
    logger.debug('Pinning GPUs to a single SMDataParallel process...')
    if gpus:
        exp.set_visible_devices(gpus[sdp.local_rank()], 'GPU')
    logger.debug('Completed pinning GPUs to a single SMDataParallel process.')
    logger.info('Completed initializing the SMDataParallel environment.')
    
## Initialize the SMDebugger for the Tensorflow framework
def init_smd():
    logger.info('GPU # {} :: Initializing the SMDebugger for the Tensorflow framework...'.format(sdp.rank()))
    # Use KerasHook - the configuration file will be copied to /opt/ml/input/config/debughookconfig.json
    # automatically by SageMaker when it launches the training container
    hook = smd.KerasHook.create_from_json_file()
    logger.info('GPU # {} :: Debugger hook collections :: {}'.format(sdp.rank(), hook.get_collections()))
    logger.info('GPU # {} :: Completed initializing the SMDebugger for the Tensorflow framework.'.format(sdp.rank()))
    return hook

def create_data_generators(args):
    train_datagen =  ImageDataGenerator(
              preprocessing_function=preprocess_input,
#               rescale=1./255,
              rotation_range=70,
              brightness_range=(0.6, 1.0),
              width_shift_range=0.3,
              height_shift_range=0.3,
              shear_range=0.3,
              zoom_range=0.3,
              horizontal_flip=True,
              vertical_flip=False)
    
    val_datagen  = ImageDataGenerator(preprocessing_function=preprocess_input)#rescale=1./255)     

    train_gen = train_datagen.flow_from_directory(args.train,
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=args.batch_size)
    val_gen = train_datagen.flow_from_directory(args.valid,
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=args.batch_size)
    return train_gen, val_gen

## Transfer learning an existing network
def make_model(dropout, num_fully_connected_layers, num_classes):
    
#     with open(DEFAULT_CONFIG_FILE) as f:
    sm_tf_config = json.loads(os.environ['TF_CONFIG'])
    
    master = sm_tf_config['cluster']['master'][0]
    session = tf.compat.v1.Session('grpc://' + master)
    
    tf.compat.v1.keras.backend.set_session(session)

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
    
    return model

def create_model2(num_classes, dropout):
    inputs = Input(shape=(HEIGHT, WIDTH, DEPTH))
    # Entry block
    x = Rescaling(1.0 / 255)(inputs)
    x = Conv2D(32, 3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual
    
    for size in [128, 256, 512, 728]:
        x = Activation("relu")(x)
        x = SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = SeparableConv2D(1024, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(dropout)(x)
    
    outputs = Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs, outputs)

## Construct the network
def create_model(num_classes, dropout):
    logger.debug('GPU # {} :: Creating the model...'.format(sdp.rank()))
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
               input_shape=(HEIGHT, WIDTH, DEPTH)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(1024, activation='relu'),

        Dense(512, activation='relu'),
        
        Dropout(dropout),

        Dense(num_classes, activation='softmax')
    ])
    # Print the model summary
    logger.debug(model.summary())
    logger.info('GPU # {} :: Completed creating the model.'.format(sdp.rank()))
    return model

# Compile the model by setting the optimizer, loss function and metrics
def compile_model(model, lr, epochs):
    logger.info('GPU # {} :: Compiling the model...'.format(sdp.rank()))
    # SMDataParallel: Scale learning rate
    lr = lr * sdp.size()

    # Instantiate the optimizer
    optimizer = SGD(lr=lr, momentum=0.9)
    
    optimizer = hook.wrap_optimizer(optimizer)
    
    # Instantiate the loss function
    loss_fn = CategoricalCrossentropy(from_logits=True)
    # Prepare the metrics
    train_acc_metric = CategoricalAccuracy()
    val_acc_metric = CategoricalAccuracy()
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=[train_acc_metric])
    
    logger.info('GPU # {} :: Completed compiling the model.'.format(sdp.rank()))
    return model, optimizer, loss_fn, train_acc_metric, val_acc_metric

## Define the training step
@tf.function
def training_step(model, x_batch_train, y_batch_train, optimizer, loss_fn, train_acc_metric, is_first_batch):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation
    # SMDebugger: Wrap the gradient tape to retrieve gradient tensors
    #with hook.wrap_tape(tf.GradientTape(persistent=True)) as tape:
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer
        logits = model(x_batch_train, training=True)
        # Compute the loss value
        loss_value = loss_fn(y_batch_train, logits)
    # SMDataParallel: Wrap tf.GradientTape with SMDataParallel's DistributedGradientTape
    tape = sdp.DistributedGradientTape(tape)
    # Retrieve the gradients of the trainable variables with respect to the loss
    grads = tape.gradient(loss_value, model.trainable_weights)
    # SMDebugger: Save the gradients
#     hook.save_tensor('gradients', grads, 'gradients')
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # Perform speicific SMDataParallel on the first batch
    if is_first_batch:
        # SMDataParallel: Broadcast model and optimizer variables
        sdp.broadcast_variables(model.variables, root_rank=0)
        sdp.broadcast_variables(optimizer.variables(), root_rank=0)
    # Update training metric
    train_acc_metric.update_state(y_batch_train, logits)
    # SMDataParallel: all_reduce call - average the loss across workers
    loss_value = sdp.oob_allreduce(loss_value)
    return loss_value


## Define the validation step
@tf.function
def validation_step(model, x_batch_val, y_batch_val, val_acc_metric):
    val_logits = model(x_batch_val, training=False)
    val_acc_metric.update_state(y_batch_val, val_logits)
    
## Perform validation
def perform_validation(model, val_dataset, num_valid_images, batch_size, val_acc_metric):
    logger.debug('GPU # {} :: Performing validation...'.format(sdp.rank()))
    
    batches = 0
    for x_batch_val, y_batch_val in val_dataset:
        validation_step(model, x_batch_val, y_batch_val, val_acc_metric)
        
        logger.info(f'GPU # {sdp.rank()}: val_accuracy: {val_acc_metric.result().numpy()}===================================')
        batches += 1
        # we need to break the loop by hand because
        # the generator loops indefinitely
        if batches >= num_valid_images / batch_size:
            break
    logger.debug('GPU # {} :: Completed performing validation.'.format(sdp.rank()))
    return val_acc_metric.result()
    
## Train the model
def train_model(model, train_dataset, num_train_images, valid_dataset, num_valid_images, lr, batch_size, epochs, checkpoints):
    
    logger.info('Beginning training...')
    
    logger.info(f'Number of Epoch: {epochs}, Learning Rate: {lr} ===============')
    
    history = []
    
    # Compile the model
    model, optimizer, loss_fn, train_acc_metric, val_acc_metric = compile_model(model, lr, epochs)
    
    hook.set_mode(smd.modes.TRAIN)
    training_start_time = time.time()
    
    # SMDataParallel & SMDebugger: Save basic details only from leader node
    if sdp.rank() == 0:
        hook.save_scalar('batch_size', batch_size, sm_metric=True)
        hook.save_scalar('number_of_epochs', epochs, sm_metric=True)
        
    for epoch in range(epochs):
        logger.info(f'GPU # {sdp.rank()} :: Starting epoch {int(epoch) + 1}...')
        
        # SMDataParallel & SMDebugger: Save the epoch number only from leader node
        if sdp.rank() == 0:
            hook.save_scalar('epoch_number', int(epoch) + 1, sm_metric=True)
            
        epoch_start_time = time.time()
        
        batches = 0

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            logger.info(f'GPU # {sdp.rank()} :: Running training step {int(step) + 1}...')
            # SMDataParallel & SMDebugger: Save the step number only from leader node
            if sdp.rank() == 0:
                hook.save_scalar('step_number', int(step) + 1, sm_metric=True)
            
            loss_value = training_step(model, 
                                       x_batch_train, 
                                       y_batch_train, 
                                       optimizer, 
                                       loss_fn, 
                                       train_acc_metric, 
                                       step==0)
            
            logger.info(f'GPU # {sdp.rank()} :: loss: {loss_value} ...')
            
            batches += 1
            
            # we need to break the loop by hand because
            # the generator loops indefinitely
            if batches >= num_train_images / batch_size:
                break

        # SMDataParallel: Perform validation only from leader node
        if sdp.rank() == 0:      
            # Perform validation and save metrics at the end of each epoch
            history.append([int(epoch) + 1, 
                            train_acc_metric.result(),
                            perform_validation(model, valid_dataset, num_valid_images, batch_size, val_acc_metric)])
        
        logger.info(f"GPU # {sdp.rank()} :: model accuracy: {train_acc_metric.result().numpy()} and val_accuracy: {val_acc_metric.result().numpy()} ...")
        # Reset metrics
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()
        
        # SMDataParallel: Perform checkpointing only from leader node
        if sdp.rank() == 0:
            # Save the model as a checkpoint
            if checkpoints.lower() == 'true':
                ModelCheckpoint(f"{args.checkpoint_path}/checkpoint-{epoch}.ckpt",
                                save_weights_only=False,
                                monitor='val_accuracy',
                                erbose=2)
            
        epoch_end_time = time.time()
        # SMDataParallel: Print epoch time only from leader node
        if sdp.rank() == 0:
            logger.info("Epoch duration = %.2f second(s)" % (epoch_end_time - epoch_start_time))

    training_end_time = time.time()
    # SMDataParallel: Print training time and result only from leader node
    if sdp.rank() == 0:    
        logger.info('Training duration = %.2f second(s)' % (training_end_time - training_start_time))
        print_training_result(history)
        
    logger.info('Completed training the model.')
    
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
    logger.info(''.join(output_table_string_list))
    
## Save the model
def save_model(model, model_dir):
    logger.info('GPU # {} :: Saving the model...')#.format(sdp.rank()))
    tf.saved_model.save(model, model_dir)
    logger.info('GPU # {} :: Completed saving the model.')#.format(sdp.rank()))
    
if __name__=='__main__':
    logger.info('Executing the main() function...')
    # Parse command-line arguments
    args, _ = parse_args()

    # Initialize the SMDataParallel environment
    init_sdp()
    
    # Log version info
    logger.info('GPU # {} :: TensorFlow version : {}'.format(sdp.rank(), tf.__version__))
    logger.info('GPU # {} :: SMDebug version : {}'.format(sdp.rank(), smdebug.__version__))
    logger.info('GPU # {} :: SMDistributedDataParallel version : {}'.format(sdp.rank(), smdistributed.dataparallel.__version__))
    
    # Initialize the SMDebugger for the Tensorflow framework
    hook = init_smd()
    
    # Create data generators for feeding training and evaluation based on data provided to us
    # by the SageMaker TensorFlow container
    train_gen, val_gen = create_data_generators(args)

    num_classes = len(glob.glob(f'{args.train}/*'))
    
    chk_valid_classes = len(glob.glob(f'{args.valid}/*'))
    
    logger.info(f'Number of training classes: {num_classes} =================')
    logger.info(f'Check Number of validation classes: {chk_valid_classes} =================')
    
    num_train_images = len(train_gen.filepaths)
    num_valid_images   = len(val_gen.filepaths)
        
    logger.info(f'Number of training images: {num_train_images} =================')
    logger.info(f'Number of validation images: {num_valid_images} =================')
    
#     model = make_model(args.dropout, 
#                        args.num_fully_connected_layers, 
#                        num_classes)

    model = create_model2(num_classes, args.dropout)
    
    model = train_model(model,
                        train_gen,
                        num_train_images,
                        val_gen,
                        num_valid_images,
                        args.lr,
                        args.batch_size,
                        args.epochs,
                        args.checkpoint_enabled)
    
    # SMDataParallel: Evaluate and save model only from leader node
    if sdp.rank() == 0:
        # Save the generated model
        save_model(model, args.model_dir)
    
    hook.close()
    
    logger.info('Completed executing the main() function.')

    
    
    
    