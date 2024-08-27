import logging

import argparse
import pathlib
import json
import os
import numpy as np
import tarfile
import glob

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD
# from smexperiments import tracker

input_path =  "/opt/ml/processing/input/test" #"output/test" #
classes_path = "/opt/ml/processing/input/classes"#"output/manifest/test.csv"
model_path = "/opt/ml/processing/model" #"model" # 
output_path = '/opt/ml/processing/output' #"output" # 

HEIGHT = 224
WIDTH  = 224
DEPTH = 3
NUM_CLASSES = 3

def load_classes(file_name):
    
    classes_file = os.path.join(classes_path, file_name)
    
    with open(classes_file) as f:
        classes = json.load(f)
        
    return classes

def get_filenames(input_data_dir):
    return glob.glob('{}/*.tfrecords'.format(input_data_dir))

def _parse_image_function(example):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example, image_feature_description)
    
    image = tf.image.decode_jpeg(features['image'], channels=DEPTH)

    image = tf.image.resize(image, [WIDTH, HEIGHT])
    
    label = tf.cast(features['label'], tf.int32)

    return image, label

def predict_bird(model, img_array):
    
    x = img_array.reshape((1,) + img_array.shape)
    instance = preprocess_input(x)
    
    del x
    
    result = model.predict(instance)
    
    predicted_class_idx = np.argmax(result)
    confidence = result[0][predicted_class_idx]
    
    return predicted_class_idx, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str, default="model.tar.gz")
    parser.add_argument("--classes-file", type=str, default="classes.json")
    args, _ = parser.parse_known_args()

    print("Extracting the model")

    model_file = os.path.join(model_path, args.model_file)
    file = tarfile.open(model_file)
    file.extractall(model_path)

    file.close()

    print("Load model")

    model = keras.models.load_model("{}/1".format(model_path), compile=False)
    model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    print("Starting evaluation.")
    
    class_name = load_classes(args.classes_file)

    filenames = get_filenames(input_path)#args[channel])
    dataset = tf.data.TFRecordDataset(filenames)

    test_dataset = dataset.map(_parse_image_function)

    preds = []
    acts = []
    num_errors = 0

    for image_features in test_dataset:
        act = image_features[1].numpy()
        acts.append(act)

        pred, cnf = predict_bird(model, image_features[0].numpy())

        preds.append(pred)

        print(f'prediction is {pred}, and the actual is {act}============')
        if (pred != act):
            num_errors += 1
            print('ERROR - Pred: {} {:.2f}, Actual: {}'.format(pred, cnf, act))
    
    precision = precision_score(acts, preds, average='micro')
    recall = recall_score(acts, preds, average='micro')
    accuracy = accuracy_score(acts, preds)
    cnf_matrix = confusion_matrix(acts, preds, labels=range(len(class_name)))
    f1 = f1_score(acts, preds, average='micro')

    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Confusion matrix: {}".format(cnf_matrix))
    print("F1 score: {}".format(f1))

    matrix_output = dict()

    for i in range(len(cnf_matrix)):
        matrix_row = dict()
        for j in range(len(cnf_matrix[0])):
            matrix_row[class_name[str(j)]] = int(cnf_matrix[i][j])
        matrix_output[class_name[str(i)]] = matrix_row


    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "f1": {"value": f1, "standard_deviation": "NaN"},
            "confusion_matrix":matrix_output
        },
    }
    
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_path}/evaluation.json"
    
    print(f"Saving the result json file to {evaluation_path} ....")
    
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
        
    print("End of the evaluation process....")
