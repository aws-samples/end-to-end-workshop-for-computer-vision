import logging

import pandas as pd
import argparse
import pathlib
import json
import os
import numpy as np
import tarfile
import uuid

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score
)

from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
# from smexperiments import tracker

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

input_path =  "/opt/ml/processing/input/test" #"output/test" #
manifest_path = "/opt/ml/processing/input/manifest/test.csv"#"output/manifest/test.csv"
model_path = "/opt/ml/processing/model" #"model" # 
output_path = '/opt/ml/processing/output' #"output" # 

HEIGHT=224; WIDTH=224

def predict_bird_from_file_new(fn, model):
    
    img = Image.open(fn).convert('RGB')
    
    img = img.resize((WIDTH, HEIGHT))
    img_array = image.img_to_array(img) #, data_format = "channels_first")

    x = img_array.reshape((1,) + img_array.shape)
    instance = preprocess_input(x)

    del x, img
    
    result = model.predict(instance)

    predicted_class_idx = np.argmax(result)
    confidence = result[0][predicted_class_idx]

    return predicted_class_idx, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str, default="model.tar.gz")
    args, _ = parser.parse_known_args()

    print("Extracting the model")

    model_file = os.path.join(model_path, args.model_file)
    file = tarfile.open(model_file)
    file.extractall(model_path)

    file.close()

    print("Load model")

    model = keras.models.load_model("{}/1".format(model_path))

    print("Starting evaluation.")
    
    # load test data.  this should be an argument
    df = pd.read_csv(manifest_path)
    
    num_images = df.shape[0]
    
    class_name_list = sorted(df['class_id'].unique().tolist())
    
    class_name = pd.Series(df['class_name'].values,index=df['class_id']).to_dict()
    
    print('Testing {} images'.format(df.shape[0]))
    num_errors = 0
    preds = []
    acts  = []
    for i in range(df.shape[0]):
        fname = df.iloc[i]['image_file_name']
        act   = int(df.iloc[i]['class_id']) - 1
        acts.append(act)
        
        pred, conf = predict_bird_from_file_new(input_path + '/' + fname, model)

        preds.append(pred)
        
        print(f'max range is {len(class_name_list)-1}, prediction is {pred}, and the actual is {act}============')
        if (pred != act):
            num_errors += 1
            logger.debug('ERROR on image index {} -- Pred: {} {:.2f}, Actual: {}'.format(i, 
                                                                   class_name_list[pred], conf, 
                                                                   class_name_list[act]))
    precision = precision_score(acts, preds, average='micro')
    recall = recall_score(acts, preds, average='micro')
    accuracy = accuracy_score(acts, preds)
    cnf_matrix = confusion_matrix(acts, preds, labels=range(len(class_name_list)))
    f1 = f1_score(acts, preds, average='micro')
    
    print("Accuracy: {}".format(accuracy))
    logger.debug("Precision: {}".format(precision))
    logger.debug("Recall: {}".format(recall))
    logger.debug("Confusion matrix: {}".format(cnf_matrix))
    logger.debug("F1 score: {}".format(f1))
    
    print(cnf_matrix)
    
    matrix_output = dict()
    
    for i in range(len(cnf_matrix)):
        matrix_row = dict()
        for j in range(len(cnf_matrix[0])):
            matrix_row[class_name[class_name_list[j]]] = int(cnf_matrix[i][j])
        matrix_output[class_name[class_name_list[i]]] = matrix_row

    
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "f1": {"value": f1, "standard_deviation": "NaN"},
            "confusion_matrix":matrix_output
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
