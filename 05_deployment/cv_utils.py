import pandas as pd
from numpy import argmax
from IPython.display import Image, display
import os
import boto3
import random

s3 = boto3.client('s3')

def get_n_random_images(bucket_name, prefix,n):
    keys = []
    resp = s3.list_objects_v2(Bucket=f'{bucket_name}', MaxKeys=500,Prefix=prefix)
    for obj in resp['Contents']:
        keys.append(obj['Key'])
    random.shuffle(keys)
    del keys[n:]
    return keys

def download_images_locally(bucket_name,image_keys):
    if not os.path.isdir('./inference-test-data'):
        os.mkdir('./inference-test-data')
    local_paths=[]
    for obj in image_keys:
        fname = obj.split('/')[-1]
        local_dl_path = './inference-test-data/'+fname

        s3.download_file(bucket_name,obj,local_dl_path)
        local_paths.append(local_dl_path)
    return local_paths

def get_classes_as_list(cf, class_filter):
    classes_df = pd.read_csv(cf, sep=' ', header=None)
    criteria = classes_df.iloc[:,0].isin(class_filter)
    classes_df = classes_df[criteria]

    class_name_list = sorted(classes_df.iloc[:,1].unique().tolist())
    return class_name_list


def predict_bird_from_file(fn, predictor,possible_classes,verbose=True, height=224,width=224):
    with open(fn, 'rb') as img:
        f = img.read()
    x = bytearray(f)

    #class_selection = '13, 17, 35, 36, 47, 68, 73, 87'
    
    results = predictor.predict(x)['predictions']
    predicted_class_idx = argmax(results)
    predicted_class = possible_classes[predicted_class_idx]
    confidence = results[0][predicted_class_idx]
    if verbose:
        display(Image(fn, height=height, width=width))
        print('Class: {}, confidence: {:.2f}'.format(predicted_class, confidence))
    del img, x
    return predicted_class_idx, confidence
