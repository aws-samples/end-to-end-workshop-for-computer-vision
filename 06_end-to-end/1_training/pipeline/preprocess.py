
import sys
import tensorflow as tf
import argparse
import boto3
import random
import os
import pathlib
import glob
import json

input_path = "/opt/ml/processing/input" #"CUB_200_2011" # 
output_path = '/opt/ml/processing/output' #"output" # 

def serialize_example(image, label):

    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def parse_manifest(manifest_file, labels, classes):

    f = open(manifest_file, 'r')
    
    lines = f.readlines()

    for l in lines:
        img_data = dict()
        img_raw = json.loads(l)

        img_data['image-path'] = img_raw['source-ref'].split('/')[-1]
        img_data['label'] = img_raw['label']
        class_name = img_raw['label-metadata']['class-name']


        if img_data['label'] not in labels:
            labels[img_data['label']] = []
            
        if img_data['label'] not in classes:
            classes[img_data['label']] = class_name

        labels[img_data['label']].append(img_data)
    
    return labels, classes

def split_dataset(labels):
    channels ={
        "train":[],
        "valid":[],
        "test":[]
    }

    for l in labels:
        images = labels[l]
        random.shuffle(images)

        splits = [0.7, 0.9]

        channels["train"] += images[0: int(splits[0] * len(images))]
        channels["valid"] += images[int(splits[0] * len(images)):int(splits[1] * len(images))]
        channels["test"] += images[int(splits[1] * len(images)):]
    
    return channels
    

# def get_filenames(input_data_dir):
#     return glob.glob('{}/*.tfrecords'.format(input_data_dir))

def building_tfrecord(channels, images_dir):
    for c in channels:

        pathlib.Path(f'{output_path}/{c}').mkdir(parents=True, exist_ok=True)


        tfrecord_file = f'{output_path}/{c}/bird-{c}.tfrecords'

        count = 0

        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            for img in channels[c]:

                image_string = open(os.path.join(images_dir, 
                                                 img['image-path']), 'rb').read()

                tf_example = serialize_example(image_string, img['label'])
                writer.write(tf_example)
                count +=1

        print(f"number of images processed for {c} is {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="manifest")
    parser.add_argument("--images", type=str, default="images")
    args, _ = parser.parse_known_args()
    
    manifest_dir = os.path.join(input_path, args.manifest)
    images_dir = os.path.join(input_path, args.images)
    
    print(f"manifest file location: {manifest_dir}...")
    
    if len(os.listdir(manifest_dir)) < 1:
        print(f"No manifest files at this location: {manifest_dir}...")
        sys.exit()
    
    labels = dict()
    classes = dict()
    
    # looping through all the manifest files and parse the values by class
    for m in os.listdir(manifest_dir):
        manifest_file = os.path.join(manifest_dir, m)
        labels, classes = parse_manifest(manifest_file, labels, classes)
    
    # split the dataset by channel
    channels = split_dataset(labels)
    
    building_tfrecord(channels, images_dir)
    
    # save the classes json file
    classes_path = f"{output_path}/classes/classes.json"
    with open(classes_path, "w") as f:
        f.write(json.dumps(classes))
    
    print("Finished running processing job")    


    