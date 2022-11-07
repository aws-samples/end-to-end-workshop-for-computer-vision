
import logging

import pandas as pd
import argparse
import boto3
import json
import os
import shutil

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

input_path = "/opt/ml/processing/input" #"CUB_200_2011" # 
output_path = '/opt/ml/processing/output' #"output" # 
IMAGES_DIR   = os.path.join(input_path, 'images')
SPLIT_RATIOS = (0.6, 0.2, 0.2)


# this function is used to split a dataframe into 3 seperate dataframes
# one of each: train, validate, test

def split_to_train_val_test(df, label_column, splits=(0.7, 0.2, 0.1), verbose=False):
    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    labels = df[label_column].unique()
    for lbl in labels:
        lbl_df = df[df[label_column] == lbl]

        lbl_train_df        = lbl_df.sample(frac=splits[0])
        lbl_val_and_test_df = lbl_df.drop(lbl_train_df.index)
        lbl_test_df         = lbl_val_and_test_df.sample(frac=splits[2]/(splits[1] + splits[2]))
        lbl_val_df          = lbl_val_and_test_df.drop(lbl_test_df.index)

        if verbose:
            print('\n{}:\n---------\ntotal:{}\ntrain_df:{}\nval_df:{}\ntest_df:{}'.format(lbl,
                                                                        len(lbl_df), 
                                                                        len(lbl_train_df), 
                                                                        len(lbl_val_df), 
                                                                        len(lbl_test_df)))
        train_df = train_df.append(lbl_train_df)
        val_df   = val_df.append(lbl_val_df)
        test_df  = test_df.append(lbl_test_df)

    # shuffle them on the way out using .sample(frac=1)
    return train_df.sample(frac=1), val_df.sample(frac=1), test_df.sample(frac=1)

# This function grabs the manifest files and build a dataframe, then call the split_to_train_val_test
# function above and return the 3 dataframes
def get_train_val_dataframes(BASE_DIR, classes, images_to_exclude, split_ratios):
    CLASSES_FILE = os.path.join(BASE_DIR, 'classes.txt')
    IMAGE_FILE   = os.path.join(BASE_DIR, 'images.txt')
    LABEL_FILE   = os.path.join(BASE_DIR, 'image_class_labels.txt')

    images_df = pd.read_csv(IMAGE_FILE, sep=' ',
                            names=['image_pretty_name', 'image_file_name'],
                            header=None)
    image_class_labels_df = pd.read_csv(LABEL_FILE, sep=' ',
                                names=['image_pretty_name', 'orig_class_id'], header=None)

    # Merge the metadata into a single flat dataframe for easier processing
    full_df = pd.DataFrame(images_df)
    full_df = full_df[~full_df.image_file_name.isin(images_to_exclude)]

    full_df.reset_index(inplace=True, drop=True)
    full_df = pd.merge(full_df, image_class_labels_df, on='image_pretty_name')

    # grab a small subset of species for testing
    criteria = full_df['orig_class_id'].isin(classes)
    full_df = full_df[criteria]
    print('Using {} images from {} classes'.format(full_df.shape[0], len(classes)))

    unique_classes = full_df['orig_class_id'].drop_duplicates()
    sorted_unique_classes = sorted(unique_classes)
    id_to_one_based = {}
    i = 1
    for c in sorted_unique_classes:
        id_to_one_based[c] = str(i)
        i += 1

    full_df['class_id'] = full_df['orig_class_id'].map(id_to_one_based)
    full_df.reset_index(inplace=True, drop=True)

    def get_class_name(fn):
        return fn.split('/')[0]
    full_df['class_name'] = full_df['image_file_name'].apply(get_class_name)
    full_df = full_df.drop(['image_pretty_name'], axis=1)

    train_df = []
    test_df  = []
    val_df   = []

    # split into training and validation sets
    train_df, val_df, test_df = split_to_train_val_test(full_df, 'class_id', split_ratios)

    print('num images total: ' + str(images_df.shape[0]))
    print('\nnum train: ' + str(train_df.shape[0]))
    print('num val: ' + str(val_df.shape[0]))
    print('num test: ' + str(test_df.shape[0]))
    return train_df, val_df, test_df

# this function copy images by channel to its destination folder
def copy_files_for_channel(df, channel_name, verbose=False):
    print('\nCopying files for {} images in channel: {}...'.format(df.shape[0], channel_name))
    for i in range(df.shape[0]):
        target_fname = df.iloc[i]['image_file_name']
#         if verbose:
#             print(target_fname)
        src = "{}/{}".format(IMAGES_DIR, target_fname) #f"{IMAGES_DIR}/{target_fname}"
        dst = "{}/{}/{}".format(output_path,channel_name,target_fname)
        shutil.copyfile(src, dst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=str, default="")
    parser.add_argument("--input-data", type=str, default="classes.txt")
    args, _ = parser.parse_known_args()


    c_list = args.classes.split(',')
    input_data = args.input_data
    
    print(c_list)
        
    CLASSES_FILE = os.path.join(input_path, input_data)

    EXCLUDE_IMAGE_LIST = ['087.Mallard/Mallard_0130_76836.jpg']
    CLASS_COLS      = ['class_number','class_id']
    
    if len(c_list)==0:
        # Otherwise, you can use the full set of species
        CLASSES = []
        for c in range(200):
            CLASSES += [c + 1]
        prefix = prefix + '-full'
    else:
        CLASSES = list(map(int, c_list))

            
    classes_df = pd.read_csv(CLASSES_FILE, sep=' ', names=CLASS_COLS, header=None)

    criteria = classes_df['class_number'].isin(CLASSES)
    classes_df = classes_df[criteria]

    class_name_list = sorted(classes_df['class_id'].unique().tolist())
    print(class_name_list)
    
    
    train_df, val_df, test_df = get_train_val_dataframes(input_path, CLASSES, EXCLUDE_IMAGE_LIST, SPLIT_RATIOS)
        
    for c in class_name_list:
        os.mkdir('{}/{}/{}'.format(output_path, 'validation', c))
        os.mkdir('{}/{}/{}'.format(output_path, 'test', c))
        os.mkdir('{}/{}/{}'.format(output_path, 'train', c))

    copy_files_for_channel(val_df,   'validation')
    copy_files_for_channel(test_df,  'test')
    copy_files_for_channel(train_df, 'train')
    
    # export manifest file for validation
    train_m_file = "{}/manifest/train.csv".format(output_path)
    train_df.to_csv(train_m_file, index=False)
    test_m_file = "{}/manifest/test.csv".format(output_path)
    test_df.to_csv(test_m_file, index=False)
    val_m_file = "{}/manifest/validation.csv".format(output_path)
    val_df.to_csv(val_m_file, index=False)
    
    print("Finished running processing job")
