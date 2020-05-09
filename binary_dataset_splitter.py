# -*- coding: utf-8 -*-
"""
Creates a file structure for properly formatting a dataset into 
train/test/validation For binary classification (malign/benign, in this script).

This should NOT be used for the most popular methods cross-validation and
K-fold cross validation, which is recommended.
https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7

Popular ML libraries automatically support this, such as tensorflow, e.g.
https://medium.com/fenwicks/tutorial-5-cross-validation-with-tensorflow-flowers-34f7ac36230b

Run this in a folder containing only this script
as well as the folders "all_benign" and "all_malign" containing the samples.
Optionally, if you have created data augmentation samples, you may include these
in folders inside "all_benign" and "all_malign" called "augmented"

This will split them into train/test/validation
and it will also generate .csv files accordingly.

Note that when this script runs, it REMOVES any currently-present folders
named "code" and "split_samples", so do not keep important data here, as it will be
overwritten by this script!

Also, if augmentation is enabled, and the file names are the same as
a file in the main folder, the non-augmented file will take precendence.
For example, an augmented file named "1.png" exists, and so does a non-augmented
file named "1.png". If these are both sent to the train directory, the
one that appears in the directory will be the NON-AUGMENTED version!
    
Example directory before starting:
    - all_malign
        - sample_1.png
        - sample_2.png
    - all_benign
        - sample_1.png
        - sample_2.png
    - this script (binary_dataset_splitter.py)
    
And the resulting directory after running this script:
    - all_malign
        - sample_1.png
        - sample_2.png
    - all_benign
        - sample_1.png
        - sample_2.png
    - this script (binary_dataset_splitter.py)
    - code
        - benign_test.csv
        - benign_train.csv
        - benign_validation.csv
        - malign_test.csv
        - malign_train.csv
        - malign_validation.csv
    - split_samples
        - test
            - benign
                - some samples...
            - malign
                - some samples...
        - train
            - benign
                - some samples...
            - malign
                - some samples...
        - validation
            - benign
                - some samples...
            - malign
                - some samples...
                
# TODO - implement below features if necessary in the future (add as GitHub requests...)
    * Speed up the script
    * Add print statements to indicate progress
    * Use a config file (text file or YAML) instead of directly editing this file
    * Change labels "benign" and "malignant" to user-defined labels
    * Extend capabilities for multiple classes instead of just binary

@author: Jesse Deppisch
"""

import shutil
import os
import glob
import random
import math
import csv

# Variables
TRAIN = 0.7
TEST = 0.2
VALIDATION = 0.1
DATA_TYPE = "png"
RENAME_DATA = False # Changes file names to sequential numbers: 1 to n
INCLUDE_AUGMENTED_DATA_IN_TRAIN = True # Includes augmented files in training set
                               # Note that these are NOT counted in the train/test splitting

def create_folders_in_dir(main_dir, folder_string_list):
    for s in folder_string_list:
        try:
            os.mkdir(os.path.join(main_dir, s))
        except FileExistsError:
            print("Folder", s, "already exists.")
            
def delete_folders_in_dir(main_dir, folder_string_list):
    for s in folder_string_list:
        try:
            shutil.rmtree(os.path.join(main_dir, s))
            print("Clearing old splits...")
        except FileNotFoundError:
            # Good, the user didn't try to run the script multiple times
            break; # Don't care about this exception: it's good
            
def set_up_folder_structure():
    # Some useful variables
    base_dir = "./"
    
    # First, create all the necessary root-level folders
    create_folders_in_dir(base_dir, ["code", "split_samples"])
        
    # Create the subdirectories
    data_dir = os.path.join(base_dir, "split_samples")
    split_folders = ['test', 'train', 'validation']
    create_folders_in_dir(data_dir, split_folders)
    
    # Create benign and malign folder within each of these
    for s in split_folders:
        sub_dir = os.path.join(data_dir, s)
        create_folders_in_dir(sub_dir, ['benign', 'malign'])
        
    return (base_dir,data_dir)

def remove_split_folders():
    # Delete folders if this script has already been run once
    # This avoids train/test/validaton from accuring EVERY sample over time
    
    base_dir = "./"
    delete_folders_in_dir(base_dir, ['code', 'split_samples'])

def rename_data(d):
    # Note: d is a dictionary of file paths for different types; see main method
    for class_type in ['benign', 'malign']:
        # First, get list of all relevant file names
        path_query = os.path.join(d[class_type], "*." + DATA_TYPE) # Only looking for our data type
        sample_names = glob.glob(path_query)
        num_samples = len(sample_names)
        
        try:
            for i in range(num_samples):
                new_name = os.path.join(d[class_type], str(i) + "." + DATA_TYPE)
                os.rename(sample_names[i], new_name)
        except FileExistsError:
            print("You already have samples with filenames 1 through n.\
                  Please set RENAME_DATA=False and run this script again.")
    print("I renamed your original data to have numbers instead. To disable this, set RENAME_DATA=False")

def split_data(d):
    # Note: d is a dictionary of file paths for different types; see main method
    for class_type in ['benign', 'malign']:
        # First, get list of all relevant file names
        path_query = os.path.join(d[class_type], "*." + DATA_TYPE) # Only looking for our data type
        sample_names = glob.glob(path_query)
        total_num_samples = len(sample_names)
        
        # Because we must use all the files, be careful with numbers
        # otherwise, might leave out some files
        num = {} # Amount of each type of sample (train, test, validaton)
        num['train'] = math.floor(TRAIN * total_num_samples)
        num['test'] = math.floor(TEST * total_num_samples)
        num['validation'] = total_num_samples - (num['train'] + num['test'])
        
        # Package these into start and end indexes for the below for loop
        # This makes it nicer to iterate over arrays
        start_index = {} # Where the for-loop will start
        start_index['train'] = 0
        start_index['test'] = num['train']
        start_index['validation'] = num['train'] + num['test']
        
        # Next, generate random indices to pick files at random
        random.shuffle(sample_names) # Shuffle list, for randomness
        
        # Finally, actually split the files up
        for split_type in ['train', 'test', 'validation']:
            #from_dir = d[class_type] # Directory we're moving samples FROM
            to_dir = os.path.join(d[split_type], class_type)  # Directory we're moving samples TO
            
            # Iterate over each file and move it
            si = start_index[split_type]
            for i in range(si, si + num[split_type]):
                # Get file name of the sample we'd like to move
                # Specifically, this is its location
                sample_location = sample_names[i] 
                
                # Copy the file
                shutil.copy(sample_location, to_dir)
        
            # Tell the user what we did
            print("Copied", num[split_type], "samples of", class_type, "to", split_type)
                
def copy_augmented_data_to_train(d):
    # Note: d is a dictionary of file paths for different types; see main method
     for class_type in ['benign', 'malign']:
        # First, get list of all relevant file names
        augmented_data_path = os.path.join(d[class_type], 'augmented')
        path_query = os.path.join(augmented_data_path, "*." + DATA_TYPE)
        aug_sample_names = glob.glob(path_query)
        total_num_aug_samples = len(aug_sample_names)
         
        # Next, copy these files to their respective folder
        to_dir = os.path.join(d['train'], class_type)

        # Iterate over each file and move it
        for i in range(total_num_aug_samples):
            # Get file name of the sample we'd like to move
            # Specifically, this is its location
            sample_location = aug_sample_names[i] 
            
            # Copy the file
            shutil.copy(sample_location, to_dir)
         
        print('Moved', total_num_aug_samples, 'augmented samples of', class_type, "to train")
    
def generate_csv_files(d):
    # Note: d is a dictionary of file paths for different types; see main method
    for split_type in ['train', 'test', 'validation']:
        for class_type in ['benign', 'malign']:
            # Get list of all relevant samples in this folder
            folder_path = os.path.join(d[split_type], class_type)
            path_query = os.path.join(folder_path, "*." + DATA_TYPE) # Only looking for our data type
            sample_names = glob.glob(path_query)
            # num_samples = len(sample_names)
            
            # Now, save these as a CSV
            csv_filename = d['code'] + class_type + "_" + split_type + ".csv"
            with open(csv_filename, 'w', newline='') as myfile: # Newline fixes: https://stackoverflow.com/questions/3348460/csv-file-written-with-python-has-blank-lines-between-each-row
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL) # Not sure whaat quoting is, but from stackoverflow
                # wr.writerows(sample_names)
                for sample_path in sample_names:
                    sample_filename = os.path.basename(sample_path)
                    wr.writerow([sample_filename]) # Must be in [], see https://stackoverflow.com/questions/15129567/csv-writer-writing-each-character-of-word-in-separate-column-cell
            
            # Tell the user what we did
            print("Wrote CSV file for", class_type, ":", split_type)

if __name__ == '__main__':
    # Checking your settings...
    assert round(TRAIN+TEST+VALIDATION, 2) == 1.0, \
        "Your TRAIN/TEST/VALIDATION split settings do not sum to 1. Please adjust these."
    
    # First...
    # Delete folders if this script has already been run once
    # This avoids train/test/validaton from accuring EVERY sample over time
    remove_split_folders()
    
    # Create necessary folders
    base_dir, data_dir = set_up_folder_structure()
    
    # Store useful folder paths
    d = {}
    d['benign'] = os.path.join(base_dir, 'all_benign')
    d['malign'] = os.path.join(base_dir, 'all_malign')
    d['code'] = os.path.join(base_dir, 'code/')
    d['train'] = os.path.join(data_dir, 'train')
    d['test'] = os.path.join(data_dir, 'test')
    d['validation'] = os.path.join(data_dir, 'validation')
    
    # Do you want to rename your data samples?
    if RENAME_DATA:
        rename_data(d)
        
    # Do you want to include data from the "augmented" folders in training datasets?
    if INCLUDE_AUGMENTED_DATA_IN_TRAIN:
        copy_augmented_data_to_train(d)
    
    # Split the data by copying from root directory
    split_data(d)
    
    # Generate CSV files based on this
    generate_csv_files(d)