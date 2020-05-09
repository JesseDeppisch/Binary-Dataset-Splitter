# Binary-Dataset-Splitter
Creates a file structure to split a dataset into train/test/validation for binary classification (malign/benign, in this script).

This was made to be a useful script for some simple ML tasks I did for my machine learning class, and it helped me learn a bit about file manipulation with Python. This is not intended to be a super-useful utility for ML practictioners - in fact, it actually doesn't make much sense the more I think about it.

This should NOT be used for the most popular methods cross-validation and K-fold cross validation, which is recommended.
https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7

Popular ML libraries automatically support this, such as tensorflow, e.g.
https://medium.com/fenwicks/tutorial-5-cross-validation-with-tensorflow-flowers-34f7ac36230b

## Usage
Run this in a folder containing only this script as well as the folders "all_benign and all_malign" containing the samples. Optionally, if you have created data augmentation samples, you may include these in folders inside "all_benign" and "all_malign" called "augmented"

This will split them into train/test/validation and it will also generate .csv files accordingly.

Note that when this script runs, it REMOVES any currently-present folders named "code" and "split_samples", so do not keep important data here, as it will be overwritten by this script!

Also, if augmentation is enabled, and the file names are the same as a file in the main folder, the non-augmented file will take precendence. For example, an augmented file named "1.png" exists, and so does a non-augmented file named "1.png". If these are both sent to the train directory, the one that appears in the directory will be the NON-AUGMENTED version!
    
Example directory before starting:
```
    - all_malign
        - sample_1.png
        - sample_2.png
    - all_benign
        - sample_1.png
        - sample_2.png
    - this script (binary_dataset_splitter.py)
```
    
And the resulting directory after running this script:
```
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
```

### TODO - implement below features if necessary in the future
* Speed up the script
* Add print statements to indicate progress
* Use a config file (text file or YAML) instead of directly editing this file
* Change labels "benign" and "malignant" to user-defined labels
* Extend capabilities for multiple classes instead of just binary
