# derm_failure
Script to detect dermoscopy images, binary classify them using a CNN and sort them into folders with high/low confidence of the benign class. The sorting script uses the maximum softmax response as a confidence function. The threshold for high confidence is a softmax response of 0.99 or higher for the benign class. To execute the script create the virtual environment and run "python sorting.py --root_folder=<folder_with_images>". If repeated looping over the selected folder is wanted run the script with the additonall argument "--loop=True".
# Training
The PyTorch scrip for training the classifier is included. To accquire the images go to the respective websites of the isic2020 challenge, ham10000, derm 7 point and ph2. The respective csv files transform the mixed multiclass challenges into a binary classification between benign and malignant based on the lesion type.
# Testing
To view testing results run the test.py script. This will create a confusion matrix as well as print the AUC score.
