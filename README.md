# DermoSort
DermoSort is a small repository to lesser the burdon on dermatologist in skin cancer detections. The repository uses public skin cancer datasets to train a SOTA neural network to classify dermoscopy images. The dermatologist can then drop all his images into a folder and they get sorted into a folder with high confidence of the benign class, which he can ignore, and a folder with small or higher probability of the malignant class for further investigation. This allows for a work reduction of around 82 % (1 - fraction predicted positive) with a sensitivity of 96 %. There are many false positives (FPR = 14 %), because of the stringent threshold for the benign class.
# Sorting
Sorting is the main script. It comes with a pretrained classifier so training and testing are optional. Run it to detect dermoscopy images and sort them into folders with high/low confidence of the benign class. The sorting script uses the maximum softmax response as a confidence function. The threshold for high confidence is a softmax response of 0.99 or higher for the benign class. To execute the script create the virtual environment and run "python sorting.py --root_folder=<folder_with_images>". If repeated looping over the selected folder is wanted run the script with the additonall argument "--loop=True".
# Training
The PyTorch scrip for training the classifier is included. To accquire the images go to the respective websites of the isic2020 challenge, ham10000, derm 7 point and ph2. The respective csv files transform the mixed multiclass challenges into a binary classification between benign and malignant based on the lesion type.
# Testing
To view results on the testset run the test.py script. This will create a confusion matrix as well as print the AUC score.
![plot](https://github.com/Levin-Kobelke/DermoSort/edit/master/confMat.png?raw=true)

# Graphic User Interface
Run the gui.py script for a graphical user interface. In this you can select the correct folder via a file dialog and sort by clicking. 
# Coming up:
Docker version and executable.
