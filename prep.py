import numpy as np
import cv2
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


# Preprocessing data retrieving from current directory and parsing data into np arrays for encoding and splitting 
direct = "./faces"
images, labels = [] , []

for employee in os.listdir(direct):
    employee_path = os.path.join(direct , employee)
    if os.path.isdir(employee_path):
        for image in os.listdir(employee_path):
            img_path = os.path.join(employee_path , image)
            img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE) #use opencv to convert the image to greyscale for easier processing
            img = cv2.resize( img, (128,128)) #resize the shape of the image
            images.append(img.flatten()) #make image 1d 
            labels.append(employee)

images = np.array(images) 
labels = np.array(labels)
enc = LabelEncoder()
y = enc.fit_transform(labels)  #encode categorical data aka labels of each name
x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)  #split into testing and training 
#we can apply pca to x because it has very similar features because we need to maintain variance in the dataset
pca = PCA()
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_train_pca)

"""Part 2   Training the Model   """

svm = SVC(kernel='rbf', C=10, gamma='scale')  # Adjust hyperparameters
svm.fit(x_train_pca, y_train)
