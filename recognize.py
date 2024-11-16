import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os

# Preprocessing data retrieving from current directory and parsing data into np arrays for encoding and splitting 
direct = "./faces"
images, labels = [] , []

for employee in os.listdir(direct):
    employee_path = os.path.join(direct , employee)
    if os.path.isdir(employee_path):
        for image in os.listdir(employee_path):
            img_path = os.path.join(employee_path , image)
            img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE) #use opencv to convert the image to greyscale for easier processing