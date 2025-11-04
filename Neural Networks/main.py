import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report


def load_data():

    digits = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module1/L1/data/digits.csv")

    labels = digits['label']
    digits = np.array(digits.drop('label', axis=1)).astype('float')
    digits.shape, labels.shape

    return digits, labels


def draw(data):
        
    plt.figure(figsize=(12,4))
    for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(random.choice(data).reshape(28,28))
            plt.axis("off")

    plt.show()

    

if __name__ == "__main__":

    digits_data, = load_data
