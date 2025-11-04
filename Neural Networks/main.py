import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
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


def train_random(X_train, X_test, y_train, y_test):

    clf = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=5)
    clf.fit(X_train[:3000], y_train[:3000]) # reduce the train set size to shorten the training time

    print("The best parameter values found are:\n")
    print(clf.best_params_)

    # store the best model found in "bestmodel"
    bestmodel = clf.best_estimator_

    y_pred = bestmodel.predict(X_test)
    print(f"The accuracy score of the best model using RandomizedSearchCV is {accuracy_score(y_test, y_pred)}\n")

    plt.figure(figsize=(12,8))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        sample = random.choice(X_test)
        plt.imshow(sample.reshape(28,28))
        pred = bestmodel.predict(sample.reshape(1,-1))
        plt.title(f"Predicted as {pred}")
        plt.axis("off")
    plt.show()
    plt.tight_layout()
    

def train_grid(X_train, X_test, y_train, y_test):
     
    clf = GridSearchCV(estimator=model, param_grid=parameters, cv=5, n_jobs=-1)
    clf.fit(X_train[:3000], y_train[:3000]) # reduce the train set size to shorten the training time

    print("The best parameter values found are:\n")
    print(clf.best_params_)

    # store the best model found in "bestmodel"
    bestmodel = clf.best_estimator_

    y_pred = bestmodel.predict(X_test)
    print(f"The accuracy score of the best model using GridSearchCV is {accuracy_score(y_test, y_pred)}\n")

    plt.figure(figsize=(12,8))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        sample = random.choice(X_test)
        plt.imshow(sample.reshape(28,28))
        pred = bestmodel.predict(sample.reshape(1,-1))
        plt.title(f"Predicted as {pred}")
        plt.axis("off")
    plt.show()
    plt.tight_layout()
    

if __name__ == "__main__":

    digits_data, labels_data = load_data()
    split = 0.7, 0.3 # train, test
    # normalize data
    digits_data /= 255.0

    split_ind = int(len(digits_data)*split[0])
    X_train, X_test, y_train, y_test = digits_data[:split_ind], digits_data[split_ind:], labels_data[:split_ind], labels_data[split_ind:]
    print(f"Shapes of training and test data: {X_train.shape} and {X_test.shape}")


    model = MLPClassifier().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Using MLPClassifier with the default parameter values gives an accuracy of {accuracy_score(y_pred, y_test)}")
    print(classification_report(y_pred, y_test))
    

    parameters = {'hidden_layer_sizes':[ 170, 200, 220],
              'alpha': [0.0008, 0.0009, 0.001,0.002], 
              'max_iter': [500, 600, 650, 700], 
              'learning_rate_init':[0.007, 0.008, 0.009]}

    model = MLPClassifier()


    train_random(X_train, X_test, y_train, y_test)
    train_grid(X_train, X_test, y_train, y_test)