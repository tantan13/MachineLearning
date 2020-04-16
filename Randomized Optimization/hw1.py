import sklearn
# Importing the required packages 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from numpy import genfromtxt
  
# Function importing Dataset 
def importdata(): 
    fifa_data = pd.read_csv('data.csv', delimiter = ',')

    # Printing the dataswet shape 
    print ("Dataset Length: ", len(fifa_data)) 
    print ("Dataset Shape: ", fifa_data.shape) 
      
    # Printing the dataset obseravtions 
    #print ("Dataset: ", fifa_data.head()) 
    print(fifa_data.head())
    return fifa_data 
  
# Function to split the dataset 
def splitdataset(fifa_data): 
  
    # Seperating the target variable 
    X = fifa_data.iloc[:, 54:] #other used data values:crossing, finishing, heading, short passing, volleys, dribbling, curves, etc)
    X = X.values[:,0:X.shape[1]-2]
    Y = (fifa_data.iloc[:, 7]) #predicting their overall (target value) 
  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 
      
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
  
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 


if __name__ == '__main__':
    print("hello")
    data = importdata()

    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    print(type(X_train))
    print("==================================")
    #rint(y_train)
    #print(np.argwhere(np.isnan(X_train)))
    print(np.any(pd.isnull(y_train)))
    #prediction(X_test, tarin_using_entropy(X_train, X_test, y_train))
    trained_classifier = tarin_using_entropy(X_train, X_test, y_train)
    y_pred = prediction(X_test, trained_classifier)
    cal_accuracy(y_test, y_pred)

