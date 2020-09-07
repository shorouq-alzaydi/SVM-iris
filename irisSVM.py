import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.datasets import load_iris
warnings.filterwarnings(action='ignore')                  # Turn off the warnings.

# Load iris dataset
data = load_iris()


X = data['data']
columns = list(data['feature_names'])
print("Explanatory variables " + str(columns))

Y = data['target']
labels = list(data['target_names'])
print("Response variable: "+ str(labels))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

C_grid = 0.02*np.arange(1,20)
gamma_grid = 0.02*np.arange(1,50)
parameters = {'C': C_grid, 'gamma' : gamma_grid}
gridCV = GridSearchCV(SVC(kernel='rbf'), parameters, cv=10, n_jobs=-1)              # "n_jobs = -1" means "use all the CPU cores".
gridCV.fit(X_train, Y_train)
best_C = gridCV.best_params_['C']
best_gamma = gridCV.best_params_['gamma']

print("SVM best C : " + str(best_C))
print("SVM best gamma : " + str(best_gamma))

SVM_best = SVC(kernel='rbf', C=best_C,gamma=best_gamma)
SVM_best.fit(X_train, Y_train);
Y_pred = SVM_best.predict(X_test)
print( "SVM best accuracy : " + str(np.round(metrics.accuracy_score(Y_test,Y_pred),3)))

#SVM hyperparameter optimization (Polynomial kernel)

C_grid = 0.0001*np.arange(1,30)
gamma_grid = 0.01*np.arange(1,30)
parameters = {'C': C_grid, 'gamma' : gamma_grid}
gridCV = GridSearchCV(SVC(kernel='poly'), parameters, cv=10, n_jobs=-1)              # "n_jobs = -1" means "use all the CPU cores".
gridCV.fit(X_train, Y_train)
best_C = gridCV.best_params_['C']
best_gamma = gridCV.best_params_['gamma']

print("SVM best C : " + str(best_C))
print("SVM best gamma : " + str(best_gamma))

SVM_best = SVC(kernel='poly', C=best_C,gamma=best_gamma)
SVM_best.fit(X_train, Y_train);
Y_pred = SVM_best.predict(X_test)
print( "SVM best accuracy : " + str(np.round(metrics.accuracy_score(Y_test,Y_pred),3)))