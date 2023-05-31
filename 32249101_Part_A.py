import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold


train_data = pd.read_csv("TrainingDataBinary.csv", skiprows=0) # Load training data from csv file

test_data = pd.read_csv("TestingDataBinary.csv", skiprows=0) # Load testing data from csv file

x_data = train_data.iloc[: , :128] # selecting data from all 128 columns and all rows of dataset available
y_data = train_data.iloc[: , 128] # selecting only data from 129th column and all rows of dataset available

x_test_data = test_data.iloc[: , :128]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.17, random_state=42) # splitting data into a training set and testing set

# Pipeline parameters
pipeline = Pipeline([
    ('scaler', StandardScaler()), # standardizes features
    ('pca', PCA(n_components=87)), # principle component analysis which decreases features to 87
    ('lr', LogisticRegression(max_iter=100000, solver='liblinear', penalty='l2', C = 1000))]) # Logistic Regression model used with set parameters

# Perform cross-validation with training data
scores = cross_val_score(pipeline, x_train, y_train, cv=10)
f1_scores = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='f1_micro')

# train and evaluate the final model on the test set
pipeline.fit(x_train, y_train)

# prints accuracy and error value estimates
print("Mean CV Accuracy for 10 folds: %f" % scores.mean())
print("Mean CV Error for 10 folds: %f" % (1 - scores.mean()))

print("Test Accuracy: %f" % pipeline.score(x_test, y_test))
print("Test Error: %f" % (1 - pipeline.score(x_test, y_test)))

print("F1_score: %f" % f1_scores.mean())

y_test_pred = pipeline.predict(x_test)
temp = pipeline.predict(x_test_data) # predict y values for 100 samples test set

print (temp)
y_test_data = pd.DataFrame({'target': temp})

test_results = pd.concat([test_data, y_test_data], axis=1) # combine predicted values with test dataset

# test_results.to_csv("TestingResultsBinary.csv", index=False, header=None)# Save results to csv file

# construct confusion matrix
cm = confusion_matrix(y_test, y_test_pred, labels=pipeline.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=pipeline.classes_)
disp.plot()
plt.show()