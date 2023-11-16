# -*- coding: utf-8 -*-
"""Assignment 3: Churning_Customers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NqDF5SsCnFqENOY1TIOB2qDI1gfZa2-1

# ***Assignment 3: Churning_Customers***

## **Importing Relivant Libraries**
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Dropout
from keras.models import Model
from keras.layers import Input, Dense
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
import pickle

"""# **Installing Required Library**"""

!pip install scikeras

"""# **Mounting and Loading Data**"""

from google.colab import drive
drive.mount('/content/drive')

Data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CustomerChurn_dataset.csv')

pd.options.display.max_columns = None

Data



"""# **Dropping Irrelivant Data**"""

Contributing_Features = Data.drop('customerID', axis =1)

Contributing_Features

"""# **Removing attributes with more than 30 % null values**"""

Condition = 0.7 * len(Contributing_Features)
Filtered_Data = Contributing_Features.dropna(thresh = Condition, axis=1)

"""# **Checking if there are still other null values**"""

nan_values = Filtered_Data.isna().any()

print (nan_values)



"""# **Encoding Non-Numeric columns**"""

To_be_Encoded = ['gender',	'Partner',	'Dependents', 'PhoneService',	'MultipleLines',	'InternetService',	'OnlineSecurity',	'OnlineBackup',	'DeviceProtection',	'TechSupport',	'StreamingTV',	'StreamingMovies',	'Contract',	'PaperlessBilling',	'PaymentMethod', 'Churn', 'TotalCharges']

for non_numeric_attribute in To_be_Encoded:
  Filtered_Data[non_numeric_attribute],_=pd.factorize(Filtered_Data[non_numeric_attribute])

Filtered_Data



"""# **Extracting dependent variable(X) and indepent variable(Y)**"""

Y_feature = pd.DataFrame(Filtered_Data['Churn'], columns= ['Churn'])

X_features  = Filtered_Data.drop('Churn', axis= 1)

Attribut = Filtered_Data.columns

Attribut

"""# **Extracting Feature importance**"""

from xgboost.sklearn import XGBRFClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot


# fit model training data
model = XGBClassifier()
model.fit(X_features, Y_feature)
# feature importance
importances = model.feature_importances_


# Print or use feature importances
sorted_indices = np.argsort(importances)[::-1]
for index in sorted_indices:
   print(f"'{Attribut[index]}', '{importances[index]}'")
   print()

"""# **Ploting Feature Importance**"""

# Plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_features.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_features.shape[1]), Attribut[sorted_indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()







"""# **Resetting Y variable and Scalling the X values**"""

final_Y_feature = Y_feature

from re import X
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_features)
final_X_feaures = pd.DataFrame(scaler.transform(X_features), columns=  ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges',])

final_X_feaures

"""# **Spliting data for training, validation, and testing**"""

Xtrain,X,Ytrain,Y=train_test_split(final_X_feaures,final_Y_feature,test_size=0.2,random_state=42,stratify=final_Y_feature)

Xvalidate,Xtest,Yvalidate,Ytest=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

Ytest

"""# **Building and Training Model First Model**



"""

# Defining input layer
input_layer = Input(shape=(19,))  # Replace input_dim with the number of features

# Defining hidden layers
hidden1 = Dense(50, activation='relu')(input_layer)
dropout1 = Dropout(0.1)(hidden1)
hidden2 = Dense(30, activation='relu')(dropout1)
dropout2 = Dropout(0.1)(hidden2)
hidden3 = Dense(30, activation='relu')(dropout2)
dropout3 = Dropout(0.1)(hidden3)
hidden4 = Dense(30, activation='relu')(dropout3)
dropout4 = Dropout(0.1)(hidden4)
hidden5 = Dense(30, activation='relu')(dropout4)
dropout5 = Dropout(0.1)(hidden5)
hidden6 = Dense(30, activation='relu')(dropout5)
dropout6 = Dropout(0.1)(hidden6)
hidden7 = Dense(30, activation='relu')(dropout6)
dropout7 = Dropout(0.1)(hidden7)
hidden8 = Dense(10, activation='relu')(dropout7)
dropout8 = Dropout(0.1)(hidden8)

# Output layer
output_layer = Dense(1, activation='sigmoid')(dropout8)

# The model
model = Model(inputs=input_layer, outputs=output_layer)

optimizer = SGD(momentum = 0.9)
# Compiling model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(Xtrain, Ytrain, epochs=4, batch_size=32, validation_data=(Xvalidate, Yvalidate))


# Making prediction on the test set and calculate accuracy and AUC score
Ypred = model.predict(Xtest)
accuracy = accuracy_score(Ytest, (Ypred > 0.5).astype(int))
auc_score = roc_auc_score(Ytest, Ypred)
Ypred
print("Accuracy:", accuracy)
print("AUC Score:", auc_score)


# Ploting training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Ploting training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

"""# **Building Grid search to find optimal Parameters**"""

#Function to create model
def create_model(neurons=30):
    input_layer = Input(shape=(19,))
    hidden1 = Dense(neurons, activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(neurons, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(hidden2)
    hidden3 = Dense(neurons, activation='relu')(dropout2)
    dropout3 = Dropout(0.2)(hidden3)
    hidden4 = Dense(neurons, activation='relu')(dropout3)
    dropout4 = Dropout(0.2)(hidden4)
    hidden5 = Dense(neurons, activation='relu')(dropout4)
    dropout5 = Dropout(0.2)(hidden5)
    hidden6 = Dense(neurons, activation='relu')(dropout5)
    dropout6 = Dropout(0.2)(hidden6)
    output_layer = Dense(1, activation='sigmoid')(dropout6)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wraping the model with KerasClassifier
model = KerasClassifier(build_fn=create_model, neurons=30, epochs=10, batch_size=10, verbose=0)

# Defining the grid search parameters
param_grid = {
    'neurons': [10, 20],
    'batch_size': [10, 20,],
    'epochs': [10, 20, 30],
    'optimizer': ['Adam'],
    # 'activation': ['ReLU', 'Leaky ReLU', 'Tanh', 'Swish']
}

# Creating GridSearch
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(Xtrain, Ytrain)

# Make predictions on the test set with the best model
best_model = grid_result.best_estimator_
Ypred = best_model.predict(Xtest)
accuracy = accuracy_score(Ytest, (Ypred > 0.5).astype(int))
auc_score = roc_auc_score(Ytest, Ypred)

print("Accuracy:", accuracy)
print("AUC Score:", auc_score)

# Print the best parameters and best score
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

"""# **Building and training Second(chosen) model based on Grid search results**"""

# Defining input layer
input_layer = Input(shape=(19,))

# Defining hidden layers
hidden1 = Dense(20, activation='relu')(input_layer)
dropout1 = Dropout(0.1)(hidden1)
hidden2 = Dense(20, activation='relu')(dropout1)
dropout2 = Dropout(0.2)(hidden2)
hidden3 = Dense(20, activation='relu')(dropout2)
dropout3 = Dropout(0.2)(hidden3)
hidden4 = Dense(20, activation='relu')(dropout3)
dropout4 = Dropout(0.2)(hidden4)


# Defining output layer
output_layer = Dense(1, activation='sigmoid')(dropout4)

# Creating the model
model = Model(inputs=input_layer, outputs=output_layer)

optimizer = 'Adam'
# Compiling the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history= model.fit(Xtrain, Ytrain, epochs=20, batch_size=20, validation_data=(Xvalidate, Yvalidate))

"""# **Predictions on the test set and calculating of accuracy and AUC score**"""

# Make predictions on the test set and calculate accuracy and AUC score
Ypred = model.predict(Xtest)
accuracy = accuracy_score(Ytest, (Ypred > 0.5).astype(int))
auc_score = roc_auc_score(Ytest, Ypred)
Ypred
print("Accuracy:", accuracy)
print("AUC Score:", auc_score)

Ypred

"""# **Saving the chosen model**"""

with open('churn_model.sav', 'wb') as model_file:
    pickle.dump(model, model_file)



"""# **Ploting training & validation accuracy and training & validation loss for the chosen model**"""

# Ploting training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Ploting training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
plt.show()
