import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from tensorflow import keras

"""
data pre-processing
"""
# df = pd.read_csv("diabetes.csv")  # read csv file
# df = df.drop(["Income"], axis=1)
# df = df.dropna()  # drop NA
# df.to_csv("diabetes_preprocessed.csv")

df = pd.read_csv("diabetes_preprocessed.csv")  # read csv file

"""
feature selection
"""
X = df.iloc[:, 2:21]
y = df.iloc[:, 1]
feature_model = ExtraTreesClassifier()  # decision tree
feature_model.fit(X, y)
feature_score = pd.Series(feature_model.feature_importances_, index=X.columns)
# feature_score.nlargest(25).plot(kind='bar')
X = X.drop(["CholCheck"], axis=1)
X = X.drop(["HvyAlcoholConsump"], axis=1)
X = X.drop(["AnyHealthcare"], axis=1)

# """
# dataset splitting
# """
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# """
# deep learning modeling
# """
model = keras.Sequential([
    keras.layers.Dense(17, activation='relu'),
    keras.layers.Dense(14, activation='relu'),
    keras.layers.Dense(1, activation="sigmoid")])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=4)
model.evaluate(x_test, y_test, batch_size=4, verbose=2)
model.summary()

