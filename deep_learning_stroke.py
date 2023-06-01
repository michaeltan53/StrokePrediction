import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from tensorflow import keras

"""
data pre-processing
"""
# df = pd.read_csv("train_strokes.csv")  # read csv file
# df = df.dropna()  # drop NA
# # Transferring string into integer
# df['gender'] = pd.factorize(df["gender"])[0].astype(np.uint16)
# df['ever_married'] = pd.factorize(df["ever_married"])[0].astype(np.uint16)
# df['work_type'] = pd.factorize(df["work_type"])[0].astype(np.uint16)
# df['Residence_type'] = pd.factorize(df["Residence_type"])[0].astype(np.uint16)
# df['smoking_status'] = pd.factorize(df["smoking_status"])[0].astype(np.uint16)
# df.to_csv("stroke_train_preprocessed.csv")

df = pd.read_csv("stroke_train_preprocessed.csv")  # read csv file

"""
feature selection
"""
X = df.iloc[:, 2:12]
y = df.iloc[:, -1]
feature_model = ExtraTreesClassifier()  # decision tree
feature_model.fit(X, y)
feature_score = pd.Series(feature_model.feature_importances_, index=X.columns)
feature_score.nlargest(15).plot(kind='bar')
print(feature_score)
# plt.show()
X = X.drop(["ever_married"], axis=1)
X = X.drop(["hypertension"], axis=1)

"""
dataset splitting
"""
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

"""
deep learning modeling
"""
model = keras.Sequential([
    keras.layers.Dense(7, activation='relu'),
    keras.layers.Dense(12, activation='relu'),
    keras.layers.Dense(12, activation='relu'),
    keras.layers.Dense(1, activation="sigmoid")])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=4)
model.evaluate(x_test, y_test, batch_size=4, verbose=2)
model.summary()
