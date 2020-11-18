"""
Solution for the TP2 - Part2 - Problem4
Apply the Artificial Neuronal Network logic findable in PythonCodes/ann.py onto the dataset PIMA Indians Diabetes

:author: Macdrien
"""

import pandas

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense

# Import data
columns = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin thickness', 'Insulin', 'Body Mass Index', 'Diabetes Pedigree Function', 'Age', 'Outcome']
data = pandas.read_csv('Datasets/pima-indians-diabetes.data', names=columns)

# Select features and column to predict
features = columns[:-1]
X = data[features]
y = data[columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Data scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model creation
model = Sequential()
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Initialize the model
model.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Do prediction
y_pred = model.predict(X_test)
y_pred = [[0] if pred <= 0.5 else [1] for pred in y_pred]

# Check the result
y_confusion_matrix = confusion_matrix(y_test, y_pred)
print(y_confusion_matrix)
