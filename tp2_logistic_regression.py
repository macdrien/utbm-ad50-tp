import os

from draw_classes import draw_classes, draw_classes_regions
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

basedir = os.path.abspath(os.path.dirname(__file__))

graph_classes_color = ('red', 'green')

def apply_logistic_regression(filepath, columns, features, to_predict, print_confusion_matrix=False):
      data = pandas.read_csv(filepath, names=columns)
      X = data[features]
      y = data[to_predict]

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

      model = LogisticRegression(random_state=0, max_iter=256)
      model.fit(X_train, y_train)
      y_predicted = model.predict(X_test)

      y_confusion_matrix = confusion_matrix(y_test, y_predicted)
      if print_confusion_matrix:
            print('Confusion matrix :\n{}'.format(y_confusion_matrix))

      return data, model, X_train, X_test, y_train, y_test, y_predicted, y_confusion_matrix

# Work on PIMA Indians Diabetes
diabetes_filepath = basedir + '/Datasets/pima-indians-diabetes.data'
diabetes_columns = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin thickness', 'Insulin', 'Body Mass Index', 'Diabetes Pedigree Function', 'Age', 'Outcome']
# diabetes_features = ['Pregnancies', 'Glucose']
diabetes_features = ['Glucose', 'Insulin']
# diabetes_features = ['Body Mass Index', 'Age']
diabetes_to_predict = 'Outcome'
print('Logistic regression on PIMA indians diabetes')
diabetes_data, diabetes_model, diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test, diabetes_y_predicted, diabetes_confusion_matrix = apply_logistic_regression(diabetes_filepath, diabetes_columns, diabetes_features, diabetes_to_predict, print_confusion_matrix=True)

draw_classes(diabetes_X_train, diabetes_y_train, diabetes_features, 'PIMA training set', graph_classes_color, diabetes_model)

# Work on Iris Plants
plants_filepath = basedir + '/Datasets/iris_proc.data'
plants_columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class']
# plants_features = ['Sepal width', 'Petal width']
plants_features = ['Sepal length', 'Petal length']
plants_to_predict = 'Class'
print('Logistic regression on iris plants')
plants_data, plants_model, plants_X_train, plants_X_test, plants_y_train, plants_y_test, plants_y_predicted, plants_confusion_matrix = apply_logistic_regression(plants_filepath, plants_columns, plants_features, plants_to_predict, print_confusion_matrix=True)

draw_classes_regions(plants_X_test, plants_y_predicted, plants_model, plants_features)