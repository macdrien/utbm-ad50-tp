import csv
import numpy
import pandas
import os

basedir = os.path.abspath(os.path.dirname(__file__)) + '/'


def load_pima_indians():
    filename = basedir + 'Datasets/pima-indians-diabetes.data'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    data = numpy.array(x).astype('float')
    print('PIMA Indians')
    print(data, data.shape)


def load_iris():
    filename = basedir + 'Datasets/iris_proc.data'
    raw_data = open(filename, 'rt')
    data = numpy.loadtxt(raw_data, delimiter=',')
    print('Iris Plant')
    print(data, data.shape)


def load_churn_modelling():
    filename = basedir + 'Datasets/Churn_Modelling.csv'
    names = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
    data = pandas.read_csv(filename)
    print('Churn Modelling')
    print(data, data.shape)


load_pima_indians()
load_iris()
load_churn_modelling()
