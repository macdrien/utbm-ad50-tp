import csv
import numpy
import pandas

prima_indians_diabetes_column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                                       'BodyMassIndex', 'DiabetesPedigreeFunction', 'Age', 'Outcome']


def load_csv_by_python_standard_library():
    filename = 'Datasets/pima-indians-diabetes.data'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    data = numpy.array(x).astype('float')
    print(data, data.shape)


def load_csv_by_with_numpy():
    filename = 'Datasets/pima-indians-diabetes.data'
    raw_data = open(filename, 'rt')
    data = numpy.loadtxt(raw_data, delimiter=',')
    print(data, data.shape)


def load_csv_with_pandas():
    filename = 'Datasets/pima-indians-diabetes.data'
    data = pandas.read_csv(filename, names=prima_indians_diabetes_column_names)
    print(data, data.shape)


load_csv_with_pandas()
