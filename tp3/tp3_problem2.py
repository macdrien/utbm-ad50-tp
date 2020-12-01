import numpy
import pandas
import matplotlib.pyplot as pyplot
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM

def apply_inverse_transform(predicted_stock_price, count_features, scaler):
    train_predicted_dataset_like = numpy.zeros( shape=(len(predicted_stock_price), count_features))
    train_predicted_dataset_like[:, 0] = predicted_stock_price[:, 0]
    predicted_stock_price = scaler.inverse_transform(train_predicted_dataset_like)[:, 0]
    return predicted_stock_price

def build_regressor(layers, input_shape, dropout=0.2, optimizer='adam', loss_function='mean_squared_error'):
    regressor = Sequential()
    count_layers = len(layers)

    regressor.add(LSTM(units=layers[0], return_sequences=True, input_shape=input_shape))
    regressor.add(Dropout(dropout))

    if 1 < count_layers:
        for layer_index in range(1, count_layers - 1):
            regressor.add(LSTM(units=layers[layer_index], return_sequences=True))
            regressor.add(Dropout(dropout))

    regressor.add(LSTM(units=layers[-1]))
    regressor.add(Dropout(dropout))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer=optimizer, loss=loss_function)

    return regressor

def build_series(inputs, count_observations):
    result = []

    for line in range(inputs.shape[1]):
        line_array = []

        for index in range(count_observations, len(inputs)):
            line_array.append(inputs[index - count_observations:index, line])
        line_array = numpy.array(line_array)
        result.append(line_array)

    result = numpy.swapaxes( numpy.swapaxes( numpy.array(result), 0, 1), 1, 2)
    return result

def get_subset(training_set_scaled, real_stock_price_scaled, count_observation):
    data = numpy.concatenate((training_set_scaled, real_stock_price_scaled), axis=0)
    return data[len(training_set_scaled)- count_observation:]

def load_dataset(input_file, count_test_example, features):
    data = pandas.read_csv(input_file)
    data = data.dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))

    training_set = data[:-count_test_example][features].values
    training_set_scaled = scaler.fit_transform(training_set)

    test_set = data[-count_test_example:][features].values
    test_set_scaled = scaler.transform(test_set)

    return training_set, training_set_scaled, scaler, test_set, test_set_scaled

def load_regressor(file_name='regressor'):
    loaded_model_json = None
    with open('{}.json'.format(file_name), 'r') as json_file:
        loaded_model_json = json_file.read()
    regressor = model_from_json(loaded_model_json)
    regressor.load_weights('{}.h5'.format(file_name))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    print('Loaded model from disk')
    return regressor

def plot_stock_prices(real_stock_price, predicted_stock_price):
    pyplot.plot(real_stock_price[:, 0], color='red', label='Real Stock Price')
    pyplot.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
    pyplot.title('Stock Price Reduction')
    pyplot.xlabel('Time')
    pyplot.ylabel('Stock Price')
    pyplot.legend()
    pyplot.show()
    
stock_file = '../Datasets/RNN/Google_Stock_Price.PA.csv'
count_observations = 60
count_test_exemple = 50
regressor_filename = None
features = ['Open', 'High', 'Low']
rnn_structure = (50, 50, 50, 50)
count_epochs = 10

training_set, training_set_scaled, scaler, real_stock_price, real_stock_price_scaled = load_dataset(stock_file, count_test_exemple, features)

X_train = build_series(training_set_scaled, count_observations)

id_price = 0
y_train = numpy.array(training_set_scaled[count_observations:, id_price])

regressor = None
if regressor_filename is None:
    regressor = build_regressor(rnn_structure, (X_train.shape[1], X_train.shape[2]))
    regressor.fit(X_train, y_train, epochs=count_epochs, batch_size=32)
else:
    regressor = load_regressor(regressor_filename)

inputs = get_subset(training_set_scaled, real_stock_price_scaled, count_observations)

X_test = build_series(inputs, count_observations)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = apply_inverse_transform(predicted_stock_price, X_train.shape[2], scaler)

plot_stock_prices(real_stock_price, predicted_stock_price)
