import numpy as np
import sklearn
from keras.layers import LSTM, Dense, Input, concatenate, Dropout
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from rand_walk import random_walk
from sklearn.model_selection import train_test_split

# Generate random walk time series data
num_steps = 1000
start = 0
time_series = random_walk(start, num_steps)

# Create additional features
rolling_mean = np.convolve(time_series, np.ones(10) / 10, mode='valid')
rolling_std = np.convolve(time_series, np.ones(10) / 10, mode='valid')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
time_series = scaler.fit_transform(time_series.reshape(-1, 1))
rolling_mean = scaler.fit_transform(rolling_mean.reshape(-1, 1))
rolling_std = scaler.fit_transform(rolling_std.reshape(-1, 1))

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(time_series, time_series, test_size=0.2, shuffle=False)

# Define the inputs
main_input = Input(shape=(1,), name='main_input')
aux_input = Input(shape=(1,), name='aux_input')
aux_input1 = Input(shape=(1,), name='aux_input1')

# Define the LSTM layer
lstm_layer = LSTM(64, return_sequences=True)(main_input)
lstm_layer = LSTM(64)(lstm_layer)

# Define the dense layer
dense_layer = Dense(32, activation='relu')(lstm_layer)
dense_layer = Dropout(0.2)(dense_layer)

# Concatenate the inputs
concatenated = concatenate([dense_layer, aux_input, aux_input1])

# Define the output layer
output = Dense(1, activation='linear')(concatenated)

# Define the model
model = Model(inputs=[main_input, aux_input, aux_input1], outputs=output)
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit([x_train, rolling_mean, rolling_std], y_train, epochs=100, batch_size=32, verbose=2)
