from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rand_walk import random_walk

# Generate random walk time series data
num_steps = 1000
start = 0
time_series = random_walk(start, num_steps)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(time_series, time_series, test_size=0.2, shuffle=False)

# Reshape the data for the LSTM layer
x_train = x_train.reshape(x_train.shape[0], 1, 1)
x_test = x_test.reshape(x_test.shape[0], 1, 1)

# Define the model
model = Sequential()
model.add(LSTM(4, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(x_train, y_train, epochs=60, batch_size=1, verbose=2)

# Make predictions
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)


def evaluate_model(model, x_train, y_train, x_test, y_test):
    # Make predictions on the training and test sets
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    # Calculate the mean squared error of the predictions
    train_mse = mean_squared_error(y_train, train_predict)
    test_mse = mean_squared_error(y_test, test_predict)

    # Calculate the mean absolute error of the predictions
    train_mae = mean_absolute_error(y_train, train_predict)
    test_mae = mean_absolute_error(y_test, test_predict)
    
    # Calculate the R2 score
    train_r2 = r2_score(y_train, train_predict)
    test_r2 = r2_score(y_test, test_predict)

    # Print the evaluation metrics
    print("Train MSE:", train_mse)
    print("Test MSE:", test_mse)
    print("Train MAE:", train_mae)
    print("Test MAE:", test_mae)
    print("Train R2:", train_r2)
    print("Test R2:", test_r2)

# Evaluate the model
evaluate_model(model, x_train, y_train, x_test, y_test)
