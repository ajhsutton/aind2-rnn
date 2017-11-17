import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    num_windows = len(series) - window_size    
    # containers for input/output pairs
    X = [series[win:win+window_size] for win in range(num_windows)]
    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
# Make the Keras Sequential RNN
    model = Sequential()
    model.add(LSTM(5,input_shape = (window_size,1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    from string import ascii_lowercase
    
    text_enum = set(text)
    for text_char in text_enum:
        if (text_char not in punctuation) and (text_char not in ascii_lowercase):
            text = text.replace(text_char,' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    num_windows = (len(text) - window_size)//step_size
    inputs = []
    outputs = [] 
    for iwin in range(num_windows):
        win_idx = (iwin - 1)*step_size
        inputs.append(text[win_idx:win_idx+window_size])
        outputs.append(text[win_idx+window_size])
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
