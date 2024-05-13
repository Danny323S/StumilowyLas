import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np 
import keras

def create_model():
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    return model


def main():
    pass

if __name__ == '__main__':
    main()  