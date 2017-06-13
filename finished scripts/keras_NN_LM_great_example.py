import keras.models
import keras.layers.core
import keras.layers.wrappers
import keras.layers.recurrent
import keras.layers.embeddings
import keras.utils.np_utils

import numpy as np


def main():

    input = [[0, 1], [1, 2], [2, 3]]
    output = [[1, 2], [2, 3], [3, 4]]

    output = np.expand_dims(output, -1)


    model = keras.models.Sequential()

    model.add(keras.layers.embeddings.Embedding(input_dim=5, output_dim=16))
    model.add(keras.layers.recurrent.LSTM(units=32, return_sequences=True))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.Dense(units=5, activation="softmax")))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.fit(input, output, batch_size=1, epochs=10)

    test = np.zeros((1, 2), dtype=np.int16)
    test[0][0] = 1
    test[0][1] = 2

    prediction = model.predict(test, verbose=0)[0]
    print prediction


if __name__ == '__main__':
    main()