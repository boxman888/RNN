import numpy as np
import math
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import load_model

# Used to limit accurecy for those trying to repoduce our results.
np.random.seed(7)

def read_and_pad(filename):
    """ Reads in rna and pads it."""
    rna_samples = []
    stage = None
    with open(filename) as file_p:
        for i, line in enumerate(file_p):
            if i % 2 != 0:

                sequence = [ord(char) for char in line]
                rna_samples.append([stage.strip(">"), np.asarray(sequence)])
            else:
                stage = line

    return np.asarray(rna_samples)

def RNN(filename, samples):
    vector_length = 32
    model = Sequential()
    model.add(Embedding(5000, vector_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.load_weights(filename)
    yhat = model.predict(samples[2])
    print(yhat)


if __name__ == "__main__":
    PKS = "sequence_test.fasta"

    PKS_RNA_SAMPLES = read_and_pad(PKS)
    print(PKS_RNA_SAMPLES)
    RNN('weights.h5', PKS_RNA_SAMPLES[:, [1]])
   
    
