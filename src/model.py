from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Bidirectional
from sklearn.preprocessing import OneHotEncoder


def create_model(net_input, n_vocab):
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(512, return_sequences=True),
            input_shape=(net_input.shape[1], net_input.shape[2]),
        )
    )
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(n_vocab+1))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    

    return model

def train(net_input, net_output, model, epochs=95):
    model.fit(net_input,
            net_output,
            epochs=epochs,
            batch_size=64)

if __name__ == '__main__':