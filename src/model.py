from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
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
            batch_input_shape=(100, net_input.shape[1], net_input.shape[2]),
        )
    )
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(388))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    return model

def train(net_input, net_output, model, filepath='data/weights.{epoch:02d}-{val_loss:.2f}.hdf5' epochs=2):
    
    checkpoint = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min')    
callbacks_list = [checkpoint]
    
    model.fit(net_input,
            net_output,
            epochs=epochs)
    model.save('data/model.h5')


if __name__ == '__main__':
    model = create_model(X, n_vocab=len(X))
    train(X, y, model)


    