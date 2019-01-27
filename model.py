import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM, Dense, Activation, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from itertools import islice


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = list(islice(it, n))
    if len(result) == n:
        yield np.array(result)
    for elem in it:
        result = result[1:] + [elem]
        yield np.array(result)

def generate_data(txt='mobydick.txt', sliding_win=10):
    with open(txt, 'r') as myfile:
        data = myfile.read()
        tokenizer = Tokenizer(filters=None, lower=False, char_level=True)
        tokenizer.fit_on_texts(data);
        # dump the word-index correspondance
        np.save("w", tokenizer.index_word)
        seq = tokenizer.texts_to_sequences(data)
        # flatten
        seq = [i for sub in seq for i in sub]
        # prepend data with oov_token (might use 0)
        oov = max(tokenizer.word_index.values()) + 1
        seq = (sliding_win-1)*[oov]+seq
        # labels
        y = seq[sliding_win:]  
        X = window(seq, sliding_win)
        X = np.array(list(X)[:-1])
        X = X.reshape((*X.shape, 1))
        return X, y

def create_model(n_outputs, blocks=2, timesteps=10):
    model = Sequential()
    model.add(LSTM(16, return_sequences=True, input_shape=(timesteps, 1)))
    for _ in range(blocks-1):
        model.add(LSTM(16, return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(n_outputs))
    model.add(Activation('softmax'))
    return model

def main():
    X, y = generate_data("mobydick.txt", sliding_win=20)
    model = create_model(max(y)+1, timesteps=20)
    log = CSVLogger('log.csv')
    ckpt = ModelCheckpoint('best.h5',
                            monitor='acc',
                            verbose=0,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='auto',
                            period=1)
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    model.fit(X, y, epochs=12, batch_size=256, callbacks=[log, ckpt])
    model.save('best.h5')



if __name__ == "__main__":
    main()
