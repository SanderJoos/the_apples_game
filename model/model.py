from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D


class HarvestModel:

    def __init__(self):
        model = Sequential()
        model.add(Conv2D(121, 5, input_shape=(225, 1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(49, 5))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='sgd',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def predict(self, vector):
        return self.model(vector)


    def fit(self, vector, result):
        self.model.fit([vector], [result])



