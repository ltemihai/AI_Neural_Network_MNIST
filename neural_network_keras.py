from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adadelta

def generate_model(input, output):
    model = Sequential()
    model.add(Dense(128, input_dim=input.shape[1], activation='sigmoid'))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    return model

def view_summary(model):
    model.summary()

def train_model(model, input, output):
    model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(input, output, epochs=1, batch_size=64)
