import numpy as np
import scipy.io as sio
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from read_dataset import *

def main():
    cnn_process_subset()

def nn_process():
    x_train, y_train, x_test, y_test = read_svhn_data()
    # x_train, y_train, x_test, y_test = read_subset()

    # data pre-processing
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    model = Sequential([
        Dense(32, input_dim=32*32*3, activation='relu'),
        Dense(10, activation='softmax')
    ])
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(
        optimizer=rmsprop,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    print('Training ------------')
    # Another way to train the model
    # model.fit(x_train, y_train, epochs=2, batch_size=2)
    model.fit(x_train, y_train, epochs=2, batch_size=32)

    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(x_test, y_test)

    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

def cnn_process_subset():
    x_train, y_train, x_test, y_test = read_svhn_subset()

    ''' Construct model '''
    c1_channel = 16
    c1_kernel = 1
    c1_size2 = 2 # ？？？
    c1_size3 = 2 # ？？？
    c2_channel = 16
    c2_kernel = 1
    c2_size2 = 2 # ？？？
    c2_size3 = 2 # ？？？
    p1_type = 'max' # Pooling Type (max / avg)
    p1_kernel = 1 # kernel size
    p1_stride = 2 # stride size
    p2_type = 'max'
    p2_kernel = 1
    p2_stride = 2
    n1 = 16 # hidden layer size
    n2 = 16
    n3 = 16
    n4 = 16
    learn_rate = 0.0002

    model = Sequential()
    # input: 28x28 images with 1 channels -> (28, 28, 1) tensors.
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu'))
    model.add(BatchNormalization())
    if p1_type == 'max':
        model.add(MaxPooling2D(pool_size=p1_kernel, strides=p1_stride))
    elif p1_type == 'avg':
        model.add(AveragePooling2D(pool_size=p1_kernel, strides=p1_stride))

    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu'))
    # model.add(BatchNormalization())
    if p2_type == 'max':
        model.add(MaxPooling2D(pool_size=p2_kernel, strides=p2_stride))
    elif p2_type == 'avg':
        model.add(AveragePooling2D(pool_size=p2_kernel, strides=p2_stride))

    model.add(Flatten())
    model.add(Dense(n1, activation='relu'))
    model.add(Dense(n2, activation='relu'))
    model.add(Dense(n3, activation='relu'))
    model.add(Dense(n4, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    adam = Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    ''' Training '''
    print('Training ------------')
    model.fit(x_train, y_train, epochs=1, batch_size=100)

    print('\nTesting ------------')
    loss, accuracy = model.evaluate(x_test, y_test)

    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

def cnn_process():
    x_train, y_train, x_test, y_test = read_svhn_data()
    # x_train, y_train, x_test, y_test = read_subset()

    ''' Construct model '''
    c1_channel = 16
    c1_kernel = 3
    c1_size2 = 2 # ？？？
    c1_size3 = 2 # ？？？
    c2_channel = 28
    c2_kernel = 1
    c2_size2 = 2 # ？？？
    c2_size3 = 2 # ？？？
    p1_type = 'max' # Pooling Type (max / avg)
    p1_kernel = 1 # kernel size
    p1_stride = 2 # stride size
    p2_type = 'avg'
    p2_kernel = 1
    p2_stride = 2
    n1 = 48 # hidden layer size
    n2 = 36
    n3 = 24
    n4 = 16
    learn_rate = 0.0002

    model = Sequential()
    # input: 28x28 images with 1 channels -> (28, 28, 1) tensors.
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', input_shape=(32, 32, 3)))
    # model.add(BatchNormalization())
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu'))
    # model.add(BatchNormalization())
    if p1_type == 'max':
        model.add(MaxPooling2D(pool_size=p1_kernel, strides=p1_stride))
    elif p1_type == 'avg':
        model.add(AveragePooling2D(pool_size=p1_kernel, strides=p1_stride))

    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu'))
    # model.add(BatchNormalization())
    if p2_type == 'max':
        model.add(MaxPooling2D(pool_size=p2_kernel, strides=p2_stride))
    elif p2_type == 'avg':
        model.add(AveragePooling2D(pool_size=p2_kernel, strides=p2_stride))

    model.add(Flatten())
    model.add(Dense(n1, activation='relu'))
    model.add(Dense(n2, activation='relu'))
    model.add(Dense(n3, activation='relu'))
    model.add(Dense(n4, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    adam = Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    ''' Training '''
    print('Training ------------')
    # Another way to train the model
    model.fit(x_train, y_train, epochs=1, batch_size=32)

    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    print(model.evaluate(x_test, y_test))
    loss, accuracy = model.evaluate(x_test, y_test)

    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

if __name__ == '__main__':
    main()