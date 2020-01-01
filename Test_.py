import CNNEncoderTrainer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from read_dataset import *
import numpy as np

"""
dataset = pd.read_csv('C:\\Users\\zkx74\\PycharmProjects\\data.csv')
dataset = dataset.drop(['Quality_label','Unnamed: 0'],axis=1)
data = np.array(dataset)
data = np.tanh(data)
data = StandardScaler().fit_transform(data)
encoder = EncoderTrainer.AutoEncoder(input_shape=(data.shape[1],), first_output_shape=(10,), second_output_shape=(2,))
encoder.train(data, epoch=1000, batch_size=256)
weights1, weights2 = encoder.get_feature()
"""
# x_train, y_train, x_test, y_test = read_mnist_subset()
# print(x_train.shape)
# # x_train = np.random.rand(4000,28,28,1)
# encoder = CNNEncoderTrainer.AutoEncoder(input_shape=(28, 28, 1), label_shape=(10,))
# # x = np.ones((1, 28, 28, 1))
# # y = encoder.auto_encoder.predict(x)
# encoder.train([x_train,y_train], epoch=10, batch_size=32)
# # print(y)
# a = 1

def main():
    encoder = CNNEncoderTrainer.AutoEncoder(input_shape=(28, 28, 1), label_shape=(10,))
    encoder.set_model()
    x_train, y_train, x_test, y_test = read_mnist_subset()
    # y = encoder.predict([x_test, y_test])
    y = encoder.get_feature()
    print(y)
    print(y.shape)

if __name__ == '__main__':
    main()
