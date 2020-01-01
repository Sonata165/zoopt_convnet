import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.optimizers import Adam

from zoopt import Dimension, Objective, Parameter, Opt
from read_dataset import *
import matplotlib.pyplot as plt

dataset = None

def main():
    global dataset
    # dataset = read_mnist_data()
    dataset = read_svhn_subset()
    dim = Dimension(
        19,
        [[16, 32], [1, 8], [1, 1], [1, 1], [16, 32],
         [1, 8], [1, 1], [1, 1], [0, 1], [1, 8],
         [1, 10], [0, 1], [1, 8], [1, 10], [40, 50],
         [30, 40], [20, 30], [10, 20], [0.0001, 0.001]],
        [False, False, False, False, False,
         False, False, False, False, False,
         False, False, False, False, False,
         False, False, False, True]
    )
    # 设定优化目标
    obj = Objective(eval, dim)
    # perform optimization
    solution = Opt.min(obj, Parameter(budget=10))
    # print result
    solution.print_solution()

    plt.plot(obj.get_history_bestsofar())
    plt.savefig('figure.png')

def eval(solution):
    '''
    要优化的函数！
    :param solution:
    :return:
    '''
    x = solution.get_x()
    print(x)
    value = evaluate_param(dataset, x)
    return value[0]

def evaluate_param(dataset, params):
    '''
    评估一组超参数（19个）在指定数据集上运行CNN的表现
    :param params: 参数列表，要求len == 19
    :param dataset: 指定的数据集
    :return: 评估指标，这里是(loss, 正确率)
    '''
    assert len(params) == 19

    x_train, y_train, x_test, y_test = dataset

    c1_channel = params[0]
    c1_kernel = params[1]
    c1_size2 = params[2]  # ？？？
    c1_size3 = params[3]  # ？？？
    c2_channel = params[4]
    c2_kernel = params[5]
    c2_size2 = params[6]  # ？？？
    c2_size3 = params[7]  # ？？？
    p1_type = params[8]  # Pooling Type (max / avg)
    p1_kernel = params[9]  # kernel size
    p1_stride = params[10]  # stride size
    p2_type = params[11]
    p2_kernel = params[12]
    p2_stride = params[13]
    n1 = params[14]  # hidden layer size
    n2 = params[15]
    n3 = params[16]
    n4 = params[17]
    learn_rate = params[18]

    # 取值范围检查
    assert isinstance(c1_channel, int) and c1_channel >= 1
    assert isinstance(c1_kernel, int) and c1_kernel >= 1 and c1_kernel <= 28
    # assert isinstance(c1_size2, int) and c1_size2 >= 1
    # assert isinstance(c1_size3, int) and c1_size3 >= 1
    assert isinstance(c2_channel, int) and c2_channel >= 1
    assert isinstance(c2_kernel, int) and c2_kernel >= 1 and c2_kernel <= 28
    # assert isinstance(c2_size2, int) and c2_size2 >= 1
    # assert isinstance(c2_size3, int) and c2_size3 >= 1
    # 注：0是max，1是avg
    assert isinstance(p1_type, int) and (p1_type == 0 or p1_type == 1)
    assert isinstance(p1_kernel, int) and p1_kernel >= 1 and p1_kernel <= 28
    assert isinstance(p1_stride, int) and p1_stride >= 1
    assert isinstance(p2_type, int) and (p2_type == 0 or p2_type == 1)
    assert isinstance(p2_kernel, int) and p2_kernel >= 1 and p2_kernel <= 28
    assert isinstance(p2_stride, int) and p2_stride >= 1
    assert isinstance(n1, int) and n1 >= 1
    assert isinstance(n2, int) and n2 >= 1
    assert isinstance(n3, int) and n3 >= 1
    assert isinstance(n4, int) and n4 >= 1
    assert isinstance(learn_rate, float) and learn_rate > 0 and learn_rate < 1

    ''' 建立模型 '''

    model = Sequential()
    # input: 28x28 images with 1 channels -> (28, 28, 1) tensors.
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', input_shape=x_train[0].shape, padding='same'))
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', padding='same'))
    if p1_type == 0:
        model.add(MaxPooling2D(pool_size=p1_kernel, strides=p1_stride, padding='same'))
    elif p1_type == 1:
        model.add(AveragePooling2D(pool_size=p1_kernel, strides=p1_stride, padding='same'))

    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    if p2_type == 0:
        model.add(MaxPooling2D(pool_size=p2_kernel, strides=p2_stride, padding='same'))
    elif p2_type == 1:
        model.add(AveragePooling2D(pool_size=p2_kernel, strides=p2_stride, padding='same'))

    model.add(Flatten())
    model.add(Dense(n1, activation='relu'))
    model.add(Dense(n2, activation='relu'))
    model.add(Dense(n3, activation='relu'))
    model.add(Dense(n4, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    adam = Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    ''' 训练和测试 '''

    print('Training ------------')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)

    print('\nTesting ------------')
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

    print('test loss: ', loss)
    print('test accuracy: ', accuracy)
    return (loss, accuracy)

def search(_dataset):
    '''
    在指定数据集上搜索最优超参数
    :param _dataset: 指定的数据集
    :return: (最优超参数，最优超参数的表现)
    '''
    global dataset
    dataset = _dataset
    dim = Dimension(
        19,
        [[16, 32], [1, 8], [1, 1], [1, 1], [16, 32],
         [1, 8], [1, 1], [1, 1], [0, 1], [1, 8],
         [1, 10], [0, 1], [1, 8], [1, 10], [40, 50],
         [30, 40], [20, 30], [10, 20], [0.0001, 0.001]],
        [False, False, False, False, False,
         False, False, False, False, False,
         False, False, False, False, False,
         False, False, False, True]
    )
    obj = Objective(eval, dim)
    # perform optimization
    solution = Opt.min(obj, Parameter(budget=100))
    # print result
    solution.print_solution()

    plt.plot(obj.get_history_bestsofar())
    plt.savefig('figure.png')
    return (solution.get_x(), solution.get_value())

def search1(_dataset):
    return ([1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4], 0.2)

if __name__ == '__main__':
    main()