import scipy.io as sio
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

def read_mnist_data():
    '''
    读入并预处理mnist数据集
    :return: (x_train, y_train, x_test, y_test)
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train / 255.).reshape([60000, 28, 28, 1])  # normalize
    x_test = (x_test / 255.).reshape([10000, 28, 28, 1])  # normalize
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    return (x_train, y_train, x_test, y_test)

def read_mnist_subset():
    mat = sio.loadmat('mnist_subset0.mat')
    x = mat['X']
    y = mat['y']
    y = y.reshape(y.shape[1])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # x_train = x_train / 255
    # x_test = x_test / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return (x_train, y_train, x_test, y_test)

def read_svhn_data():
    mat1 = sio.loadmat('../12.27_dataset/train_32x32.mat')
    X1 = mat1['X']
    x_train = []
    for i in range(X1.shape[3]):
        x_train.append(X1[:,:,:,i])
    x_train = np.array(x_train)
    Y1 = mat1['y']
    for i in range(len(Y1)):
        if Y1[i] == 10:
            Y1[i] = 0
    y_train = np_utils.to_categorical(Y1, num_classes=10)

    mat2 = sio.loadmat('../12.27_dataset/test_32x32.mat')
    X2 = mat2['X']
    x_test = []
    for i in range(X2.shape[3]):
        x_test.append(X2[:,:,:,i])
    x_test = np.array(x_test)
    Y2 = mat2['y']
    for i in range(len(Y2)):
        if Y2[i] == 10:
            Y2[i] = 0
    y_test = np_utils.to_categorical(Y2, num_classes=10)

    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train, x_test, y_test)

def read_svhn_subset():
    mat = sio.loadmat('svhn_subset0.mat')
    x = mat['X']
    y = mat['y']
    y = y.reshape(y.shape[1])
    print(x.shape)
    print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = x_train / 255
    x_test = x_test / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print('读入svhn子集成功！')
    return (x_train, y_train, x_test, y_test)
