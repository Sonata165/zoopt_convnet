import os
import pickle as pk
import scipy.io as sio
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from zoopt_test import search

def main():
    '''
    读入./datasets/subset/下所有数据集，用zoopt计算最优参数，
    算好后保存在./cnn_label.csv里，格式如下：
                param1  param2  ...     param19
    dataset1    a1      b1              s1
    dataset2    a2      b2              s2
        ...
    '''
    DATASET_PATH = '../12.27_dataset/subset/'
    files = os.listdir(DATASET_PATH)
    res = {}
    for file in files:
        dataset = read_dataset(DATASET_PATH + file)
        RESULT_PATH = '../12.27_dataset/result/'
        reslts = os.listdir(RESULT_PATH)
        this_name = file + '.pkl'
        if this_name in reslts:
            continue
        # 否则，寻找最优参数
        param, result = search(dataset)
        res[file] = param
        res[file].append(result)

        # 保存到pkl文件
        f = open('../12.27_dataset/result/' + file + '.pkl', 'wb')
        pk.dump(param, f)
        pk.dump(result, f)
        f.close()
    # datasets = read_datasets()
    # res = {}
    # for name in datasets:
    #     print(name)
    #     # 如果已经算过，那么跳过
    #     RESULT_PATH = '../12.27_dataset/result/'
    #     files = os.listdir(RESULT_PATH)
    #     this_name = name + '.pkl'
    #     if this_name in files:
    #         continue
    #
    #     # 否则，寻找最优参数
    #     param, result = search(datasets[name])
    #     res[name] = param
    #     res[name].append(result)
    #
    #     # 保存到pkl文件
    #     f = open('../12.27_dataset/result/' + name + '.pkl', 'wb')
    #     pk.dump(param, f)
    #     pk.dump(result, f)
    #     f.close()
    # column = []
    # for i in range(1, 20):
    #     column.append('param' + str(i))
    # column.append('res')
    # df = pd.DataFrame(res).transpose()
    # df.columns = column
    # df.to_csv('cnn_label.csv')

def read_datasets():
    '''
    该函数读取datasets下所有数据集，
    Parameters:
      None - None
    Returns:
      一个字典，包含所有读入的数据集，格式如 数据集名称:数据集内容
      数据集类型为'.mat'
    '''
    print('读取数据集')
    INPUTPATH = '../12.27_dataset/subset/'
    files = os.listdir(INPUTPATH)
    datasets = {}
    for file in files:
        dataset = sio.loadmat(INPUTPATH + file)
        x = dataset['X']
        y = dataset['y']
        y = y.reshape(y.shape[1])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        datasets[file] = (x_train, y_train, x_test, y_test)
    print('读取完成！')
    return datasets

def read_dataset(path):
    '''
    读入一个mat格式的数据集
    :param path: 数据集路径
    :return: (x_train, y_train, x_test, y_test)
    '''
    dataset = sio.loadmat(path)
    x = dataset['X']
    y = dataset['y']
    y = y.reshape(y.shape[1])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return (x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()