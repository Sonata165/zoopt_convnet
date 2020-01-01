import keras
from keras.layers import *
from keras.models import *
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import *

class AutoEncoder:
    def __init__(self, input_shape, label_shape, auto_encoder=None, encoder=None):
        '''
        双层自编码机，单独训练
        input_size->first_output_shape->second_output_shape
        数据需要归一化至(-1,1)区间
        :param input_shape: 形如(width,height,channel)
        :param feature_shape: 形如(width,height,channel)
        :param label_shape: (label_size,)
        '''
        self.auto_encoder = auto_encoder
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.encoder = encoder
        self.build_encoder_decoder()

    def set_model(self):
        '''
        从本地文件读取预训练的模型
        :return: None
        '''
        # self.encoder = load_model('encoder.h5')
        self.auto_encoder = load_model('auto_encoder.h5')

    def build_encoder_decoder(self):
        '''
        建立编码机模型
        :return: 编码器编码器+解码器（用于训练）
        '''
        # 第一个编码机
        input_data = Input(self.input_shape)
        input_label = Input(self.label_shape)
        x = Conv2D(16, (3, 3), activation='relu', padding='same', name='encoder1')(input_data)
        x = MaxPooling2D((2, 2), padding='same', name='encoder2')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same', name='encoder3')(x)
        x = MaxPooling2D((2, 2), padding='same', name='encoder4')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same', name='encoder5')(x)
        x = MaxPooling2D((2, 2), padding='same', name='encoder6')(x)
        # 保存展平前的shape
        x_shape = K.int_shape(x)
        # 展平
        x = Flatten()(x)
        x_size = K.int_shape(x)[1]
        label = Dense(self.label_shape[0], name='encoder7')(input_label)
        # 引入标签至末尾
        x_with_label = Concatenate()([x, label])
        # 统一编码
        encoded = Dense(x_size, activation='relu', name='encoder8')(x_with_label)

        # 统一解码
        x_with_label = Dense(x_size, activation='relu')(encoded)
        # 从结果中提取label
        label = Dense(self.label_shape[0])(x_with_label)
        # 从结果中重建高维x（展平逆操作）
        x = Reshape((x_shape[1], x_shape[2], x_shape[3]))(x_with_label)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same',name='decoded')(x)

        self.auto_encoder = Model(inputs=[input_data,input_label], outputs=[decoded,label])
        self.auto_encoder.compile(optimizer=keras.optimizers.Adam(0.001), loss=keras.losses.mse)
        self.auto_encoder.summary()
        #visualize_activation(self.auto_encoder,utils.find_layer_idx(self.auto_encoder,'decoded'),filter_indices=0,input_range=(0,1))
        return self.auto_encoder

    def predict(self, input_data):
        '''
        对数据进行编码
        :param input_data: 数据
        :return: 编码结果
        '''
        return self.encoder.predict(input_data)

    def train(self, x, epoch, batch_size):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=50, min_lr=0.0001)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        check_point = keras.callbacks.ModelCheckpoint(filepath='./auto_encoder_checkPoint.h5', monitor='val_loss',
                                                      save_best_only=True, verbose=1)
        tensor_board = keras.callbacks.TensorBoard(log_dir='./auto_encoder_tensor_board_logs', write_grads=True,
                                                   write_graph=True,
                                                   write_images=True)
        self.auto_encoder.fit(x, x, epochs=epoch, batch_size=batch_size, validation_split=0.1,
                              callbacks=[reduce_lr, early_stop, check_point])
        self.auto_encoauder.save('to_encoder.h5')
        self.encoder = Model(inputs=self.auto_encoder.input, outputs=self.auto_encoder.get_layer('encoder8').output)
        self.encoder.save('encoder.h5')

    def get_feature(self):
        '''
        返回encoder的权重作为对数据集提取的特征
        :return: encoder1权重,encoder2权重
                类型为list(numpy数组)
                例如encoder1的参数为
                [编码器中每层的get_weights结果]
        '''
        res = []
        st = 'encoder'
        # 生成编码器每层的名字并提取权重
        for i in range(1, 9):
            res.append(self.auto_encoder.get_layer(st + str(i)).get_weights())
        return res
