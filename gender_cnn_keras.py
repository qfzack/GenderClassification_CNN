import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(7)
batch_size = 50
nb_classes = 2
epochs = 18

img_rows,img_cols = 182,182
nb_filters = 32
pool_size = (2,2)
kernel_size = (7,7)

# (x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = np.load('cnn_gender_traindata.npy')
y_train = np.load('cnn_gender_traindata_label.npy')
x_test = np.load('cnn_gender_testdata.npy')
y_test = np.load('cnn_gender_testdata_label.npy')
input_shape = (img_rows,img_cols,3)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#将测试数据转换成one_hot类型
# y_train = np_utils.to_categorical(y_train, nb_classes)
# y_test = np_utils.to_categorical(y_test, nb_classes)

####################################################################
model = Sequential()

#添加卷积层
model.add(Conv2D(48,(kernel_size[0],kernel_size[1]),
                 padding='same',
                 activation='relu',
                 input_shape=input_shape))

#添加卷积层
model.add(Conv2D(48,(5,5),strides=(1,1),
                 activation='relu'))

#添加池化层
model.add(MaxPooling2D(pool_size=(3,3)))

#添加dropout层，神经元随机失活，避免过拟合
model.add(Dropout(0.5))

#拉成一维数据
model.add(Flatten())

#添加全连接层
model.add(Dense(128,activation='relu'))

#添加dropout层
model.add(Dropout(0.5))

#添加全连接层
model.add(Dense(2))

model.add(Activation('softmax'))
####################################################################

#编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#训练模型
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,
          verbose=1,validation_data=(x_train,y_train))

score = model.evaluate(x_test,y_test,verbose=0)#评估模型


print('test score:',score[0])
print('test accuracy:',score[1])

#保存训练的模型
model.save('CNN_gender_keras.h5')