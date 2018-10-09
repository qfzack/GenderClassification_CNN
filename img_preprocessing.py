import os
import numpy as np
import cv2 as cv
from keras.utils import np_utils

a = cv.imread('./gender/female/female1.jpg')
count_female = len(os.listdir('./gender/female/'))
train1 = np.zeros(shape=(count_female,a.shape[0],a.shape[1],a.shape[2]))
k = 0
for i in range(1,count_female+1):
    fold1 = './gender/'+'female/'+'female'+str(i)+'.jpg'
    train1[k] = cv.imread(fold1)
    k = k+1

b = cv.imread('./gender/male/male1.jpg')
count_male = len(os.listdir('./gender/male/'))
train2 = np.zeros(shape=(count_male,b.shape[0],b.shape[1],b.shape[2]))
j = 0
for i in range(1,count_male+1):
    fold2 = './gender/'+'male/'+'male'+str(i)+'.jpg'
    train2[j] = cv.imread(fold2)
    j = j+1

#汇总得到训练数据集和训练数据的标签
train = np.vstack((train1, train2))
train /= 255
train = train.astype('float32')
train1_label = np.zeros(count_female)
train2_label = np.ones(count_male)
train1_label = np_utils.to_categorical(train1_label, 2)
train2_label = np_utils.to_categorical(train2_label, 2)
train_label = np.vstack((train1_label,train2_label))

c = cv.imread('./gender/female_validation/female1.jpg')
count_female_validation = len(os.listdir('./gender/female_validation/'))
test1 = np.zeros(shape=(count_female_validation,c.shape[0],c.shape[1],c.shape[2]))
l = 0
for i in range(1,count_female_validation+1):
    fold3 = './gender/'+'female_validation/'+'female'+str(i)+'.jpg'
    test1[l] = cv.imread(fold3)
    l = l+1

d = cv.imread('./gender/male_validation/male1.jpg')
count_male_validation = len(os.listdir('./gender/male_validation/'))
test2 = np.zeros(shape=(count_male_validation,d.shape[0],d.shape[1],d.shape[2]))
m = 0
for i in range(1,count_male_validation+1):
    fold4 = './gender/'+'male_validation/'+'male'+str(i)+'.jpg'
    test2[m] = cv.imread(fold4)
    m = m+1

#汇总得到检验数据集和检验数据的标签
test = np.vstack((test1,test2))
test /= 255
test = test.astype('float32')
test1_label = np.zeros(count_female_validation)
test2_label = np.ones(count_male_validation)
test1_label = np_utils.to_categorical(test1_label, 2)
test2_label = np_utils.to_categorical(test2_label, 2)
test_label = np.vstack((test1_label,test2_label))

#保存训练数据和监测数据
np.save('cnn_gender_traindata.npy',train)
np.save('cnn_gender_traindata_label.npy',train_label)
np.save('cnn_gender_testdata.npy',test)
np.save('cnn_gender_testdata_label.npy',test_label)