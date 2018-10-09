# GenderClassification_CNN
对于获取的已标记性别的人脸图片，将其分为训练集和测试集，建立CNN分类器（Keras）并用训练集训练；                  
对于任意一张包含人脸的图片，使用OpenCV识别人脸位置，并将其裁剪下来，再用已训练好的CNN分类器进行分类。                    
