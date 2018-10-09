import numpy as np
from keras.models import load_model
import matplotlib.image as processimage

#载入训练好的cnn模型
model = load_model('CNN_gender_keras.h5')

#载入需要预测的数据
pred_img = processimage.imread('gender/cutting.jpg')
pred_img = np.array(pred_img)/225
pred_img = pred_img.reshape(-1,182,182,3)
prediction = model.predict(pred_img)
final_prediction = [result.argmax() for result in prediction][0]
a = 0
for i in prediction[0]:
    print('Percent:{:.10%}'.format(i))
    a = a+1
if final_prediction==0:
    print('预测结果为：女性')
if final_prediction==1:
    print('预测结果为：男性')