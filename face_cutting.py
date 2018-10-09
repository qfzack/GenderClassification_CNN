import cv2

filepath = 'gender/7.jpg'

img = cv2.imread(filepath)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
color = (0,255,0)

faceRects = classifier.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))

#裁剪人脸
x,y,w,h = faceRects[0]
cut = img[y+h//2-91:y+h//2+91,x+w//2-91:x+w//2+91]
cv2.imwrite('gender/cutting.jpg',cut)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',cut)
cv2.waitKey(0)
cv2.destroyAllWindows()


#人脸检测
# if len(faceRects):
#     for faceRect in faceRects:
#         x,y,w,h = faceRect
#         cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()










