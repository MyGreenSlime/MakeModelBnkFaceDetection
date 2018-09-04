import sys
import numpy as np
import cv2
#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "CAN", "CHERPRANG","IZURINA","JAA",'JANE','JENNIS','JIB','KAEW','KAIMOOK','KATE','KORN','MAYSA','MIND','MIORI','MOBILE','MUSIC','NAMNUENG','NAMSAI','NINK',"NOEY","ORN","PIAM","PUN","PUPE","SATCHAN","TARWAAN"]
model = cv2.face.LBPHFaceRecognizer_create()
model.read('./model.xml')
print('finishloadmodel')
def predict(test_img):
    img = test_img.copy()
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    gray_img = cv2.GaussianBlur(img,(21,21),0) #v1 = 21,21
    gray_img = cv2.cvtColor(gray_img,cv2.COLOR_BGR2GRAY)
    facepre = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img,scaleFactor =1.05,minNeighbors = 5)
    labellist = []
    labeltextlist = []
    print(faces)
    for x,y,w,h in faces:
        label= model.predict(cv2.resize(facepre[y:y+w, x:x+h], (64,64), interpolation=cv2.INTER_CUBIC))
        resize = cv2.resize(facepre[y:y+w, x:x+h], (64,64), interpolation=cv2.INTER_CUBIC)
        flat = resize.flatten()
        #label = clf.predict(flat.reshape(1,-1))
        labellist.append(label)
        labeltextlist.append(subjects[label[0]])
    print(labeltextlist)
    count = 0
    for x,y,w,h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, labeltextlist[count], (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        count+=1
    return img
paths = sys.argv
check = True
for path in paths:
    if(check):
        check = False
        continue
    print("Predicting images...")
    test_img1 = cv2.imread(path)
    predicted_img1 = predict(test_img1)
    print("Prediction complete")
    resize = cv2.resize(predicted_img1, (int(test_img1.shape[1]/2),int(test_img1.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('test1', resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()