
# coding: utf-8

# In[2]:


import os,sys
import numpy as np
import cv2

# author: qzane@live.com
# reference: http://stackoverflow.com/a/23565051
# further reading: http://docs.opencv.org/master/da/d56/group__text__detect.html#gsc.tab=0
def text_detect(fno,img,ele_size=(6,2)): #
    if len(img.shape)==3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_sobel = cv2.Sobel(img,cv2.CV_8U,1,0)#same as default,None,3,1,0,cv2.BORDER_DEFAULT)
    img_threshold = cv2.threshold(img_sobel,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS ,ele_size)
    img_threshold = cv2.morphologyEx(img_threshold[1],cv2.MORPH_CLOSE,element)
    cv2.imshow('img',img_threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    _,contours,_ = cv2.findContours(img_threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #Rect = [cv2.boundingRect(i) for i in contours[1] if i.shape[0]>100]
    #RectP = [(int(i[0]-i[2]*0.08),int(i[1]-i[3]*0.08),int(i[0]+i[2]*1.1),int(i[1]+i[3]*1.1)) for i in Rect]
    print("no of contours",len(contours))
    i=0
    contours=sorted(contours,key=lambda ctr:cv2.boundingRect(ctr)[0])
    for cnt in contours:
        
        rect2=cv2.boundingRect(cnt)
        
        x,y,w,h=rect2
        roi = img[y:y+h+3, x:x+w] 
        #print(roi)
        
        cv2.rectangle(img,(x,y),( x + w, y + h ),(0,255,0),2)
        if w > 2 and h > 2: 
            i=i+1
            cv2.imshow('roi',roi)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()
            cv2.imwrite('C:/Users/hp/Desktop/projecttitle/text-detection-master/text-detection-master/output/wordfiles/'+str(fno)+'/{}.png'.format(i), roi)
    #print(rect)
    print(rect2)
    
    
    
    

def main(inputFile):
    
    fno=1
    for img in os.listdir(inputFile):
        
        img = cv2.imread(os.path.join(inputFile,img))
        #print(img)
        rect = text_detect(fno,img)
        fno=fno+1
    
if __name__ == '__main__':
    directory='C:/Users/hp/Desktop/projecttitle/text-detection-master/text-detection-master/output2/'
    main(directory)
   


