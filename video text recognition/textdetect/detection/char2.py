
# coding: utf-8

# In[2]:


import cv2,os
import numpy as np

## (1) read
directory='C:/Users/hp/Desktop/projecttitle/text-detection-master/text-detection-master/output/wordfiles'
directory2='C:/Users/hp/Desktop/projecttitle/text-detection-master/text-detection-master/output/charfiles'
worddir=0
for folder in os.listdir(directory):
    worddir=worddir+1
    if not os.path.exists(directory2+'/'+str(worddir)):
        os.mkdir(directory2+'/'+str(worddir))

    fno=0
    print(folder)
    for files in os.listdir(os.path.join(directory,folder)):
        fno=fno+1
        if not os.path.exists(directory2+'/'+str(worddir)+'/{}'.format(fno)):
            os.mkdir(directory2+'/'+str(worddir)+'/{}'.format(fno))

        img = cv2.imread(os.path.join(directory,folder,files))
        print(folder,files)
        scale_percent = 220 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        
        
        
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        print('Resized Dimensions : ',img.shape,img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        cv2.imshow('thresh', thresh)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        im2, ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        j=0
        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv2.boundingRect(ctr)

            roi = img[y:y + h, x:x + w]

            area = w*h
            print(area)
            if 100 < area < 900:
                rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                j=j+1
                cv2.imwrite('C:/Users/hp/Desktop/projecttitle/text-detection-master/text-detection-master/output/charfiles/'+str(worddir)+'/'+str(fno)+'/{}.png'.format(j), roi)
                    #cv2.imshow('rect', rect)
                

