
# coding: utf-8

# In[5]:
import cv2
import numpy as np
import os.path
def line_detect(uploaded_file_url):

## (1) read
	img = cv2.imread(uploaded_file_url)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imshow('gray',gray)
	## (2) threshold

	#threshed1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
			   # cv2.THRESH_BINARY,3,7)
	th, threshed = cv2.threshold(gray,100, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

	#cv2.imshow('threshed1',threshed1)
	cv2.imshow('threshold',threshed)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	## (3) minAreaRect on the nozeros
	pts = cv2.findNonZero(threshed)
	ret = cv2.minAreaRect(pts)

	(cx,cy), (w,h), ang = ret
	if w>h:
		w,h = h,w
		ang += 90

	## (4) Find rotated matrix, do rotation
	#M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
	#rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))

	## (5) find and draw the upper and lower boundary of each lines
	hist = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)#single column
	print(hist)
	print(len(hist))
	th = 10
	H,W = img.shape[:2]
	print(H)
	print(W)
	uppers = [y for y in range(7,H-1) if hist[y]<=th and hist[y+1]>th] #<2 and >2
	print('th=2',uppers)
	lowers = [y for y in range(7,H-1) if hist[y]>th and hist[y+1]<=th]
	print('th=2',lowers)

	   
	print('th=2',uppers)
	print('th=2',lowers)

	if not len(uppers)>2:
		if not len(lowers)>2:
			th=255
			uppers = [y for y in range(H-1) if hist[y]>=th and hist[y+1]<th] #<2 and >2 white text
			print('th=200',uppers)
			lowers = [y for y in range(H-1) if hist[y]<th and hist[y+1]>=th]
			print('th=200',lowers)
		
	if uppers[0]>lowers[0]:
		for i in range(len(lowers)-1):
			lowers[i]=lowers[i+1]
			
	print(uppers)
	print(lowers)
	rotated = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

	for y in uppers:
		cv2.line(rotated, (0,y), (W, y), (255,0,0), 1)
		
		
	for y in lowers:
		cv2.line(rotated, (0,y), (W, y), (0,255,0), 1)
		
	#lowers[i]:uppers[i],0:W
	j=0
	for i in range(len(uppers)):
		print(lowers[i])
		roi=rotated[uppers[i]-3:lowers[i]+2,0:W]
		a,b=roi.shape[:2]
		
		if a>5:
			print(a)
			j=j+1
			directory='C:/Users/hp/Desktop/projecttitle/text-detection-master/text-detection-master/output/wordfiles/'
			cv2.imwrite('C:/Users/hp/Desktop/projecttitle/text-detection-master/text-detection-master/output2/{}.png'.format(j), roi)
			if not os.path.exists(directory+'{}'.format(j)):
				os.mkdir(directory+'{}'.format(j))


	cv2.imwrite("result.png", rotated)
	cv2.imshow('rotated',rotated)

	cv2.waitKey(0)
	cv2.destroyAllWindows()




