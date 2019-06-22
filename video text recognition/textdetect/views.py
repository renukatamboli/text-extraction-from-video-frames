from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
import shutil
import cv2,os
import numpy as np
import os.path
from ffpyplayer.player import MediaPlayer
from tensorflow.keras.models import Sequential, load_model


d=r'C:/Users/Admin/Desktop/image processing/test3'
directory=str(d)+'/media/output/wordfiles/'
waitTime = 40
rect = (0,0,0,0)
j = 0
refPt = []
cropping = 1

def click_and_crop(event, x, y, flags, param):
	global refPt, cropping

	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = 2

	elif event == cv2.EVENT_LBUTTONUP:

		refPt.append((x, y))
		cropping = 3

def flick(x):
    pass

def process(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def video(request):
	if(os.path.exists(str(d) + '/media/output/screenshots/')):
		shutil.rmtree(str(d) + '/media/output/screenshots/')
	if not os.path.exists(str(d) + '/media/output/screenshots/'):
		os.mkdir(str(d) + '/media/output/screenshots/')

	global cropping, refPt, rect, waitTime,j
	refPt=[]
	if 'y_submitted' in request.POST :
		import pafy, youtube_dl
		url2 = request.POST.get('youtubeurl')
		vPafy = pafy.new(url2)
		play = vPafy.getbest(preftype="mp4")

		cv2.namedWindow('Video')
		cv2.moveWindow('Video', 250, 150)

		vidcap = cv2.VideoCapture(play.url)

		tots = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
		i = 0
		cv2.createTrackbar('S', 'Video', 0, int(tots) - 1, flick)
		cv2.setTrackbarPos('S', 'Video', 0)
		status = 'stay'
		player=MediaPlayer(str(play.url))
		while (vidcap.isOpened()):

			try:
				if i == tots - 1:
					i = 0
				vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
				success, frame = vidcap.read()
				r = 750.0 / frame.shape[1]
				dim = (750, int(frame.shape[0] * r))
				frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
				if frame.shape[0] > 600:
					frame = cv2.resize(frame, (500, 500))


				audio_frame, val = player.get_frame()

				if val != 'eof' and audio_frame is not None:
					img, t = audio_frame

				cv2.setMouseCallback('Video', click_and_crop)

				if cropping == 3:
					roi = frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
					cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 0)
					print(refPt)

					cv2.imwrite(d + '/media/output/screenshots/' + str(j) + '.jpg', roi)
					j = j + 1
					cropping = 1


				cv2.imshow('Video', frame)

				status = {ord('s'): 'stay', ord('S'): 'stay',
						  ord('w'): 'play', ord('W'): 'play',
						  -1: status,
						  27: 'exit'}[cv2.waitKey(28)]

				if status == 'play':
					i += 1
					cv2.setTrackbarPos('S', 'Video', i)
					player.set_pause(False)
					continue
				if status == 'stay':
					i = cv2.getTrackbarPos('S', 'Video')

					player.set_pause(True)
				if status == 'exit':
					break


				if (cv2.waitKey(1) & 0xFF == ord('q')):
					break
			except KeyError:
				print("Invalid Key was pressed")
		cv2.destroyWindow('Video')
		vidcap.release()
		cv2.destroyAllWindows()
	elif 'v_submitted' in request.POST:

		myfile = request.FILES['video']
		print(myfile)

		fs = FileSystemStorage()
		#filename = fs.save(myfile.name, myfile)
		filename = fs.save('video', myfile)

		uploaded_file_url = fs.url(filename)

		cv2.namedWindow('Video')
		cv2.moveWindow('Video', 250, 150)


		vidcap = cv2.VideoCapture(d+uploaded_file_url)
		tots = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

		i = 0
		cv2.createTrackbar('S', 'Video', 0, int(tots) - 1, flick)
		cv2.setTrackbarPos('S', 'Video', 0)


		status = 'stay'

		player = MediaPlayer(d+uploaded_file_url)

		while (vidcap.isOpened()):

			try:
				if i == tots - 1:
					i = 0
				vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
				success, frame = vidcap.read()

				r = 750.0 / frame.shape[1]
				dim = (750, int(frame.shape[0] * r))
				frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
				if frame.shape[0] > 600:
					frame = cv2.resize(frame, (500, 500))


				audio_frame, val = player.get_frame()

				if val != 'eof' and audio_frame is not None:
					img, t = audio_frame


				cv2.setMouseCallback('Video', click_and_crop)

				if cropping == 3:

					roi = frame[refPt[0][1]+1:refPt[1][1], refPt[0][0]+1:refPt[1][0]]
					cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 0)
					print(refPt)

					cv2.imwrite(d + '/media/output/screenshots/' + str(j) + '.jpg', roi)
					j = j + 1
					cropping = 1


				cv2.imshow('Video', frame)
				status = {ord('s'): 'stay', ord('S'): 'stay',
						  ord('P'): 'play', ord('p'): 'play',

						  -1: status,
						  27: 'exit'}[cv2.waitKey(28)]

				if status == 'play':
					i += 1
					cv2.setTrackbarPos('S', 'Video', i)
					player.set_pause(False)
					continue
				if status == 'stay':
					i = cv2.getTrackbarPos('S', 'Video')

					player.set_pause(True)
				if status == 'exit':
					break
				if (cv2.waitKey(1) & 0xFF == ord('q')):
					break
			except KeyError:
				print("Invalid Key was pressed")
		cv2.destroyWindow('Video')
		vidcap.release()
		cv2.destroyAllWindows()
	context={}
	file = open('file.txt', 'w')
	file.write("")
	for img in os.listdir(d+'/media/output/screenshots'):
		if os.path.exists(str(d) + '/media/line/'):
			shutil.rmtree(str(d) + '/media/line/')
		if os.path.exists(str(d) + '/media/output/wordfiles/'):
			shutil.rmtree(str(d) + '/media/output/wordfiles/')
		if os.path.exists(str(d) + '/media/output/charfiles/'):
			shutil.rmtree(str(d) + '/media/output/charfiles/')

		if not os.path.exists(str(d) + '/media/line'):
			os.mkdir(str(d) + '/media/line')
		if not os.path.exists(str(d) + '/media/output/wordfiles'):
			os.mkdir(str(d) + '/media/output/wordfiles')
		if not os.path.exists(str(d) + '/media/output/charfiles'):
			os.mkdir(str(d) + '/media/output/charfiles')

		line_detect(d+'/media/output/screenshots/'+img)
		word_detect(d+'/media/line')
		char_detect()
		document()
		file=open("file.txt",'r')
		data=file.read()
		context={'data':data}
	return render(request,'app/index.html',context)



def examples(request):
	if (request.method == 'POST'):
		myfile = request.FILES['image']
		fs = FileSystemStorage()
		filename = fs.save('image', myfile)
		uploaded_file_url = fs.url(filename)
		print(uploaded_file_url)

		if os.path.exists(str(d) + '/media/line/'):
			shutil.rmtree(str(d) + '/media/line/')
		if os.path.exists(str(d) + '/media/output/wordfiles/'):
			shutil.rmtree(str(d) + '/media/output/wordfiles/')
		if os.path.exists(str(d) + '/media/output/charfiles/'):
			shutil.rmtree(str(d) + '/media/output/charfiles/')

		if not os.path.exists(str(d) + '/media/line'):
			os.mkdir(str(d) + '/media/line')
		if not os.path.exists(str(d) + '/media/output/wordfiles'):
			os.mkdir(str(d) + '/media/output/wordfiles')
		if not os.path.exists(str(d) + '/media/output/charfiles'):
			os.mkdir(str(d) + '/media/output/charfiles')

		line_detect(str(d)+uploaded_file_url)
		word_detect(str(d)+'/media/line')
		char_detect()
		file = open('file.txt', 'w')
		file.write("")
		document()
		file = open("file.txt", 'r')
		data = file.read()
		context = {'data': data}
		return render(request, 'app/examples.html',context)
	return render(request, 'app/examples.html')

def AboutUs(request):
	return render(request, 'app/page.html')

def char_detect():
	directory2=str(d)+'/media/output/charfiles'
	worddir=0
	folder_s=list()
	for folder in os.listdir(directory):
		folder_s.append((int)(folder))
	folder_s.sort()
	for folder in folder_s:
		worddir=worddir+1
		if not os.path.exists(directory2+'/'+str(worddir)):
			os.mkdir(directory2+'/'+str(worddir))

		fn=0
		#print(folder)
		files_s=list()
		for files in os.listdir(os.path.join(str(directory),str(folder))):
			files_s.append((int)(files.split('.')[0]))
		files_s.sort()
		for files in files_s:
			fn=fn+1
			if not os.path.exists(directory2+'/'+str(worddir)+'/{}'.format(fn)):
				os.mkdir(directory2+'/'+str(worddir)+'/{}'.format(fn))

			img = cv2.imread(directory+str(folder)+"/"+str(files)+".png")
			print(directory+str(folder)+"/"+str(files)+".png")
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
			#cv2.imshow('thresh', thresh)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			ctrs, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
			j=0
			for i, ctr in enumerate(sorted_ctrs):
				x, y, w, h = cv2.boundingRect(ctr)

				roi = img[y:y + h, x:x + w]

				area = w*h
				if 100 < area < 900:
					j=j+1
					cv2.imwrite(str(d)+'/media/output/charfiles/'+str(worddir)+'/'+str(fn)+'/{}.png'.format(j), roi)

def word_detect(inputFile):
	fno=1
	imgs=os.listdir(inputFile)
	l=list()
	for img in imgs:
		l.append((int)(img.split('.')[0]))
	l.sort()
	for img in l:
		s=str(img)+'.png'
		img=cv2.imread(inputFile+"/"+s)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		text_detect(fno,img)
		fno=fno+1


def text_detect(fno,img,ele_size=(8,2)):
    if len(img.shape)==3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_sobel = cv2.Sobel(img,cv2.CV_8U,1,0)
    img_threshold = cv2.threshold(img_sobel,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS ,ele_size)
    img_threshold = cv2.morphologyEx(img_threshold[1],cv2.MORPH_CLOSE,element)
    #cv2.imshow('img',img_threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours,_ = cv2.findContours(img_threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #print("no of contours",len(contours))
    i=0
    contours=sorted(contours,key=lambda ctr:cv2.boundingRect(ctr)[0])
    for cnt in contours:
        rect2=cv2.boundingRect(cnt)
        x,y,w,h=rect2
        roi = img[y:y+h+3, x:x+w]
        #cv2.rectangle(img,(x,y),( x + w, y + h ),(0,255,0),2)
        if w > 2 and h > 2:
            i=i+1
            #cv2.imshow('roi',roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(str(d)+'/media/output/wordfiles/'+str(fno)+'/{}.png'.format(i), roi)


def line_detect(uploaded_file_url):
	img = cv2.imread(uploaded_file_url)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	if len(img.shape)==3:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#cv2.imshow('gray',gray)
	th, threshed = cv2.threshold(gray,100, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

	#cv2.imshow('threshed1',threshed1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	hist = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)
	#print(hist)
	minimum=min(hist)
	maximum=max(hist)
	th=maximum-50
	th = 10
	H,W = img.shape[:2]

	uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]

	lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

	if len(uppers)<2:
		if len(lowers)<2:
			th=minimum+100
			uppers = [y for y in range(H-1) if hist[y]>=th and hist[y+1]<th]

			lowers = [y for y in range(H-1) if hist[y]<th and hist[y+1]>=th]


	if uppers[0]>lowers[0]:
		for i in range(len(lowers)-1):
			lowers[i]=lowers[i+1]

	print("upper",uppers)
	print("lower",lowers)

	rotated = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

	j=0

	for i in range(len(uppers)):
		print(lowers[i])
		roi=rotated[uppers[i]-2:lowers[i]+2,0:W]
		a,b=roi.shape[:2]

		if a>5:

			j=j+1

			cv2.imwrite(str(d)+'/media/line/{}.png'.format(j), roi)
			if not os.path.exists(directory+'{}'.format(j)):
				os.mkdir(directory+'{}'.format(j))

	cv2.waitKey(0)
	cv2.destroyAllWindows()


def document():
	path = str(d)+'/media/output/charfiles/'
	file = open('file.txt', 'a+')
	model = load_model("sample.model")

	categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
				  "A", "a", "B", "b", "c", "D", "d", "E", "e", "F", "f", "G", "g", "H", "h", "I", "i", "j", "k", "L",
				  "l", "M","m", "N", "n", "o", "p", "Q", "q", "R", "r", "s", "T", "t", "U", "u", "v", "w", "x", "Y", "y",
				  "z", "forward slash", "#", ",", "$", "&", "%", "^", "+", "?", "non_char"
				  ]
	folder_s=list()
	for f in os.listdir(path):
		folder_s.append((int)(f))
	folder_s.sort()
	#print("folder",folder_s)
	for folder in folder_s:
		word_s=list()
		for w in os.listdir(os.path.join(path, str(folder))):
			word_s.append((int)(w))
		word_s.sort()
		#print("word",word_s)
		for word in word_s:
			char_s=list()
			for c in os.listdir(os.path.join(path, str(folder), str(word))):
				char_s.append((int)(c.split('.')[0]))
			char_s.sort()
			#print("char",char_s)
			for char in char_s:
				#print("char",char)
				print(path+ str(folder)+"/"+str(word)+"/"+str(char))
				img_array = cv2.imread(path+ str(folder)+"/"+str(word)+"/"+str(char)+".png",cv2.IMREAD_GRAYSCALE)
				img_array.shape[:2]
				if img_array is not None:
					new_array = cv2.resize(img_array, (50, 50))
					test_image = new_array.reshape(-1, 50, 50, 1)
					prediction = model.predict_classes(test_image, verbose=0)
					if(categories[prediction[0]]=='non_char'):
						continue
					else:
						file.write(categories[prediction[0]])
						print('{}'.format(categories[prediction[0]]))
			file.write(" ")
		file.write("\n")
	file.close()




