# text-extraction-from-video-frames
An optical character recognition (OCR) algorithm built using OpenCV and TensorFlow.           
In this application, the user can play video (e.g. tutorials) and select text in video using mouse. 
The selected region is saved as an image. The text in the selected area is extracted using OCR algorithm 
and is made available to the user in a text document file.

   A saved image is segmented into text line images. Those lines are segmented into words 
and words are further segmented into character images. 
Each character image is passed to convolutional neural network (CNN) implemented using TensorFlow library. 
The CNN is trained using custom dataset created with aforementioned segmentation process implemented using OpenCV, 
i.e, we manually classified those segmented character images. 

Requirements:
django==2.2,
tensoflow==1.11,
ffpyplayer,
opencv
