# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
from cv2 import cv2
import os



# while loop -> create black image-> yolo detect -> bbox -> make bbox pixels white in black image -> bitwise and (margin and black)

#TO display: 
# bitwiseanded image -> gray2bgr -> cv2.addweighted(orig frame, 0.8, anded, 0.2)=final image

# xywh
# blankimage[y:y+h, x:x+w] = cropped or dilated or thresh

# Bbox-> crop bbox rectangle from car frame and background frame -> cv2.subtract(car frame, background) -> threshold Otsu 
# -> (1) dilate binary image ->find contours-> bitwise and       or  (2) findcontours -> rotated rectangle fit -> bitwise anding with margin image


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
# ap.add_argument("-o", "--output", required=True,
# 	help="path to output video")

ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]



# # initialize the video stream, pointer to output video file, and
# # frame dimensions
vs = cv2.VideoCapture(args["input"])
frametime=1
writer = None
(W, H) = (None, None)


# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# # loop over frames from the video file stream
while True:
	#create black image 
	imgbinary= np.zeros((640,480,3),np.uint8)
	# margin = cv2.imread('safe_area.jpg')
	margin= cv2.imread('safe_area_1.jpg')
	margin2=cv2.imread('safe_area.jpg')
	margin1=cv2.cvtColor(margin,cv2.COLOR_BGR2GRAY)
	background = cv2.imread('Background.jpg')
	# read the next frame from the file
	(grabbed, frame) = vs.read()
		

	# if the frame was not grabbed, then we have reached the end
	# of the stre1am
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
		

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				# classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for  i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the frame
			# color = [int(c) for c in COLORS[classIDs[i]]]
			rect= cv2.rectangle(frame, (x, y), (x + w, y + h),(0,0,255), 2)
			# text = "{}: {:.2f}".format(LABELS[classIDs[i]],
			# 	confidences[i])
			# cv2.putText(frame, text, (x, y - 5),
			# cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			
			#crop the bbox rectangle from car frame and background
			imgcrop=frame[y:y+h,x:x+w]
			# imgcrop=cv2.cvtColor(imgcrop,cv2.COLOR_BGR2GRAY)
			imgbgcrop=background[y:y+h,x:x+w]
			# imgbgcrop=cv2.cvtColor(imgbgcrop,cv2.COLOR_BGR2GRAY)
			margincrop=margin2[y:y+h,x:x+w]
			# margincrop=cv2.cvtColor(margin1crop,cv2.COLOR_BGR2GRAY)
			margin1crop=margin1[y:y+h,x:x+w]

			#subtract the carframe from background 
			imgsub= cv2.subtract(imgbgcrop,imgcrop)
			imgsub=imgsub*3
			# cv2.imshow('sub img ',imgsub)

			#converting the subtracted image into gray image for thresholding 
			imggray=cv2.cvtColor(imgsub,cv2.COLOR_BGR2GRAY)
			# cv2.imshow('gray',imggray)
			
			#thresholding the image
			ret,imgthresh= cv2.threshold(imggray,0,255,cv2.THRESH_OTSU)
			# cv2.imshow('thresholded image ',imgthresh)
			

			#dilating the image
			kernel=np.ones((5,5),np.uint8)
			imgdilate=cv2.dilate(imgthresh,kernel=kernel,iterations=2)
			# cv2.imshow('dilated image',imgdilate)
			
			#find contours
			imgcontour=imgdilate.copy()
			savedcontour=-1
			maxArea=0
			h,w=imgcontour.shape[:2]
			mask=np.zeros((h+2,w+2),np.uint8)
			contours,hierarchy=cv2.findContours(imgdilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			for i in range(len(contours)):
				area=cv2.contourArea(contours[i])
				if(area>maxArea):
					if(savedcontour>=0):
						cv2.floodFill(imgcontour,mask,(0,0),255)
					maxArea= area
					savedcontour=i
				else:
					cv2.floodFill(imgcontour,mask,(0,0),255)
			cv2.drawContours(imgcontour,contours,i,(0,255,0),3)
			
		
			#create hull
			hull=[]
			#calculate all the points of contour
			for i in range(len(contours)):
				hull.append(cv2.convexHull(contours[i],False))
			
			#draw hull 
			for i in range(len(contours)):
				cv2.drawContours(imgcontour,hull,i,(255,0,0),3)
			cv2.imshow('hull',imgcontour)
			# cv2.waitKey(1)

			
			# imghull=cv2.cvtColor(imgcontour,cv2.COLOR_GRAY2BGR)
			imghull=imgcontour.copy()
			
			# #bitwise anding the image 
			carframe=cv2.bitwise_and(imghull,margin1crop)
			carframe=cv2.cvtColor(carframe,cv2.COLOR_BGR2RGB)
			cv2.imshow('bitwise and',carframe)
		
			car_frame=cv2.addWeighted(imgcrop,0.8,margincrop,0.2,0)
			car_frame=cv2.addWeighted(car_frame,0.8,carframe,0.2,0)
			cv2.imshow('car',car_frame)
			car_frame.resize(640,480,3)
			print(car_frame.shape)
			print(frame.shape)
			cv2.waitKey(1)

			final_frame=cv2.addWeighted(car_frame,0.8,frame,0.2,0)
			cv2.imshow('final frame',final_frame)
			# displayframe=frame
			

			# cv2.imshow('car',displayframe)
			
			# #find if the margin is left crossed or right crossed
			# leftcrossed=False
			# rightcrossed=False
			# redoverlay=[]
			# g=np.zeros(margincrop.size(),cv2.CV_8UC1)
			# channels=[]
			# channels.append(g)
			# channels.append(Margin) 
			# channels.append(g)
			# cv2.merge(channels,redoverlay)
			# cv2.addWeighted(frame,0.8,redoverlay,0.2,0,frame)

			# if(sum):


	if cv2.waitKey(1) & 0xFF==ord('q'):
		cv2.destroyAllWindows()
