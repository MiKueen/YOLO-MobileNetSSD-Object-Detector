# USAGE
# To read and write back out to video:
# python counter.py --input <path/to/input/video/file> \
#	--output <path/to/output/folder>/yolo.avi
# To read from webcam and write back out to disk:
# python counter.py --output <path/to/output/folder>/yolo_webcam.avi


# import the necessary packages
from __future__ import division
from torch.autograd import Variable
from imutils.video import FPS
from darknet import Darknet
from util import *
import time
import torch 
import torch.nn as nn
import argparse
import os 
import os.path as osp

import imutils
import dlib
import cv2

from centroidtracker import CentroidTracker
from trackableobject import TrackableObject

def arg_parse():
	"""
	Parse arguements to the detect module
    
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--bs", help = "Batch size", default = 2, type = int)
	parser.add_argument("-o", "--output", help ="path to output video")
	parser.add_argument("--confidence", help = "Object Confidence to filter predictions", default = 0.5, type = float)
	parser.add_argument("--nms_thresh", help = "NMS Threshhold", default = 0.4, type = float)
	parser.add_argument("--cfg", help = "Config file", default = "config/yolov3.cfg", type = str)
	parser.add_argument("--weights", help = "weightsfile", default = "config/yolov3.weights.1", type = str)
	parser.add_argument("--reso", help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default = "640", type = str)
	parser.add_argument("--input", help = "Video file to run detection on", type = str)
	parser.add_argument("--skip-frames", help="frames to skip between detections", default = 25, type = int)
	
	return vars(parser.parse_args())
    

args = arg_parse()

batch_size = args.get("bs")
confidence = args.get("confidence")
nms_thesh = args.get("nms_thresh")
num_classes = 80
classes = load_classes("config/coco.names")


ct = CentroidTracker(maxDisappeared=15, maxDistance=70)
trackers = []
trackableObjects = {}

frames = 0 
W = None
H = None
 
writer = None
res = int(args["reso"])


#Set up the neural network
print("Loading network.....")
model = Darknet(args.get("cfg"))
model.load_weights(args.get("weights"))
print("Network successfully loaded")

model.net_info["height"] = args.get("reso")
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
CUDA = torch.cuda.is_available()
if CUDA:
	model.cuda()


#Set the model in evaluation mode
model.eval()


#Detection phase

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	cap = cv2.VideoCapture(0)

else:
	videofile = args.get("input") #or path to the video file. 

	cap = cv2.VideoCapture(videofile)  

#cap = cv2.VideoCapture(0)  for webcam

assert cap.isOpened(), 'Cannot capture source'

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	frame = cap.read()
	frame = frame[1] if args.get("input", False) else frame

	# if we are viewing a video and we did not grab a frame 
	#then we have reached the end of the video
	if args["input"] is not None and frame is None:
		break

	# resize the frame to have a maximum width of 500 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)
     
	status = "Waiting"
	rects = []

	if totalFrames % args["skip_frames"] == 0:
        
		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers = []
        
		img = prep_image(frame, inp_dim)
		im_dim = frame.shape[1], frame.shape[0]
		im_dim = torch.FloatTensor(im_dim).repeat(1,2)

		if CUDA:
			im_dim = im_dim.cuda()
			img = img.cuda()

		with torch.no_grad():
			output = model(Variable(img, volatile = True), CUDA)
		output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)

		if type(output) == int:
			time.sleep(0.0105492)
            
		else:
			im_dim = im_dim.repeat(output.size(0), 1)
			scaling_factor = torch.min(res/im_dim,1)[0].view(-1,1)

			output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
			output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
			
			output[:,1:5] /= scaling_factor

			for i in range(output.shape[0]):
				output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
				output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
	
        
			for x in output:
				if classes[int(x[-1])] == "person":
					centerX ,centerY = tuple(map(int,x[1:3]))  
					width,height= tuple(map(int,x[3:5]))
					img = frame

					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					startX, startY, endX, endY = x, y ,width,height

					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(centerX, centerY, width, height)
		                
					tracker.start_track(rgb, rect)
					trackers.append(tracker)
                       
        
	else:
		time.sleep(0.0218)
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"

			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			# check to see if the object has been counted or not
			if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True

				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("Up", totalUp),
		("Down", totalDown),
		("Status", status),
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	frames += 1

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	cap.stop()

# otherwise, release the video file pointer
else:
	cap.release()

# close any open windows
cv2.destroyAllWindows()
