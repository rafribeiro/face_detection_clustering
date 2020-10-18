# -*- coding: utf-8 -*-

# Based on https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
# and https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/

# USAGE
# python faces_from_video-cv.py --video <path to video> --prototxt deploy.prototxt.txt
# --model res10_300x300_ssd_iter_140000.caffemodel --confidence <confidence required 
# of face detections> --min_width <minimum width of faces> -- jobs <# of paralles jobs to run>
# --epsilon <distance between clusters>

# import the necessary packages
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import face_recognition
import argparse
import time
import cv2
import os

from shutil import copyfile

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to video file")
ap.add_argument("-p", "--prototxt", type=str, default="deploy.prototxt.txt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", type=str, default="res10_300x300_ssd_iter_140000.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.75,
	help="minimum probability to filter weak detections")
ap.add_argument("-w", "--min_width", type=int, default=40,
	help="minimum width of faces to consider")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of parallel jobs to run (-1 will use all CPUs)")
ap.add_argument("-e", "--epsilon", type=float, default=0.4,
	help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
args = vars(ap.parse_args())

# Create output folder, if it doesn't exist. If it already exists, ask the user if [s]he wants to continue
video_folder = os.path.dirname(args["video"])
video_name = os.path.splitext(os.path.basename(args["video"]))[0]
output_folder = os.path.join(video_folder, video_name + "_output-cv" + "-c_"
 + str(args["confidence"]) + "-w_" + str(args["min_width"]) + "-e_" + str(args["epsilon"]))
if not os.path.exists(output_folder):
		os.makedirs(output_folder)
else:
	reply = str(input("It seems that you already processed this video with the same parameters. Continue? [Y/n]"))
	if reply == "n" or reply == "N":
		quit()


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
#vs = FileVideoStream(args["video"]).start()
vs = cv2.VideoCapture(args["video"])
time.sleep(2.0)
frame_number=0
data=[]
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream
	# consider resizing it to have a maximum width of e.g. 400 pixels
	ret, frame = vs.read()
	if not ret:
		break
	frame_number += 1
 
	# grab the frame dimensions and convert it to a blob
	#frame = imutils.resize(frame, width=800)
	(h, w) = frame.shape[:2]
	
	# test to consider only part of the video, to try to detect smaller faces
	# adjustments to startX, endX, startY and endY may be necessary
	#frame_crop = frame[0:int(h/2), 0:int(w/2)]
	
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		

		# adjustments, according to frame_crop
#		startX += int(w/2)
#		endX += int(w/2)
#		startY += int(h/2)
#		endY += int(h/2)

		width = endX-startX
		height = endY-startY
		
		# filter out small faces
		if width < args["min_width"] or height < 1.33*args["min_width"]:
			continue

		# add a 100% margin on the bounding box
		out_width = (endX-startX)*2.0
		out_height = (endY-startY)*2.0
		
		startX_out = int((startX+endX)/2 - out_width/2)
		endX_out = int((startX+endX)/2 + out_width/2)
		startY_out = int((startY+endY)/2 - out_height/2)
		endY_out = int((startY+endY)/2 + out_height/2)
		
		# compute the top, right, bottom and left coordinates of the face,
		# referenced to the output (cropped) image.
		out_top = int(out_height/4)
		out_right = int(3*out_width/4)
		out_bottom = int(3*out_height/4)
		out_left = int(out_width/4)
		
		out_box = [(out_top, out_right, out_bottom, out_left)]
		#print(out_box)
				
		#tests if the output bbox coordinates are out of frame limits
		if startX_out < 0:
			startX_out = 0
		if endX_out > int(w):
			endX_out = int(w)
		if startY_out < 0:
			startY_out = 0
		if endY_out > int(h):
			endY_out = int(h)

		#export the face (with added margin)
		face_crop = frame[startY_out:endY_out, startX_out:endX_out]
		imgPath = os.path.join(output_folder, str(frame_number) + "_face_" + str(i) + ".png")
		cv2.imwrite(imgPath,face_crop)

		# compute the facial embedding for the face
		encoding = face_recognition.face_encodings(face_crop, out_box)
		print(imgPath)
		print (encoding)

		# build a dictionary of the image path and bounding box location,
		# and facial encoding for the current face
		d = [{"imagePath": imgPath, "loc": box, "encoding": enc}
			for (box, enc) in zip(out_box, encoding)]
		data.extend(d)
		 
		# draw the bounding box of the face along with the associated
		# probability - Not necessary if running in background!
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


	# show the output frame - Not necessary if running in background!
	#cv2.rectangle(frame, (0,0), (int(w/2),int(h/2)), (0,255,0), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()

###### Cluster faces #######

# extract the set of encodings to so we can cluster on
# them

encodings = [d["encoding"] for d in data]

# cluster the embeddings
print("[INFO] clustering...")
clt = DBSCAN(eps=args["epsilon"], metric="euclidean", n_jobs=args["jobs"])
clt.fit(encodings)

print("core_sample_indices:")
print(clt.core_sample_indices_)

print("labels:")
print(clt.labels_)

# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

#write parameters to clusters.txt
clusters_txt = os.path.join(output_folder, "clusters.txt")

with open(clusters_txt, "a") as myfile:
	myfile.write("Video: " + args["video"]+"\n"+
	             "Confidence: "+ str(args["confidence"])+"\n"+
	  	         "Minimum face width: " + str(args["min_width"])+"\n"+
	   	         "Cluster distance parameter (epsilon): " + str(args["epsilon"])+"\n"+
	   	         "Prototxt: " + args["prototxt"]+"\n"+
	   	         "Model: " + args["model"]+"\n\n")

# loop over the unique face integers
for labelID in labelIDs:
	# find all indexes into the `data` array that belong to the
	# current label ID
	print("[INFO] faces for face ID: {}".format(labelID))
	idxs = np.where(clt.labels_ == labelID)[0]
	
	# initialize the list of faces to include in the montage
	faces = []
	
	# create a folder to each labelID
	if labelID == -1:
		label_folder = "face_unknown"
	else:
		label_folder = "face_" + str(labelID)
	if not os.path.exists(label_folder):
		os.makedirs(os.path.join(output_folder,label_folder))
	

	# loop over the sampled indexes
	for i in idxs:
		# load the input image and extract the face ROI
		image = cv2.imread(data[i]["imagePath"])
		(top, right, bottom, left) = data[i]["loc"]
		face = image[top:bottom, left:right]
		
		# force resize the face ROI to 96x96 and then add it to the
		# faces montage list
		face = cv2.resize(face, (96, 96))
		faces.append(face)

		#copy file to folder of corresponding labelID
		src_path = data[i]["imagePath"]
		f_name = os.path.basename(src_path)
		dst_path = os.path.join(output_folder, label_folder, f_name)
		copyfile(src_path, dst_path)

		# append label and file name to clusters.txt
		if os.name == "nt": #running on windows
			with open(clusters_txt, "a") as myfile:
				myfile.write(str(labelID) + ";" + label_folder +"\\" + f_name + "\n")
		else: #assuming running on linux
			with open(clusters_txt, "a") as myfile:
				myfile.write(str(labelID) + ";" + label_folder +"/" + f_name + "\n")
		
	
	# create a montage using 96x96 "tiles" with 5 rows and 5 columns
	montage = build_montages(faces, (96, 96), (5, 5))[0]
	
	# show the output montage
	title = "Face ID #{}".format(labelID)
	title = "Unknown Faces" if labelID == -1 else title
	cv2.imshow(title, montage)
	cv2.waitKey(0)


