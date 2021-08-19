# -*- coding: utf-8 -*-

# Based on https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
# and https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/

# USAGE
# python faces_from_video-cv-wo_cluster.py --video <path to video> 
# --min_width <minimum width of faces> --device <cpu|gpu>

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from insightface.app import FaceAnalysis
import pickle


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to video file")
ap.add_argument("-w", "--min_width", type=int, default=40,
	help="minimum width of faces to consider")
ap.add_argument("-d","--device",type=str, default='gpu',
	help="device. cpu or gpu")
args = vars(ap.parse_args())

#initialize face detection and recognition model
model = FaceAnalysis(ga_name=None)
ctx_id = 0 if args["device"]=="gpu" else -1
model.prepare(ctx_id=ctx_id, nms=0.4)

# Create output folder, if it doesn't exist. If it already exists, ask the user if [s]he wants to continue
video_folder = os.path.dirname(args["video"])
video_name = os.path.splitext(os.path.basename(args["video"]))[0]
output_folder = os.path.join(video_folder, video_name)
if not os.path.exists(output_folder):
		os.makedirs(output_folder)
else:
	reply = str(input("It seems that you already processed this video with the same parameters. Continue? [Y/n]"))
	if reply == "n" or reply == "N":
		quit()


# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
#vs = FileVideoStream(args["video"]).start()
vs = cv2.VideoCapture(args["video"])
time.sleep(2.0)
frame_number=0
data={}
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream
	# consider resizing it to have a maximum width of e.g. 400 pixels
	ret, frame = vs.read()
	if not ret:
		break
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	frame_original = frame.copy()
	if w > 400:
		scale = 400./w
		frame = cv2.resize(frame,None, fx=scale, fy=scale,interpolation=cv2.INTER_AREA)
	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	
	# test to consider only part of the video, to try to detect smaller faces
	# adjustments to startX, endX, startY and endY may be necessary
 
	faces = model.get(frame)

    # Indicate if no face is detected
	if len(faces) == 0:
		print("No face found on frame #{}".format(frame_number))
		frame_number += 1
		continue
    
	dist=[]
	center = np.array([frame.shape[0]//2,frame.shape[1]//2])
    # Compute centroids of faces and distances from certer of image
	for face in faces:
		box=face.bbox.astype(np.int).flatten()
		centroid = np.array([(box[0]+box[2])//2,(box[1]+box[3])//2])
		dist.append(np.linalg.norm(center-centroid))
    
    # Get embeddings of the face with centroid closest to the center of the image
	idx_face = dist.index(min(dist))
	rep = faces[idx_face].normed_embedding
	data[frame_number] = rep
	startX, startY, endX, endY = (faces[idx_face].bbox/scale).astype(np.int)
    
	# add a 100% margin on the bounding box
	out_width = (endX-startX)*2.0
	out_height = (endY-startY)*2.0
		
	startX_out = int((startX+endX)/2 - out_width/2)
	endX_out = int((startX+endX)/2 + out_width/2)
	startY_out = int((startY+endY)/2 - out_height/2)
	endY_out = int((startY+endY)/2 + out_height/2)
		
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
	face_crop = frame_original[startY_out:endY_out, startX_out:endX_out]
	imgPath = os.path.join(output_folder, "frame_"+ str(frame_number) + ".png")
	cv2.imwrite(imgPath,face_crop)
	
	cv2.rectangle(frame_original, (startX, startY), (endX, endY),(0, 0, 255), 2)

	frame_number += 1
	
	# show the output frame - Not necessary if running in background!
	cv2.imshow("Frame", frame_original)
	
	

	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()

datapath = os.path.join(output_folder, "features.pkl")
with open(datapath,'wb') as f:
	pickle.dump(data,f)