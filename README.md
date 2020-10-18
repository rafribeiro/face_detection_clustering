# Face detection in video with OpenCV DNN and clustering of similar faces with DBSCAN

This is a tool for detecting faces in videos and then clustering similar faces.

It may be helpful for building face datasets from CCTV footage or for analysing long video files in search of
moments when faces are visible.

Face detection uses the dnn module of OpenCV, with a pre-trained caffe model (.prototxt and .caffemodel files are provided).

Clustering is based on the DBSCAN algorithm. There's no need to specify the number of different faces in advance. In fact, there's no way to do this.

The detection part is based on https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/ and the clustering part is based on https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/.

It was tested on Windows 10 and Ubuntu 18.04 and python 3.6.8, but should work on MAC OS and newer versions of python.


Changelog: 

2020-07-11
- output now created on the same folder of the video
- better formatting of clusters.txt
- name of the output folder now includes parameters "confidence", "min_width" and "epsilon"

2020-07-09
- compatibility with POSIX OSes. Tested on Ubuntu 18.04. Should work on MacOS

2020-07-05
- initial release

