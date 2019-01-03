
AdaBoost frontal face detector (short for Adaptive Boosting) is a machine learning meta-algorithm formulated 
by Yoav Freund and Robert Schapire, who won the 2003 Gödel Prize for their work
("A decision-theoretic generalization of on-line learning and an application to boosting". 
Journal of Computer and System Sciences. 55: 119, 1997). Viola and Jones introduced
in "Rapid Object Detection Using a Boosted Cascade of Simple Features" (2001)
a technique for speeding up processing of boosted classifiers testing each potential object 
with as many layers of the final classifier speeding up computation for cases where the class 
of the object can easily be determined (here, this is face detector).  

Haar_cascades_classif.py

  * uses AdaBoost face detector 
       haarcascade_frontalface_default.xml, 
	   cv2.CascadeClassifier
	   
  * get array of faces 
       face_cascade.detectMultiScale
	   
  * draw detected faces as a red rectangle 	   
	   cv2.rectangle
	   

input:          multi_faces.jpg
gray case: 	    multy_faces_gray.png
detected faces: boxed_faces.png