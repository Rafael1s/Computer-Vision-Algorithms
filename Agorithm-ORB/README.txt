Oriented FAST and rotated BRIEF (ORB) is a fast robust local feature detector, 
first presented by Ethan Rublee et al. in 2011

The algorithm ORB can be efficient alternative for well-known descriptors SIFT and SURF.
The algorithm has the good invariant properties, it robust with respect to
scaling, rotation, noising, illuminations, and also the case where the image
contains several faces. 

Three Python files shows that alogorithm ORB obtains true keypoints even for 
rotated, noised images, or for image containing several faces.

The following cv2 methods are used:
  * ORB_create,
  * detectAndCompute
  * BFMatcher (Brute-Force Matcher)
 
The input files are in the subdirectory 'images'.
The metod drawMatches connects the keypoints in the training image 
with their best matching keypoints in the query image,
the result images are in the subdirectory 'match_keypoints'.

1) ORB_alg_part4_RotateInv.py 
   * get keypoints for the image face.jpeg and its rotated case faceR.jpeg       
   * compare obtained keypoints, see the result file rotate_match_keypoints.png

2) ORB_alg_part6_NoiseInv.py
   * get keypoints for the image face.jpeg and its rotated and noised case faceRN5.jpeg       
   * compare obtained keypoints, see the result file match_noise_keypoints.png

3) ORB_alg_part7_ObjectDet.py 
   * find the face in the image Team.jpeg
   * gets keypoints for the image face.jpeg and its mapping in faceRN5.jpeg       
   * compare obtained keypoints, see the result file  team_match_leypoints.jpeg
   



