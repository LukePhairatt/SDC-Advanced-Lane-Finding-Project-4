# **Project 4: Advanced Lane Finding Project** 

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* [**Rubric Points**](https://review.udacity.com/#!/rubrics/571/view)

[//]: # (Image References)

[image1]: ./output_images/camera_calib.png "Undistorted"
[image2]: ./output_images/warp.png "Road Transformed"
[image3]: ./output_images/SobelPlusColorBinary.png "Binary Sobel and Color Combine Example"
[image4]: ./output_images/mask_roi.png "Binary Mask Roi Example"
[image5]: ./output_images/curvefit_startup.png "Fit Visual Start"
[image6]: ./output_images/curvefit_iterate.png "Fit Visual Next"
[image7]: ./output_images/final.png "Output"
[video1]: ./project_video_out.mp4 "Video"


### Intro
_The challenges I face were_
* Finding the good set of thresholding ranges that work on normal, bright, dark conditions.
* Finding the bit mask logic for combining binary extraction in sobel edge detection and color channel hls or rgb   
  in order to get the best possible result and work on various conditions.  
* Trying to find the solution to detect bad conditions in order to stabilize the curve radius and vehicle position calculation  
  as well as drawing back the project lines and region with the stable result.   
* Exception handling when things go wrong. The system should recovery and continue without stopping. 

_Solution to the problems_
* To find the good thresholding, I extracted the images with normal, bright, dark conditions and observe the thresholding result on each channel (sobel, hls, rgb).  
  After extracting/scaling the information on each channel, one might work better than the other on one condition but not the other. Which is good in the way so that  
  I can combine (bit mask) them to work on various conditions.  
* Using a  moving n-window average to compute the mean result of the curve radius and 	vehicle position.   
* Do the sanity check to avoid recording the bad result to the history.  
  When a bad condition detected in a consecutive order, it is a time to reset and start fresh.  


I took the following steps to acheive the good result on the project video.

### Camera Calibration
* The code for this step is in  CameraCalibExtra.py file (line 10-73). 
* Calibrate with 9x6 grid pattern
* I gain extra calibation pictures (picture 1&5) when they failed with 9x6 pattern due to incomplete FOV.  
  These picture were remapped to 9x5 and 7x6 pattern for the calibration.
* The camera parameters are save to the pickle file

The sample undistorted image is shown below:  
![Undistorted][image1]  

### Image processing pipline 
* Codes in **AdvanceLane_PiplineVideo.py**
* This has been achieved with the combination of Sobel edge detection, HLS, and RGB
* From experimenting with various images on different lighting condition such as Normal, Bright, Dark. I succeeded with this combination.

**Sobel**  
bin_output[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1 

**HLS and RGB**  
bin_output[ ((bin_h == 1) & (bin_s == 1))  | (bin_r ==1) ] = 1

**Combined result**  
bin_output[ ((bin_sobel == 1) | (bin_color == 1))] = 1  
![Binary Sobel and Color Combine Example][image3]  


* Later the image was cropped to the specific ROI, as shown below  
![Binary Mask Roi Example][image4]  


### Perspective transformation  
![Road Transformed][image2]  

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 203, 720      | 320, 720      | 
| 580, 460      | 320, 0        |
| 700, 460      | 960, 0        | 
| 1100,720      | 960, 720      |

* The code for my perspective transform includes a function called **‘ComputeTransform()’**,   
  which appears in lines 78 through 91 in the file **‘CameraCalibExtra.py’**.

### Left/Right Lane lines detection 
* On start up, I use the histogram peak approach to search the line pixels and fit the 2nd order polynomial line like the one below.   
  **ExtractLinesFromInitial()** in **AdvanceLane_PiplineVideo.py** (line 218-334)  
![Fit Visual Start][image5]


* On the next image frame, I don’t start to search blindly. I use the previous fit line to pick the neighbor pixels around these line for the fit.  
  **ExtractLinesReiterate()** in **AdvanceLane_PiplineVideo.py** (line 337-405)  
![Fit Visual Next][image6]

* When bad condition detected such as the lines are not parallel, the gap is not in the range it should be, the left and right curvature is very different  
  and also the is a big jump in the average curvature from the previous loop for e.g. 2 consecutive order, we need to reset the iteration and start the search from  
  the beginning.
* There a function call **SantityCheck()** in **AdvanceLane_PiplineVideo.py** (line 549-596) to check all of these condition on every iteration. 

### Curvature radius and Vehicle position calculation
Pixel to Meter conversion: Work out from the warped straight line image  
y = line length 487 to 553  = 66 pixels for 3.0 m   =>  0.0455 m/pixel  
x = line gap 327 to 970     = 643 pixels for 3.7 m  =>  0.0060 m/pixel  

Curvature calculation function: ComputeLineCurvature() in AdvanceLane_PiplineVideo.py (line 433-469).  
In brief, I used the line fit (e.g. best fit from n-iterations) to compute the radius of curvature in a meter unit (AdvanceLane_PiplineVideo.py, line 464).

The position of the vehicle from the center was computed using the x- base point (e.g. best x after n-iterations) of the left/right line (code line 802-811).
**Position** = (image width – left base point – right base point)/2
**Direction:** On the right if Position +, On the left if Position –

### Output image  
![Output][image7]  

### Output video  
![Video][video1]  

### When it might fail
* Twist and turn corners so some of the lane lines are out of camera view, or fixed ROI.   
* When it snows or rains heavily! So not lane lines are visible or follow other track lines.   
* Moving lane might be interesting.

### Something to try out further
* Experiment with contrast adjustment/enhancement on each channel of RGB images.  
* Explore registration techniques combine with speed information to record broken line for a better polynomial fit.   
* Try optical flow to predict the line connection in substitution to the polynomial fit.   
* Try lane tracking and predicting using probabilistic approach?   




