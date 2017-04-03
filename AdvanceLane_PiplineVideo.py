import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from pprint import pformat
from moviepy.editor import VideoFileClip

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, n_iteration):
        self.ReInitData(n_iteration)
  
    def __repr__(self):
        #return pformat(vars(self), indent=2, width=1)
        attrs = vars(self)
        return ''.join( "%s: %s\n" % item for item in attrs.items() )
    
    def ReInitData(self, n_iteration):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = n_iteration*[False] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([False])
        #polynomial coefficients for the most n recent fit
        self.all_fit = n_iteration*[np.array([False])] 
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #number of iteration for the best fit
        self.n_iteration = n_iteration 
        #iteration index
        self.n_index = 0
        #data filled length
        self.n_dataFilled = 0
        


# Plot result of undistorted calib image
def CheckUndistorted(mtx, dist):
    images = glob.glob('./camera_cal/calibration*.jpg')
    for fname in images:
        img = mping.imread(fname) #original distorted
        dst = cv2.undistort(img, mtx, dist, None, mtx) #undistorted
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(dst)
        plt.show()

# Check perspective tranformation
def CheckTransform(img_rgb, mtx, dist, M):
    # undistort image rgb
    undist_img = cv2.undistort(img_rgb, mtx, dist, None, mtx) #undistorted
    # Apply transformation
    img_size = (img_rgb.shape[1],img_rgb.shape[0])
    warped_img = ApplyTransformation(undist_img, M, img_size)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.subplot(1,2,2)
    plt.imshow(warped_img)
    plt.show()
    

# Undistort the image
def UndistortImage(img_src, mtx, dist, plot_result=False):
    img_undist = cv2.undistort(img_src, mtx, dist, None, mtx)
    if plot_result:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img_src)
        plt.title("Original Image")
        plt.subplot(1,2,2)
        plt.imshow(img_undist)
        plt.title("Undistorted Image")
        plt.show()
        
    return img_undist

def ApplyTransformation(undist_img, M, img_size):
    return cv2.warpPerspective(undist_img, M, img_size)

    
# Sobel-Calculate directional gradient
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    grad_binary = np.zeros_like(scaled_sobel)
    # 6) Apply threshold
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

# Sobel-Calculate gradient magnitude
def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # 3) Calculate the magnitude
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))
    # 5) Create a binary mask where mag thresholds are met
    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return mag_binary

# Sobel-Calculate gradient direction
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary

def ApplySobelThreshold(image_rgb, gradx, grady,mag, direct):
    # Note: all params include (kernel,threshold_min, threshold_max)
    # Convert RGB to gray scale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=gradx[0], thresh=(gradx[1], gradx[2]))  #3,30,100   (3,20,100)
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=grady[0], thresh=(grady[1], grady[2]))  #9,50,100   (3,20,100)
    mag_binary = mag_thresh(gray, sobel_kernel=mag[0], mag_thresh=(mag[1], mag[2]))                 #9,40,100   (3,30,100)
    dir_binary = dir_threshold(gray, sobel_kernel=direct[0], thresh=(direct[1], direct[2]))         #15,0.7,1.3 (15,0.7,1.3)
    # Combine result
    bin_output = np.zeros_like(dir_binary)
    bin_output[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return bin_output
    
    
def ColorChannelThreshold(img, hls_thresh, rgb_thresh):
    # Convert to HLS color space
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Apply a threshold to the HLS channel
    hls_h = img_hls[:,:,0] 
    hls_l = img_hls[:,:,1] 
    hls_s = img_hls[:,:,2] 
    
    # Return a binary image of threshold result
    h_min, h_max = hls_thresh[0]
    l_min, l_max = hls_thresh[1]
    s_min, s_max = hls_thresh[2]
    bin_h = np.zeros_like(hls_h)
    bin_h[(hls_h > h_min) & (hls_h <= h_max)] = 1
    bin_l = np.zeros_like(hls_l)
    bin_l[(hls_l > l_min) & (hls_l <= l_max)] = 1
    bin_s = np.zeros_like(hls_s)
    bin_s[(hls_s > s_min) & (hls_s <= s_max)] = 1
    
    # Apply a threshold to the RGB channel
    r_min, r_max = rgb_thresh[0]
    rgb_r = img[:,:,0] 
    bin_r = np.zeros_like(rgb_r)
    bin_r[(rgb_r > r_min) & (rgb_r <= r_max)] = 1
    
    # Combine result
    bin_output = np.zeros_like(hls_h)
    #bin_output[ ((bin_h == 1) | (bin_r == 1)| (bin_s == 1)) ] = 1
    bin_output[ ((bin_h == 1) & (bin_s == 1))  | (bin_r ==1) ] = 1
    return bin_output

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (1,) * channel_count
    else:
        ignore_mask_color = 1
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def ExtractLinesFromInitial(binary_warped, plot_result=False):
    #### INIT ####
    found_left = False
    found_right = False
    left_fit = None
    right_fit = None

    #### HISTOGRAM PEAK ####
    #compute each row sum from bottom half (note indexing n:,)
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    if plot_result:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    # Fix! Note Hopefully the midpoint is in between left-right peak!
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    #### SEARCH WINDOW #####
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    # return tuple x,y of np array type
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    #### EXTRACT LINE WITHIN WINDOW ROI ####
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        # Note: There are 2 parallel bounding boxes centred at LEFT and RIGHT
        #       So we need to form the corredinate like this
        '''
            win_xleft_low   win_xleft_high    win_xright_low   win_xright_high
                    +------------+                  +-------------+ win_y_low
                          + leftx_current                   + rightx_current
                    +------------+                  +-------------+ win_y_high                   
        '''
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if plot_result:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window bounding box
        good_left_inds  = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        # Take x,y average from the pixel in the bounding box left and right
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
     
    #### RECORD FOUND X,Y INDEX OF THE LEFT/RIGHT LINES
    # This will repackage list of [[],[],[]] to numpy([])
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions (Extract pixel positions from indices)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    #### FITPOLYNOMIAL LINE ####
    # Fit a second order polynomial to each
    # Obtain polynomial coef.
    # Note: this fit given y to predict x
    found_left, left_fit = FitLinePolynomial(lefty, leftx, 2)
    found_right, right_fit = FitLinePolynomial(righty, rightx, 2)

    #### PLOT RESULT ####
    if plot_result:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        if left_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            plt.plot(left_fitx, ploty, color='yellow')
            
        if right_fit is not None:    
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.plot(right_fitx, ploty, color='yellow')
        
        # Overlay result
        plt.imshow(out_img)
        plt.xlim(0, out_img.shape[1])
        plt.ylim(out_img.shape[0], 0)
        plt.show()
    
    return left_fit,right_fit, (leftx,lefty), (rightx,righty), (found_left, found_right), (leftx_current, rightx_current), (leftx_base, rightx_base)
 

def ExtractLinesReiterate(binary_warped, left_fit, right_fit, plot_result=False):
    # TODO- This code is repeated from ExtractLinesFromInitial and is needed to repackage for reusablity
    # Wihout searching blindly
    # Use the same poly fit line (that might be good enough) it doesn't change much
    # Search the line within this fit line region
    # Do it this way the search will be quicker
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #### FITPOLYNOMIAL LINE ####
    # Fit a second order polynomial to each
    # Obtain polynomial coef.
    # Note: this fit given y to predict x
    found_left, left_fit = FitLinePolynomial(lefty, leftx, 2)
    found_right, right_fit = FitLinePolynomial(righty, rightx, 2) 
  
    #### COMPUTE X BASE, X TOP POINTS ####
    # given y=0 : x top
    # given y=image height: x base
    leftx_current, leftx_base  = GetTopBottomFromFit(left_fit,0,binary_warped.shape[0])
    rightx_current,rightx_base = GetTopBottomFromFit(right_fit,0,binary_warped.shape[0])
    
    #### PLOT RESULT ####
    if plot_result and found_left and found_right:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show() 
    return left_fit,right_fit, (leftx,lefty), (rightx,righty), (found_left, found_right), (leftx_current, rightx_current), (leftx_base, rightx_base)
    
    
def FitLinePolynomial(y_points, x_points, fit_order):
    # Fit a second order polynomial to the line x,y data
    # Note: this fit given y data(row) to predict x data (column)
    # TODO- Safe guard - check if points are valid/not empty
    if len(y_points) > 10:
        line_fit = np.polyfit(y_points, x_points, fit_order)
        line_found = True
    else:
        print("Warning not enough points to fit the line.. return none fit")
        line_fit = None
        line_found = False
    return line_found, line_fit

def GetTopBottomFromFit(line_fit, y_top, y_base):
    # Given line fit and y points, produce x points
    # x_base will be used to track the vehicle positioning
    try:
        x_top  = line_fit[0]*y_top**2  + line_fit[1]*y_top  + line_fit[2]
        x_base = line_fit[0]*y_base**2 + line_fit[1]*y_base + line_fit[2]
    except:
        print("Could not find X top bottom points")
        x_top = None
        x_base = None
    return x_top, x_base 

def ComputeLineCurvature(x_pixels, y_pixels, y_eval, xm_per_pixel, ym_per_pixel, line_fit):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image (front wheel near the bonet)
    # curvature equation R = ( 1+ (2Ay+B)**2)**1.5
    #                        ---------------------
    #                                abs(2A)
    # where f(y) = Ay**2 + By + C
    # A = fit[0], B = fit[1], C = fit[2]
    ############ Now we want that in the real work size, not in the pixel size ###################
    # Define conversions in x and y from pixels to meters
    # ym_per_pix = 30/720  # meters per pixel in y dimension (720 pixels/30 meter)
    # xm_per_pix = 3.7/700 # meters per pixel in x dimension (700 pixels/3.7 meter)
    # Work out from the warped stratight line image
    # y = line length 487 to 553 = 66 pixels for 3.0 m => 0.0455 m/pixel
    # x = line gap 327 to 970 = 643 pixels for 3.7 m => 0.006 m/pixel
    # Calculate the new radii of curvature
    # Fit new polynomials to x,y in world space if none given
    if line_fit is None:
        try:
            fit_cr = np.polyfit(y_pixels*ym_per_pixel, x_pixels*xm_per_pixel, 2)
        except:
            print("Compute curvature fail in poly fit function.. check x y input pixels")
            return None
    # use the given fit function (Note line_fit is in Pixel unit):
    else:
        #Compute poly fit from pixel to m unit
        ypix = np.linspace(0, 720-1, 720)
        xpix = line_fit[0]*ypix**2 + line_fit[1]*ypix + line_fit[2]
        fit_cr = np.polyfit(ypix * ym_per_pixel, xpix * xm_per_pixel, 2)
        
    try:
        curverad_m = ((1 + (2*fit_cr[0]*y_eval*ym_per_pixel + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    except:
        print("Compute curvature fail check poly fit coeff")
        return None
    
    return curverad_m

def InverseProjectionImage(bin_warped, img_undist, img_size, left_fit, right_fit, left_pixs, right_pixs, Minv, plot_result=False):
    try:
        # img_size is x,y or column,row
        warp_zero = np.zeros_like(bin_warped).astype(np.uint8)
        # Stack them up
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Add the detected left and right line pixels to color_warp channel
        color_warp[left_pixs[1], left_pixs[0]] = [255, 0, 0]
        color_warp[right_pixs[1], right_pixs[0]] = [0, 0, 255]
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        ploty = np.linspace(0, img_size[1]-1, num=img_size[1])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,200, 0))
        
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, img_size) 
        # Combine the result with the original image
        result = cv2.addWeighted(img_undist, 0.7, newwarp, 0.4, 0)
        if plot_result:
            plt.imshow(result)
            plt.show()
    except:
        print("Inverse projection inconsistency..check line fits")
        #TODO- Save the problem image for further analysis
        result = img_undist
        
    return result
    
def RecordLineHistory(Line, found_line, basepoint_x, line_base_pose, line_fit, line_curverad, line_pix):
    # Record line data for the next frame
    Line.detected = found_line                                          # was the line detected in the last iteration?
    # If no line found we skip to maintain the existing data, so we have something to work on/display
    if found_line:
        # Add data to the line
        Line.diffs =  Line.current_fit - line_fit                       # (Pixel) difference in fit coefficients between last and new fits
        Line.current_fit = line_fit                                     # (Pixel) polynomial coefficients for the most recent fit
        Line.radius_of_curvature = line_curverad                        # (Meter) radius of curvature of the line in some units 
        Line.line_base_pos = line_base_pose                             # (Meter) distance in meters of vehicle center from the line 
        Line.allx = line_pix[0]                                         # (Pixel) x values for detected line pixels 
        Line.ally = line_pix[1]                                         # (Pixel) y values for detected line pixels
        Line.all_fit[Line.n_index] = line_fit                           # (Pixel) polynomial coefficients for the last n iteration
        Line.recent_xfitted[Line.n_index] = basepoint_x                 # (Pixel) x values (centroid) of the last n fits of the line                                            # Initial zero index increment
        
        # Compute average over n iteration like avering n window
        #n_length = len([k for k,i in enumerate(fitx) if i.all()!=False])
        # finding data length up till now
        n_length = len([k for k,i in enumerate(Line.recent_xfitted) if i!=False])
        Line.bestx = np.sum(Line.recent_xfitted)/n_length               # (Pixel) mean x over the last n iteration
        Line.best_fit = np.sum(Line.all_fit, axis=0)/n_length           # (Pixel) polynomial coefficients averaged over the last n iterations
        
        # Increment or Reset
        if Line.n_index >= Line.n_iteration-1:
            # All filled up- so restart from 0 position
            Line.n_index = 0
        else:
            # Just start from begining and all slots have not been filled so Keep indexing
            Line.n_index += 1
        
            
def DrawText(image,x,y,size,color,thickness,result_curvature, result_position, direction):
    position = "{0:.2f}".format(result_position)
    radius   = str(int(result_curvature))
    text_curvature = 'Radius of Curvature = ' + radius + ' (m)'
    text_position  = 'Vehicle is ' + position + ' (m) ' + direction
    cv2.putText(image, text_curvature, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    cv2.putText(image, text_position, (int(x), int(y+40)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    return image


def SanityCheck(LeftLineData, RightLineData, left_fit, right_fit, left_curve, right_curve, img_size, xm_per_meter, ym_per_meter):
    # TODO- Remove hard code for this particular project
    diff_curve_sts = False
    gap_sts = False
    parallel_sts = False
    curve_jump_sts = False
    ret_sts = False
    # Lines have similar curvature
    diff_curve =  abs(right_curve-left_curve)
    # Simple check between say 10 lines for average
    yp = np.linspace(0, img_size[1]-1, 10 )
    xp_l = left_fit[0]*yp**2 + left_fit[1]*yp + left_fit[2]
    xp_r = right_fit[0]*yp**2 + right_fit[1]*yp + right_fit[2]
    gap  = (xp_r-xp_l)*xm_per_meter
    gap_mean = np.mean(np.abs(gap))
    # Lines are roughly parallel
    # Simple check from the mean gap size: something close to zero is good
    parallel = gap- np.mean(gap)
    parallel_mean = np.mean(np.abs(parallel))
    
    # Check if the curve is changed all the sudden, this is the bad sign
    if LeftLineData.radius_of_curvature is None or RightLineData.radius_of_curvature is None:
        # this is normally the first loop or the line has been reset, don't need to check it
        left_jump = 0.0
        right_jump = 0.0
    else:    
        left_jump  = np.abs(LeftLineData.radius_of_curvature - left_curve)
        right_jump = np.abs(RightLineData.radius_of_curvature - right_curve)
    
    
    # Return state- Check all state or cascade style (would works faster)
    if(left_jump < 3000.0 and right_jump < 3000.0):
        curve_jump_sts = True
    
    if(diff_curve < 3000.0):
        diff_curve_sts = True
    
    if(gap_mean > 3.3 and gap_mean < 3.8):
        gap_sts = True
    
    if(parallel_mean < 0.25):
        parallel_sts = True
        
    # Combine
    if(diff_curve_sts and gap_sts and parallel_sts and curve_jump_sts):
        ret_sts = True
    
    return diff_curve, gap_mean, parallel_mean, ret_sts
    
    
class AdvanceLaneDetection():
    def __init__(self,xm_per_pixel,ym_per_pixel,line_iteration,mtx,dist,M,Minv,Sobel,hls_thresh,rgb_thresh,vertices, mode):
        self.CheckDistCalibData = False             # plot image from camera parameters 
        self.CheckTransformData = False             # plot perspective transformation
        self.PlotBinary = False                     # plot image processing
        self.ym_per_pixel = 30.0/720                # meters per pixel in y dimension (720 pixels/30 meter)
        self.xm_per_pixel = 3.7/700                 # meters per pixel in x dimension (700 pixels/3.7 meter)
        self.line_iteration = line_iteration        # n lines iteration for the best fit
        self.LeftLine = Line(line_iteration)        # construct left line data container
        self.RightLine = Line(line_iteration)       # construct right line data container
        self.mtx = mtx                              # camera matrix
        self.dist = dist                            # camera distortion
        self.M = M                                  # perspective projection 
        self.Minv = Minv                            # inverse projection
        self.gradx = Sobel[0]                       # sobel gradient x min,max
        self.grady = Sobel[1]                       # sobel gradient y min,max
        self.mag = Sobel[2]                         # sobel magnitude min,max
        self.direct = Sobel[3]                      # sobel derivative min,max
        self.hls_thresh = hls_thresh                # hls threshold min,max
        self.rgb_thresh = rgb_thresh                # rgb threshold min,max
        self.vertices = vertices                    # crop region
        self.gap = []                               # mean line gap
        self.curve = []                             # mean curve
        self.parallel = []                          # mean parallel calculation
        self.det_sts = []                           # overall detection state
        self.restart = False                        # request start
        self.badDetection = 0                       # number of consecutive bad detection
        self.badDetectionThresh = 1                 # request restart after badDetection going beyond this number
        self.mode = mode                            # 'live', 'average': overlay current or average best fit result
        self.frame_counter = 0                      # counting processed frame
        
    def ProcessorPipline(self, image):
        self.frame_counter += 1
        ####
        ##                 CHECK CAMERA PARAMETERS
        ###
        
        # Check undistorted calibration image
        if self.CheckDistCalibData:
            CheckUndistorted(self.mtx,self.dist)
       
        # Check transformation on the test sample
        if self.CheckTransformData:
            CheckTransform(self.img, self.mtx, self.dist, self.M)
            
        ####
        ##                 EXTRACT BINARY IMAGE
        ###
        # Undistort image
        img_undist = UndistortImage(image, self.mtx, self.dist, plot_result = False)
        # Thresholding to binary image: Sobel/HLS/HSV/RGB
        # SOBEL
        bin_sobel = ApplySobelThreshold(img_undist, self.gradx, self.grady, self.mag, self.direct)
        # COLOR SPACE
        bin_color = ColorChannelThreshold(img_undist, self.hls_thresh, self.rgb_thresh)     
        # Combine Edge&Color binary
        bin_output = np.zeros_like(bin_color)
        bin_output[ ((bin_sobel == 1) | (bin_color == 1))] = 1
        #mpimg.imsave('./output_images/'+ 'test' + str(self.frame_counter)+ '_bin', bin_output, format='jpg')
        # Mask ROI
        bin_output = region_of_interest(bin_output, vertices)
        #mpimg.imsave('./output_images/'+ 'test' + str(self.frame_counter)+ '_binROI', bin_output, format='jpg')
        # Perspective transform of the lane line
        img_size = (bin_output.shape[1],bin_output.shape[0])
        bin_warped = ApplyTransformation(bin_output, self.M, img_size)
        if self.PlotBinary:
            plt.figure(1)
            plt.imshow(bin_sobel, cmap='gray')
            plt.title("Sobel")
            plt.figure(2)
            plt.imshow(bin_color, cmap='gray')
            plt.title("Color Channel")
            plt.figure(3)
            plt.imshow(bin_output, cmap='gray')
            plt.title("Sobel+Color Channel")
            plt.figure(4)
            plt.imshow(bin_output, cmap='gray')
            plt.title("final combined binary mask ROI")
            plt.figure(5)
            plt.imshow(bin_warped, cmap='gray')
            plt.title("Warp")
            plt.show()
        #mpimg.imsave('./output_images/'+ 'test' + str(self.frame_counter)+'_binWarp', bin_warped, format='jpg')            
            
        ####
        ##                 LANE DETECTION LOOP
        ###
 
        # Detect a left/right line logic
        # Init local variable left/right line found state
        found_line = [False, False]
        # Following the previous lines using previously detection data if both line fit do exist
        if (self.LeftLine.current_fit[0] != False and self.RightLine.current_fit[0] != False and self.restart != True):
            left_fit, right_fit, left_pix, right_pix, found_line, midpoint_x, basepoint_x = ExtractLinesReiterate(bin_warped, self.LeftLine.best_fit, self.RightLine.best_fit, plot_result=False) 
        
        # First loop or above search failed or badly conditions from previous (need to restart again from scratch)
        if((not found_line[0]) or (not found_line[1]) or (self.restart == True) ):
            # Search from scratch
            left_fit, right_fit, left_pix, right_pix, found_line, midpoint_x, basepoint_x = ExtractLinesFromInitial(bin_warped, plot_result=False)
            
            # if we request a restart due to badly conditions, we will clean up data upto this point before adding the previously good one
            if self.restart == True:
                l_temp_bestx = self.LeftLine.bestx
                l_temp_bestfitx = self.LeftLine.best_fit
                r_temp_bestx = self.RightLine.bestx
                r_temp_bestfitx = self.RightLine.best_fit
                # clean data all
                self.LeftLine.ReInitData(self.line_iteration)
                self.RightLine.ReInitData(self.line_iteration)
                # restall initial state from the previous best fit state, hopefully the good one
                self.LeftLine.bestx = l_temp_bestx  
                self.LeftLine.best_fit= l_temp_bestfitx 
                self.RightLine.bestx= r_temp_bestx 
                self.RightLine.best_fit= r_temp_bestfitx
            else:
                # best fit is empty in the beginning: no history data yet at this point in time, so we must put something in from the current search
                self.LeftLine.bestx     = basepoint_x[0]
                self.RightLine.bestx    = basepoint_x[1]
                self.LeftLine.best_fit  = left_fit
                self.RightLine.best_fit = right_fit
                
            # Flag reset    
            self.restart = False 
            
        
        ####
        ##               INITIAL CHECK WE DETECT OR MISSING SOME LINES
        ###
        
        # If all effort is failed, so let it be, the rest of the case should handle empty data properly in calculating result further
        if not found_line[0]:
            print("Left line detection missing...Old result will be displayed")
            
        if not found_line[1]:
            print("Right line detection missing..Old result will be displayed")
        
        
        ####
        ##               CURRENT LINE RESULTS 
        ####
        
        # TODO Safe guard if a least one line is missing
        #      We need to handle it when this happens internally inside the function as well
        
        # Compute curvature in meter
        y_eval = bin_warped.shape[0]    # front wheel curvature
        left_curverad  = ComputeLineCurvature(left_pix[0], left_pix[1], y_eval, self.xm_per_pixel, self.ym_per_pixel, None)
        right_curverad = ComputeLineCurvature(right_pix[0], right_pix[1], y_eval, self.xm_per_pixel, self.ym_per_pixel, None)
        current_curverad = (left_curverad+right_curverad)/2.0
        
        # Current vehicle off center
        # Shift from the center is diff(left-right)/2
        # Compute the image center from the x base point
        line_base_pose_l = basepoint_x[0]*xm_per_pixel   # distance in meters of the base point on the left line (base)
        line_base_pose_r = basepoint_x[1]*xm_per_pixel   # distance in meters of the base point on the right line (base)
        cur_vehicle_off = (img_undist.shape[1]*self.xm_per_pixel - line_base_pose_l - line_base_pose_r)/2.0
        
        if cur_vehicle_off < 0:
            # line shifted to the right, mean camera/vehicle center in shift to the left
            cur_direction = 'left of center'
        else:
            # line shifted to the left, mean camera/vehicle center in shift to the right
            cur_direction = 'right of center'
           
        ####
        ##                SANITY CHECK
        ####
        curve, gap, parallel, ret_sts = SanityCheck(LeftLine, RightLine, left_fit, right_fit, left_curverad, right_curverad, img_size, self.xm_per_pixel, self.ym_per_pixel)
        self.curve.append(min(curve,3000.0))
        self.gap.append(gap)
        self.parallel.append(parallel)
        self.det_sts.append(ret_sts)
        
        ####
        ##                RECORD DATA 
        ####
        
        # If sanity check is pass, record this line otherwise skip this one and use the old best fit data + request restart if necessary
        # So bad detection won't pass through the best fit
        if ret_sts:
            # Record line data
            RecordLineHistory(self.LeftLine, found_line[0], basepoint_x[0], line_base_pose_l, left_fit,  left_curverad,  left_pix)
            RecordLineHistory(self.RightLine,found_line[1], basepoint_x[1], line_base_pose_r, right_fit, right_curverad, right_pix)
            # Reset bad detection in a row to 0
            self.badDetection = 0
        else:
            # do we have bad n-detection in row if yes  # request new restart and use the old fit data for this image
            self.badDetection +=1
            if(self.badDetection > self.badDetectionThresh):
                # request restart if seen bad n detection in a row
                self.restart = True
        
        ####
        ##               AVERAGING OUTPUT RESULTS AND DISPLAY 
        ####
        
        # Average/Iterate the result from previous detections for a stable result
        # n_iterate Curvature - given best fit to compute curvature
        left_curverad_n  = ComputeLineCurvature(None, None, y_eval, self.xm_per_pixel, self.ym_per_pixel, self.LeftLine.best_fit)
        right_curverad_n = ComputeLineCurvature(None, None, y_eval, self.xm_per_pixel, self.ym_per_pixel, self.RightLine.best_fit)
        avg_curverad_n = (left_curverad_n+right_curverad_n)/2.0
        
        # n_iterate Vehicle position - given best base point to compute off center positioning
        avg_base_pose_l = self.LeftLine.bestx   # pixel distance of the image center from the left line (base)
        avg_base_pose_r = self.RightLine.bestx  # pixel distance of the iamge center from the right line (base)
        avg_vehicle_off = self.xm_per_pixel*(img_undist.shape[1] - avg_base_pose_l - avg_base_pose_r)/2.0
        
        if avg_vehicle_off < 0:
            # line shifted to the right, mean camera/vehicle center in shift to the left
            avg_direction = 'left of center'
        else:
            # line shifted to the left, mean camera/vehicle center in shift to the right
            avg_direction = 'right of center'
        
        
        # Project the warped image back to the original image
        # Display current 'live' or average best fit data
        if self.mode == 'live':
            final_result = InverseProjectionImage(bin_warped, img_undist, img_size, left_fit, right_fit, left_pix, right_pix, self.Minv, plot_result=False)
        else:
            final_result = InverseProjectionImage(bin_warped, img_undist, img_size, self.LeftLine.best_fit, self.RightLine.best_fit, left_pix, right_pix, self.Minv, plot_result=False)
        
        
        # Display average result on the image
        x = 10
        y = 60
        size = 1.3
        color = (255,255,255)
        thickness = 2
        final_result = DrawText(final_result,x,y,size,color,thickness,avg_curverad_n, abs(avg_vehicle_off), avg_direction)
          
        return final_result
        
        
if __name__ == '__main__':
    ym_per_pixel = 30.0/720                      # meters per pixel in y dimension (720 pixels/30 meter)
    xm_per_pixel = 3.7/700                       # meters per pixel in x dimension (700 pixels/3.7 meter)
    line_iteration = 5                           # 5 iterations
    
    # line data container
    LeftLine  = Line(line_iteration)
    RightLine = Line(line_iteration)
    
    # Read in the camera calibration and Warp forward and backward transformation
    dist_pickle = pickle.load( open( "dist_camera.pkl", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    M = dist_pickle["M"]
    Minv = dist_pickle["Minv"]
    
    # [kernel, min thresh, max thresh]
    gradx  = [3,20,150]    #[3,30,150] 
    grady  = [3,20,150]    #[5,30,150]
    mag    = [5,40,150]    #[5,20,150] 
    direct = [15,0.8,1.2]  #[15,0.5,1.3]
    Sobel  = [gradx, grady, mag, direct]
    
    # Color space thresholding
    hls_thresh = [(15,80),(120,230),(120,230)]
    rgb_thresh = [(220,255),(None,None),(None,None)]
    
    # Mask region of interest
    # Mask region of interest
    vertices = np.array([[    (100, 690),
                              (575, 430), 
                              (724, 430),
                              (1270, 690)
                              ]],dtype=np.int32)
    
    # Lane detection instance
    display_mode = 'average'                        # 'live': display current line,  'average': n-iterations display
    LaneDetection = AdvanceLaneDetection(xm_per_pixel,ym_per_pixel,line_iteration,mtx,dist,M,Minv,\
                                         Sobel,hls_thresh,rgb_thresh, vertices, display_mode)
    
    
    
    # Test image sequence
    #test_images = glob.glob('./mytest_images/Pictures*.jpg')
    test_images = glob.glob('./test_images/test*.jpg')
    #sort file in order test01,test02,test03.....
    test_images = np.sort(test_images)
    for fname in test_images:
        print("Image: ", fname)
        img = mpimg.imread(fname)
        final_img = LaneDetection.ProcessorPipline(img)
        # save to local disk
        out_folder = './test_images/test_output/'
        out_path   = out_folder + 'out_' + fname.split('/')[-1]
        mpimg.imsave(out_path, final_img, format='jpg')
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.title("diff L/R curve")
    plt.plot(LaneDetection.curve)
    plt.subplot(2,2,2)
    plt.title("mean gap")
    plt.plot(LaneDetection.gap)
    plt.subplot(2,2,3)
    plt.title("parallelness")
    plt.plot(LaneDetection.parallel)
    plt.subplot(2,2,4)
    plt.title("all conditions")
    plt.plot(LaneDetection.det_sts)
    plt.show()
    
    
    '''
    # Video processing
    drive_output = 'project_video_out.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    drive_clip = clip1.fl_image(LaneDetection.ProcessorPipline) #NOTE: this function expects color images!!
    drive_clip.write_videofile(drive_output, audio=False)
    '''
    
    
    
    
    
    
    
        
        
        
    
      
        
     

        
        
        
        
        
          
        
        
    
    
    