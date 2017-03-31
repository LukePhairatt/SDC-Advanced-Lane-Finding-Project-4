import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mping
import glob
#%matplotlib.inline


def CameraCalibration(image_path, nx_target,ny_target, plot_result):
    # list images
    images = glob.glob(image_path)
    
    # Array to store corner data for each images for the calibration/undistortion
    objpoints = []
    imgpoints = []

    # Extract image corners (Note some image might failed)
    index = 0
    for fname in images:
        img = mping.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        name = fname.split('/')[-1]
        if name == 'calibration1.jpg':
            nx = 9
            ny = 5
            objp = np.zeros((nx*ny,3), np.float32)
            objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) 
            
        elif name == 'calibration5.jpg':
            nx = 7
            ny = 6
            objp = np.zeros((nx*ny,3), np.float32)
            objp[:,:2] = np.mgrid[2:9,0:6].T.reshape(-1,2) 
        else:
            # we are using the grid that has 9 points (along row axis) x 6 points (along col axis
            # 54 points in x,y,z=0 column (3)
            # Add x,y value of the object in the world space
            # z = 0 we don't change it
            nx = nx_target
            ny = ny_target
            objp = np.zeros((nx*ny,3), np.float32)
            objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)    
        
        ret, corners = cv2.findChessboardCorners(gray,(nx,ny), None)
        index+=1
        print("Read images: ", index,"Name: ", fname, "Found point: ", ret)
        
        # if coners found on this image append them
        if ret == True:
            imgpoints.append(corners) # 2D points x,y
            objpoints.append(objp)    # 3D points x,y,z = 0 same for all images!
            # draw image corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            plt.imshow(img)
            plt.show()
            
    # Do the calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Visualise correct distortion for each image
    if plot_result:
        for fname in images:
            img = mping.imread(fname)
            dst = cv2.undistort(img, mtx, dist, None, mtx)
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(img)
            plt.subplot(1,2,2)
            plt.imshow(dst)
            plt.show()
        
    return ret, mtx, dist, rvecs, tvecs  
    
# Compute a top view transformation from the grid image 
def ComputeTransform(img, mtx, dist, src_points, dst_points):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
        
    # Grab the image shape in to x, y coordinate here!
    img_size = (undist.shape[1], undist.shape[0])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    # Warp the image using OpenCV warpPerspective() img_size is x,y or col,row format
    warped = cv2.warpPerspective(undist, M, img_size)
    
    return undist, warped, M, Minv
    
def draw_lines(img, points, color=[255, 0, 0], thickness=2):
    num_points = len(points)
    for i in range(num_points):
        x1 = points[i][0]                   #current X point
        y1 = points[i][1]                   #current Y point
        x2 = points[(i+1)%num_points][0]    #next X point (round to zero index if exceed)
        y2 = points[(i+1)%num_points][1]    #next Y point (round to zero index if exceed)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

if __name__ == '__main__':
    # Camera calibration
    nx = 9
    ny = 6
    image_path = './camera_cal/calibration*.jpg'
    save_data = True
    plot_image = False
    ret, mtx, dist, rvecs, tvecs = CameraCalibration(image_path, nx, ny, plot_image)
    
    # Compute perspective transformation
    img = mping.imread('./test_images/straight_lines1.jpg')
    
    # Hard code X,Y points
    # start bottom left the go CW
    # Adjusting src points to get the best possible result
    src = np.float32([  [203,img.shape[0]],
                        [580,460], 
                        [700,460], 
                        [1100,img.shape[0]]])
    
    dst = np.float32([  [320,img.shape[0]],
                        [320,0], 
                        [960,0], 
                        [960,img.shape[0]]])
    
    
    undist_img, warped_img, M, Minv = ComputeTransform(img, mtx, dist, src, dst)
    # overlay lines to both images
    plt.figure()
    plt.subplot(1,2,1)
    draw_lines(undist_img, src)
    plt.imshow(undist_img)
    plt.title("Undistorted Image")
    plt.subplot(1,2,2)
    draw_lines(warped_img, dst)
    plt.imshow(warped_img)
    plt.title("Undistorted Warped Image")
    plt.show()
    
    # Save calibration data for later use
    if save_data:
        dist_camera = {'mtx':mtx, 'dist':dist, 'rvecs':rvecs, 'tvecs':tvecs, 'M':M, 'Minv':Minv}
        with open('dist_camera.pkl','wb') as f:
            pickle.dump(dist_camera,f)
            print('Data save to local disk....')
