import os 
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from feature_detection_helper import * 



def hough_space_n_lines(image, n, 
                        smooth = False, kernel_size = 3, 
                        non_max_suppression = False,
                        double_thr_lower = 0.3, double_thr_upper = 0.4):
    
    '''
    Gets and input image as numpy array and the number
    of desired lines n
    
    smooth - Boolean, identifies whether to apply Gaussian kernel for image smoothing
    kernel_size - Integer, the size of applied Gaussian kernel
    
    non_max_suppression - Boolean, identifies whether to apply Non-Maximum Suppression algorithm
    
    double_thr_lower - Float, the lower value of double threshold coefficient
    double_thr_upper - Float, the upper value of double threshold coefficient
    Both are responsible for the amount of pixels that are considered sufficient after Double Threshold algorithm iteration
    
    
    
    Returns the list of lines, each containing four coordinates and
    the accumulator of Hough space.
    '''

    original_image = image.copy()
    
    # Apply Gaussian kernel to smoothe image edges
    if(smooth):
        print("Applying Gaussian kernel of size %d x %d..." % (kernel_size, kernel_size))
        image = convolve_kernel(image, gaussian_kernel(kernel_size))
        
    # Find the gradients of the image
    print("Calculating gradients of the image...")
    grad_matrix, theta_matrix = sobel_filters(image)
    
    # Apply max suppression algorithm for filtering unnecessary pixels
    if(non_max_suppression):
        print("Applying Non-Max Suppression algorithm...")
        image = non_max_suppression(grad_matrix, theta_matrix)
        
    # Apply double threshold algorithm to identify weak and strong pixels
    print("Applying Double Threshold algorithm...")
    threshold_image, weak_pixels, strong_pixels = double_threshold(grad_matrix, double_thr_lower, double_thr_upper)
    
    print("Applying Hysteresys algorithm")
    image = hysteresis(threshold_image, weak_pixels, strong_pixels)
        
    print("Calculating the Hough Space accumulator...")
    accumulator, rhos, thetas = hough_lines_acc(image)
    
    # Plot the Hough Space lines
    plot_hough_lines(original_image, image, accumulator, rhos, thetas)
    
    hough_acc_peaks = hough_peaks(accumulator, n)
    lines_list = hough_lines_transofrm(hough_acc_peaks, rhos, thetas, n)
    
    return lines_list, accumulator
    

def hough_lines_acc(img, angle_step=1, value_threshold=5):
    '''
    Takes an input image as numpy 2d-array
    angle_step is the step between angles of thetas (-90, +90)
    value_threshold is the value of pixels to be detected as edges
    NOTE: It's better to use this method after some preliminary image
    pre-processing
    returns accumulator - a numpy 2d-array of hough trans. accumulation
    rhos - array of rho values of the lines
    thetas - array of angles computed
    '''
    
    """
    Hough transform for lines
    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    lines_are_white - boolean indicating whether lines to be detected are white
    value_threshold - Pixel values above or below the value_threshold are edges
    Returns:
    accumulator - 2D array of the hough transform accumulator
    theta - array of angles used in computation, in radians.
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image.
    """
    
    width, height = img.shape
    hypot = int(np.hypot(width, height))
    
    # Rho and Theta arrays
    thetas = np.deg2rad(np.arange(0, np.pi, angle_step))
    rhos = np.linspace(-hypot, hypot, hypot * 2)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
                
    num_thetas = len(thetas)
    num_rhos = len(rhos)

    accumulator = np.zeros((2 * hypot, num_thetas), dtype=np.uint16)
                
    edges = img > value_threshold 
    y_id, x_id = np.nonzero(edges)

    # Accumulator voting
    for i in range(len(x_id)):
        x = x_id[i]
        y = y_id[i]

        for t_idx in range(num_thetas):
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + hypot
                
            # Add one vote for prominent line
            accumulator[rho, t_idx] += 1

    return accumulator, rhos, thetas


def hough_peaks(accumulator, n):
    '''
    Gets the accumulator and the number of lines
    and returns the most prominent n-lines
    NOTE: The Hough Space method is very sensitive to
    any noise so carefully denoise image before
    applying Hough Space transformation
    '''
    indices =  np.argpartition(accumulator.flatten(), -2)[-n:]
    return np.vstack(np.unravel_index(indices, accumulator.shape)).T   

def plot_hough_lines(image, trans_image, accumulator, rhos, thetas):
    
    # Create output figure
    
    fig, ax = plt.subplots(1,3, figsize = (12,8))
    
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    
    ax[1].imshow(trans_image, cmap=plt.cm.gray)
    ax[1].set_title('Hough input image')
    
    ax[2].imshow(accumulator, cmap='jet', extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[2].set_aspect('equal', adjustable='box')
    ax[2].set_title('Hough transformation')
    ax[2].set_xlabel('Angles (degrees)')
    ax[2].set_ylabel('Distance (pixels)')
    
    plt.show()


def hough_lines_transofrm(indicies, rhos, thetas, n):
    '''
    Takes the indicies of most prominent lines
    rhos and thetas from hough_space function and returns the list
    of lines. Each lines is in format of x1,y2,x2,y2.
    '''
    lines = []
    
    for i in range(n):
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        
        x0 = a*rho
        y0 = b*rho
        
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*a)
        
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        
        line_cartesian = {'a': a, 'b': b}
        line_coords = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
        
        lines.append((line_cartesian, line_coords))
    
    return lines


def plot_lines(img, lines):
    '''
    Inspired by OpenCV Python docs
    '''
    for line in lines:
        coords = line[1]
        cv2.line(img, (coords['x1'], coords['y1']), (coords['x2'], coords['y2']), (0, 255, 0), 2)
        
    cv2.imshow('Hough lines',img)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        