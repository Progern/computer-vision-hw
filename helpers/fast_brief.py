from matplotlib import pyplot as plt
from feature_detection_helper import *

"""
Helper script for performing FAST corner detection
and BRIEF descriptor.
"""

# FAST

    # Helper methods

def bresenham_circle_points(x, y):
    '''
    Return a list of tuples each of them corresponds to 
    a points in the Bresenham circle algorithm around the
    pixel of interest.
    '''
    
    p1 = (x+3, y)
    p3 = (x+3, y-1)
    p5 = (x+1, y+3)
    p7 = (x-1, y+3)
    p9 = (x-3, y)
    p11 = (x-3, y-1)
    p13 = (x+1, y-3)
    p15 = (x-1, y-3)
    
    return [p1, p3, p5, p7, p9, p11, p13, p15]

    # Inspiration for these methods was taken from Tibliu
    # Blur methods

def insert_sort(lst):
    '''
    Performs and insertion sort into the window 
    if values. Specific version for median kernel blur.
    Inspired by Tibliu.
    '''
    for index in range(1, len(lst)):
        currentvalue = lst[index]
        position = index

        while position > 0 and lst[position - 1] > currentvalue:
            lst[position] = lst[position - 1]
            position = position - 1

        lst[position] = currentvalue

def median_kernel_blur(image, x0, y0, x1, y1, n = 3):
    '''
    Applies median blur on an image patch (x0, y0) - (x1,y1) to remove salt and pepper noise.
    Median blur replaces each pixel with the median of the NxN pixels surrounding it.
    
    The salt and pepper noise is the main reason of bad corner detection.
    '''
    
    dst = image[:] 
    for y in range(x0, y0):
        for x in range(x1, y1):
            window = []
            for i in range(y - n // 2, y + n // 2 + 1):
                for j in range(x - n//2, x + n//2 + 1):
                    window.append(image[i][j])
            insert_sort(window)
            dst[y][x] = window[len(window)//2]

    return dst


    # Corner detection

def is_corner(image, x, y, bresenham_circle, threshold):
    '''
    Identifies is the pixel at (x,y) is a corner by 
    comparing with corresponding pixels in bresenham_circle
    and the provided threshold
    '''
    
    # Interest pixel indensity
    Ip = int(image[x,y])
    
    # Get the values of corresponding Bresenham circle pixels
    
    x1, y1 = bresenham_circle[0]
    x5, y5 = bresenham_circle[2]
    x9, y9 = bresenham_circle[4]
    x13, y13 = bresenham_circle[6]
    
    # Define the count of cardinal pixels that are brighter/darker by threshold value than the (x,y)
    count = 0
    
    # Get the intensities of corresponding Bresenham cardinal direction pixels
    try:
        I1 = int(image[x1, y1])
        I5 = int(image[x5, y5])
        I9 = int(image[x9, y9])
        I13 = int(image[x13, y13])
        
        # Check for particular differences in intensity
    
        if abs(I1 - Ip) > threshold:
            count += 1 
        if abs(I5 - Ip) > threshold:
            count += 1
        if abs(I9 - Ip) > threshold:
            count += 1
        if abs(I13 - Ip) > threshold:
            count += 1

    except:
        pass
    
    return count >= 3

    # Non-Maximum suppression

def suppress_score(image, p, bresenham_circle):
    '''
    Calculates the non-max suppression score for a given 
    pixel p(x,y) with his oreol circle.
    
    Returns the sum of absolute differences between the point of
    interest and surrounding pixel values.
    '''
    
    x, y = p
    # Intensity of interest pixel p
    Ip = int(image[x,y])
    
    # Get the coordinates of surrounding Bresenham pixels
    
    x1, y1 = bresenham_circle[0]
    x3, y3 = bresenham_circle[1]
    x5, y5 = bresenham_circle[2]
    x7, y7 = bresenham_circle[3]
    x9, y9 = bresenham_circle[4]
    x11, y11 = bresenham_circle[5]
    x13, y13 = bresenham_circle[6]
    x15, y15 = bresenham_circle[7]
    
    # Get the corresponding intensity values of surrounding Bresenham pixels
    I1 = int(image[x1, y1])
    I3 = int(image[x3, y3])
    I5 = int(image[x5, y5])
    I7 = int(image[x7, y7])
    I9 = int(image[x9, y9])
    I11 = int(image[x11, y11])
    I13 = int(image[x13, y13])
    I15 = int(image[x15, y15]) 
    
    score = abs(Ip - I1) + abs(Ip - I3) + abs(Ip - I5) + abs(Ip - I7) + \
            abs(Ip - I9) + abs(Ip - I11) + abs(Ip - I13) + abs(Ip - I15)
        
    return score

def non_max_suppression(image, corners, bresenham_circle):
    '''
    Performs non-max suppression over the corners in-place.
    '''
    
    i = 1 # To prevent IndexError
    while i < len(corners):
        c_i = corners[i]
        c_i_prev = corners[i - 1]
        
        if are_adjacent(c_i, c_i_prev):
            c_i_score = suppress_score(image, c_i, bresenham_circle)
            c_i_prev_score = suppress_score(image, c_i_prev, bresenham_circle)
            
            if (c_i_score > c_i_prev_score):
                del(corners[i - 1])
            else:
                del(corners[i])
        else:
            i += 1
            continue
    return

def are_adjacent(p1, p2):
    '''
    Identifies if two points are adjacent by calculating distance in terms of x, y for borth
    Two points are adjacent if they are within four pixels of each other (Euclidean distance)
    
    Accepts two tuples of type (x,y)
    '''
    
    x1, y1 = p1
    x2, y2 = p2
    
    x_d = x1 - x2
    y_d = x2 - y2
    
    return (x_d ** 2 + y_d ** 2) ** 0.5 <= 4

    # Detection algorithm

def fast_detection(image, threshold = 50):
    '''
    Receives an image matrix of size [n,m] and performs
    the corner detection using FAST algorithm.
    
    Image should be a grayscale immage (one channel) of 
    tuple shape (n, m, c), where c = 1
    
    Returns the list of corners as tuples (x,y) coordinates
    '''
    
    # Initialize empty list of corners
    corners = []
    num_rows, num_cols = image.shape
    
    # Apply median blur
    
    # We should apply some padding from the edges of the mage 
    # Due to circular pixels search
    x0, y0 = int(num_rows * 0.05), int(num_rows * 0.95)
    x1, y1 = int(num_cols * 0.05), int(num_cols * 0.95)
    
    gauss_kernel = gaussian_kernel(3)
    image = convolve_kernel(image, gauss_kernel)
    
    for row in range(x0, y0):
        for col in range(x1, y1):
            # Build a Bresenham circle for each pixel
            bresenham_circle = bresenham_circle_points(row, col)
            
            if is_corner(image, row, col, bresenham_circle, threshold):
                corners.append((col, row))
    
    # Perform non-max suppression
    
    non_max_suppression(image, corners, bresenham_circle)
    
    return corners

# BRIEF

def compute_binary_vector(image, keypoint, n, patch_size, seed):
    '''
    Computes BRIEF binary vector for one keypoint.
    image - image matrix 
    keypoint - tuple of (x,y) coordinates
    n - size of output vector
    patch_size - size of patch square around keypoint 
    
    Returns vector of size n containing binary values.
    '''
    np.random.seed(seed)
    
    # Create empty array for storing binary values
    output_arr = np.zeros(n, dtype = np.int8)
    
    # Compute value of diagonal, assuming keypoint is in the middle of the patch
    x, y = keypoint
    diagonal = patch_size / 2
    
    # We need to check if the keypoint patch is in the bounds of the image
    x_max, y_max = image.shape
    
    # Check if our keypoint is not too near to the right edge 
    if((x - diagonal < 0) or (y - diagonal < 0) or (x + diagonal > x_max) or (y + diagonal > y_max)):
        return None
    
    # Coordinates of patch square   
    x0 = int(x - diagonal)
    y0 = int(y - diagonal)
    x1 = int(x + diagonal)
    y1 = int(y + diagonal)
    
    # Extract patch from image
    patch = image[x0:x1, y0:y1]
        
    for i in range(n):
        # Get intensity of keypoint
        Ip = image[x, y]
        
        # Get random uniform value for point in patch
        x_rand = int(np.random.uniform(low = x0, high = x1))
        y_rand = int(np.random.uniform(low = y0, high = y1))
        
        # Get intensity of random point
        Ir = image[x_rand, y_rand]
        
        if(Ip < Ir):
            output_arr[i] = 1
    
    return output_arr


def compute_descriptors(image, key_points, n = 128, patch_size = 32, seed = 10):
    '''
    Computes descriptor for each of the key points and
    
    image - image matrix
    key_points - array of key points calculated by some algorithm, for ex. FAST or SIFT
    n - size of descriptor vector
    patch_size - size of surrounding patch for each key point
    seed - random seed for numpy uniform probability distribution
    
    returns list of descriptor vectors
    '''
    
    descriptors = []
    
    for key_point in key_points:
        x, y = key_point[0], key_point[1]
        key_point_descriptor = compute_binary_vector(image, (x,y), n, patch_size, seed)
        
        # Check if key point was not rejected
        if(key_point_descriptor is not None):
            descriptors.append(key_point_descriptor)

    return descriptors

