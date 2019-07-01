import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt


# SSD

def template_match_ssd(image, patch):
    """
    Performs template matching usind Sum of Squared Differences
    
    Input:
    image - numpy (w, h) matrix of grayscaled image
    patch - numpy (k_w, k_h) matrix of grayscaled image patch
    
    Output:
    tuple of (p1, p0) of most prominent points of the matched patch
    """
    
    # Get the dimensions of the image and the patch
    h, w = image.shape
    k_h, k_w = patch.shape
    
    # Create matrix to store SSD between patch and sub-matrices of the image
    ssd_scores = np.zeros((h - k_h, w - k_w))
    
    # Iterate over the image and calculate SSD
    
    for i in range(0, h - k_h):
        for j in range(0, w - k_w):
            score = (image[i:i + k_h, j:j + k_w] - patch)**2
            ssd_scores[i, j] = score.sum()
            
    # Find the minimum points
    min_points = np.unravel_index(ssd_scores.argmin(), ssd_scores.shape)
    
    return (min_points[1], min_points[0])   


# NCC

def template_match_ncc(image, patch):
    """
    Performs template matching usind Normalized Cross Correlation
    
    Input:
    image - numpy (w, h) matrix of grayscaled image
    patch - numpy (k_w, k_h) matrix of grayscaled image patch
    
    Output:
    tuple of (p1, p0) of most prominent points of the matched patch
    """
    
    # Get the dimensions of the image and the patch
    h, w = image.shape
    k_h, k_w = patch.shape
    
    # Create matrix to store SSD between patch and sub-matrices of the image
    ssd_scores = np.zeros((h - k_h, w - k_w))
    
    # Convert matrices to arrays
    image = np.array(image, dtype="float")
    patch = np.array(patch, dtype="float")
    
    for i in range(0, h - k_h):
        for j in range(0, w - k_w):
            # Get submatrix from source image
            image_sub = image[i:i + k_h, j:j + k_w]
            # Cross-correlation computation
            numerator = np.sum(image_sub * patch)
            denominator = np.sqrt( (np.sum(image_sub ** 2))) * np.sqrt(np.sum(patch ** 2))
            
            if(denominator == 0):
                # To prevent ZeroDivisionError
                ssd_scores[i, j] = 0
            else:
                ssd_scores[i, j] = numerator / denominator
                
    # Find the minimum points
    min_points = np.unravel_index(ssd_scores.argmax(), ssd_scores.shape)
    
    return (min_points[1], min_points[0])   


# SAD


# General method 

def perform_template_matching(image_path, patch_path, method = 'ssd'):
    """
    Performs template matching using one of the methods and 
    draws a rectangle on the most promising part of image
    
    Input:
    image_path - string, absolute path to image frame
    patch_path - string, absolute path to patch
    method - string, one of the methods: ssd, ncc, sad.
    
    Output:
    file with drawed rectangle
    """

    # Colored image would be used for visualization
    full_image = cv2.imread(image_path)
    full_image = cv2.cvtColor(full_image,cv2.COLOR_BGR2RGB)
    full_image_gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    
    patch_image = cv2.imread(patch_path, 0)
    
    # Dimensions of the image would be used for drawing a rectangle
    h, w = patch_image.shape
    
    # Perform matching
    if(method == 'ssd'):
        points = template_match_ssd(full_image_gray, patch_image)
    elif(method == 'ncc'):
        points = template_match_ncc(full_image_gray, patch_image)
    elif(method == 'sad'):
        points = template_match_ncc(full_image_gray, patch_image)
    else:
        raise ValueError("Unknown template matching method. Supported methods: ssd, ncc, sad")
    
    
    # Draw a rectangle
    cv2.rectangle(full_image, (points[0], points[1] ), (points[0] + w, points[1] + h), (0, 0, 200), 3)
    plt.imshow(full_image)