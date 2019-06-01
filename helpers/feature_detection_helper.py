import os 
import math
import numpy as np
import cv2
import matplotlib.image as mpimg

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def load_data(image_name):    
    img = mpimg.imread(image_name)
    img = rgb2gray(img)
    
    return img

def convolve_kernel(matrix, kernel):
    '''
    Performs a 2-d convolution between the
    input matrix(x,y) and kernel(s,t)
    
    Returns an convolved image(z, g)
    '''
    
    # Calculate the output image size
    
    x_max = matrix.shape[0]
    y_max = matrix.shape[1]
    
    # Get the kernel size
    
    s_max = kernel.shape[0]
    t_max = kernel.shape[1]
    
    s_mid = s_max // 2
    t_mid = t_max // 2
    
    # Define the size of output image
    
    z_max = x_max + 2 * s_mid
    g_max = y_max + 2 * t_mid
    
    # Create empty array which will be the output image
    
    output = np.zeros([z_max, g_max], dtype = np.float32)
    
    for z in range(z_max):
        for g in range(g_max):
            
            # Calculate the z,g pixel value for output image
            
            # Compute range for kernel values
            s_from_range = max(s_mid - z, -s_mid)
            s_to_range = min((z_max - z) - s_mid, s_mid + 1)
            
            t_from_range = max(t_mid - g, -t_mid)
            t_to_range = min((g_max - g) - t_mid, t_mid + 1)
            
            value = 0
            
            for s in range(s_from_range, s_to_range):
                for t in range(t_from_range, t_to_range):
                    x = z - s_mid + s
                    y = g - t_mid + t
                    value += kernel[s_mid - s, t_mid - t] * matrix[x, y]
                    
            output[z, g] = value
            
    return output        


def gaussian_kernel(kernel_size, sigma = 1):
    '''
    Create a Gaussian Kernel of given size
    for applying blur to image
    '''
    
    kernel_size = int(kernel_size) // 2
    x, y = np.mgrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1] 
    norm = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-(  (x**2 + y**2) / (2.0*sigma**2))) * norm
    
    return kernel


def sobel_filters(image):
    '''
    Calculates the gradient of the image by
    Sobel operators (edge detection operators)
    
    And returns the matrix G and array of thetas
    '''
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = convolve_kernel(image, Kx)
    Iy = convolve_kernel(image, Ky)
    
    g = np.hypot(Ix, Iy)
    g = g / g.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (g, theta)


def non_max_suppression(image, diagonal):
    '''
    Inputs an image and matrix of angles
    and outputs a matrix of thinned edges
    '''
    
    m, n = image.shape
    
    # Create zeros matrix of gradient size
    zer_proc = np.zeros((m,n), dtype = np.int32)
    
    # Calculate angles of the pixels
    angle = diagonal * 180. / np.pi
    
    # Supress zero angles
    angle[angle <0] += 180
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            try :
                # Get the values of next and previous pixel in direction
                next_pixel = image[i, j+1]
                previous_pixel = image[i, j-1]
                
                #0°
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    next_pixel = image[i, j+1]
                    previous_pixel = image[i, j-1]
                    
                #45°
                elif (22.5 <= angle[i,j] < 67.5):
                    next_pixel = image[i+1, j-1]
                    previous_pixel = image[i-1, j+1]
                    
                #90°
                elif (67.5 <= angle[i,j] < 112.5):
                    next_pixel = image[i+1, j]
                    previous_pixel = image[i-1, j]
                    
                #135°
                elif (112.5 <= angle[i,j] < 157.5):
                    next_pixel = image[i-1, j-1]
                    previous_pixel = image[i+1, j+1]

                    
                if (image[i,j] >= next_pixel) and (image[i,j] >= previous_pixel):
                    zer_proc[i,j] = image[i,j]
                else:
                    zer_proc[i,j] = 0
                
            except IndexError:
                pass
            
    return zer_proc
                

def double_threshold(image, low_threshold_ratio, high_threshold_ratio):
    '''
    Applies double threshold algorithm to image with
    given ratio and outputs processed image, weak pixels and strong pixels
    Weak pixels and strong pixels could be used in Hysteresis edge tracking 
    or ommited
    '''
    
    # Calculate the threshold intensity values for image
    high_threshold = image.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio
    
    # Create resulting matrix - the same size as previously processed image
    m,n = image.shape
    res_matrix = np.zeros((m,n), dtype = np.int32)
    
    # Create matrices for weak and strong pixels
    weak_pixels = np.int32(25)
    strong_pixels = np.int32(255)
    
    # Find the indexes for corresponding pixels
    i_str, j_str = np.where(image >= high_threshold)
    i_low, j_low = np.where(image < low_threshold)
    i_weak, j_weak = np.where((image <= high_threshold) & (image >= low_threshold)) 
    
    res_matrix[i_str, j_str] = strong_pixels
    res_matrix[i_weak, j_weak] = weak_pixels
    
    return (res_matrix, weak_pixels, strong_pixels)   



def hysteresis(image, weak_pixels, strong_pixels):
    '''
    Gets an input image and checks pixel-by-pixel
    all surrounding pixels to be strong
    
    Outputs an converted image
    '''
    
    m,n = image.shape
    
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if(image[i,j] == weak_pixels):
                try:
                    # Check all possible options for surrounding pixels to be strong
                    if ((image[i+1, j-1] == strong_pixels) or 
                        (image[i+1, j] == strong_pixels) or 
                        (image[i+1, j+1] == strong_pixels) or 
                        (image[i, j-1] == strong_pixels) or 
                        (image[i, j+1] == strong_pixels) or 
                        (image[i-1, j-1] == strong_pixels) or 
                        (image[i-1, j] == strong_pixels) or
                        (image[i-1, j+1] == strong_pixels)):
                         
                        image[i, j] = strong_pixels
                    else:
                        image[i, j] = 0
                except IndexError:
                    pass
                
    return image  