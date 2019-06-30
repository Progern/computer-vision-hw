import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt
import math


# File operation methods

def read_image(img_name, dataset_dir_path):
    return cv2.imread(get_img_abs_path(img_name, dataset_dir_path))

def get_img_abs_path(img_name, dataset_dir_path):
    return os.path.join(dataset_dir_path, img_name)


# Color histogram extraction

def color_hist(image):
    """
    Creates a color histogram of the input image
    
    Input:
    image - matrix, input image matrix
    
    Output:
    hist - numpy array consisting of 256 values, each will indicate
    the number of occurences of pixel intensity
    
    TODO: Grayscale images
    """
    
    hist = np.zeros(256)
    
    flat_img = image.ravel()
    
    # Iterate over each pixel
    for pixel in flat_img:
        # Increment the occurence of pixel value
        hist[pixel] += 1
    
    return hist

# L2 distance measure

def l2_dist(hist1, hist2):
    """
    Computes L2 distance between two numpy vectors/matrices
    
    Input:
    hist1, hist2 - numpy vectors/matrices to compute distance between
    
    Output:
    distance - scalar value showing distance between two vectors/matrices
    """
    distance = np.sum(np.abs(hist1 - hist2))
    
    return distance

# Multi-images color histogram calculation

def compute_color_hist_for_all_images(images_path, log = True):
    """
    Computes the color histogram for all images from the data folder
    and returns two arrays image_names, image_names_color_hists
    
    Input:
    images_path - string, absolute path to images directory
    log - boolean, whether to print how many images already processed
    """
        
    # Variable for indicating the current processed image
    processed_image = 0
    
    # List of image names
    image_names_list = os.listdir(images_path)
    image_names_len = len(image_names_list)
    
    # Create two empty lists for image names and corresponding histograms
    image_names = []
    color_hists = []
    

    # Iterate over each image
    for index, image_name in enumerate(image_names_list):
    
        # Set the corresponding values in arrays
        image_names.append(image_name)
        color_hists.append(color_hist(read_image(image_name)))
        
        processed_image += 1
        if(log and (processed_image % 100 == 0)):
            print("Processed %d of %d images" % (processed_image, image_names_len))     
        
        
    return image_names, color_hists


# Image correspondense querying

def compute_query_for_image(query_image_name, image_names, image_colors_hists, k = 10):
    """
    Computes L2 distance between query_image_name and each of image_names entries
    
    Input:
    query_image_name - string, name of the image to be queried on
    image_names - list, list of image names
    image_colors_hists - list, list of corresponding to image_names entries color histograms by index
    k - int, indicates how many best matches should be searched for query_image_name
    
    Output:
    (query_image_name, k_best_matches_names) - tuple, best_matches is a list containing names of k-best matched images
    
    """
    
    # Empty array to store l2 distances from target image to all other images
    l2_distances = np.zeros(len(image_names))
    
    # Target image color histogram
    target_hist = color_hist(read_image(query_image_name))
    
    
    # Iterate over each of the image_color_hists entries
    for index, image_hist in enumerate(image_colors_hists):
        l2_distances[index] = l2_dist(target_hist, image_hist) 
        
        # A small hack to remove self-duplicate
        if(l2_distances[index] == 0):
            l2_distances[index] = 10000000
        
    
    # Find k-indicies of the best matches
    k_best_matches = np.argpartition(l2_distances, k)[:k]
    
    # Get names of k-best matched indicies
    k_best_matches_names = [image_names[i] for i in k_best_matches]
    
    return (query_image_name, k_best_matches_names)  


def plot_target_query_and_results(query_image_test_name, k_best_test_matches, columns = 4, rows = 3):
    """
    Plots the target query image alongside with k-best image matches
    
    Input:
    query_image_test_name - string, name of the target query image file
    k_best_test_matches - list, containing string names of best k-matches
    columns, rows - int, adjustable parameters of figure subplots location
    
    Output:
    Displays a figure with subplots.
    """
    
    # Create figure for plotting several images
    fig=plt.figure(figsize=(7, 7))
    
    # Add query image
    fig.add_subplot(rows, columns, 1)
    plt.axis('off')
    plt.title("Target image", fontsize = 10)
    query_image = read_image(query_image_test_name)
    b,g,r = cv2.split(query_image)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)
    
    # Add all best k-matches images
    for i in range(len(k_best_test_matches)):
        img = read_image(k_best_test_matches[i])
        b,g,r = cv2.split(img)       # get b,g,r
        rgb_img = cv2.merge([r,g,b])     # switch it to rgb
        fig.add_subplot(rows, columns, i+2)
        plt.axis('off')
        plt.title(k_best_test_matches[i], fontsize = 10)
        plt.imshow(rgb_img)
        
    plt.show()


def compute_k_best_hist_matches_and_plot(query_image_name, image_names, image_colors_hists, k = 10):
    """
    Function that combines the call for compute_query_for_image and plot_target_query_and_results
    """
    
    img_query, k_best_matches = compute_query_for_image(query_image_name, image_names, image_colors_hists)
    plot_target_query_and_results(img_query, k_best_matches)
    
    return img_query, k_best_matches


# Precision-Recall (PR) calculation 

def get_image_number(image_name):
    """
    Clears the image filename from the characters and returns integer number
    
    Input: 
    image_name - string, corresponding file name
    
    Output:
    int, corresponding to image number
    """
    return int(image_name.replace('ukbench', '').replace('.jpg', ''))


def is_relevant_image(target_image_name, image_name):
    """
    Finds the difference between two image numbers. 
    Our dataset has relevant images sequentially by fourplexes so we could count
    the absolute difference.
    
    Input:
    target_image_name - string, target query image name
    image_name - string, top-match image to compute the difference
    
    Output:
    boolean, indicates whether the queried image is in the sequence with other image
    
    WARNING: This will only work for the target images that are first of the sequence and
    can be divided by 4 without any remainder
    """
    
    target_img_num = get_image_number(target_image_name)
    img_num = get_image_number(image_name)
    
    if(target_img_num % 4 == 0):
        # Means our target image is first of the sequence of the images
        return (target_img_num - img_num == -3 
                or target_img_num - img_num == -2 
                or target_img_num - img_num == -1) 
    
    # For now return false otherwise
    return False


def precision_recall_for_query(img_query, k_best_matches):
    """
    Counts the number of relevant true images found for query image
    
    Input:
    img_query - string, filename of query target image
    k_best_matches = list of strings, containing filenames of best matches for target query image
    
    Output:
    tuple in format (image_file_name, precision_value, recall_value)
    """
    relevant_images_count = 0.0
    k = len(k_best_matches)
    
    for match in k_best_matches:
        if(is_relevant_image(img_query, match)):
            relevant_images_count += 1
            
    
    if(relevant_images_count == 0.0):
        print("No relevant images found for %s" % img_query)
        return (img_query, 0, 0)
    else:
        print("Number of relevant images found %d/3" % relevant_images_count)
        # Count precision as retrieved_relevant_images / retrieved_relevant_images + other_retrieved_images
        precision = float(relevant_images_count / k)
    
        # Count recall as retrieved_relevant_images / retrieved_relevant_images + not_retrieved_relevant_images
        recall = relevant_images_count / (relevant_images_count + (3 - relevant_images_count))
    
        print("Precision: %.2f\nRecall: %.2f" % (precision, recall))
        
        return (img_query, precision, recall)


def plot_precision_recall(precision_recall_list):
    """
    Plots the P-R for all queries provided
    
    Input:
    precision_recall_list - list of tuples in view (image_file_name, precision_value, recall_value)
    
    Output:
    P-R plot for all queries
    """
    
    fig, ax = plt.subplots()
    
    for value in precision_recall_list:
        ax.scatter(value[1], value[2], label = value[0], alpha = 0.5)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3)
    ax.grid(True)
    plt.title('P-R plot for image queries')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.show()
        