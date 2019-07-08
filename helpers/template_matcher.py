import cv2
import numpy as np
import os 
import sys
import matplotlib.pyplot as plt

class TemplateMatcher:

    def __init__(self, video_file_name, method):
        """
        Setups values for all class
        """
        self.box_selected = False
        self.old_points = np.array([[]])
        self.boxes = []
        self.video_file_name = video_file_name
        self.selection_mode = True
        self.method = method

        # Setup constants for OpenCV
        # Default value that should be changed due to used method 
        self.windowName = "Content-based template matching" 


    def rectangle_select_callback(self, event, x, y, flags, params):
        """
        Callback for OpenCV rectangle emulation method. Listens for two
        key events - LMB pressed and released and computes the selected
        rectangle base on two points
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Clear the state of the boxes
            self.boxes = []
            self.box_selected = False
            self.boxes.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            print("Box selected")
            self.boxes.append((x, y))
            self.box_selected = True
            # Calculate the relative center of rectangle
            c_x, c_y = (self.boxes[0][0] + self.boxes[1][0])/2, (self.boxes[0][1] + self.boxes[1][1])/2
            # Set old points 
            self.old_points = np.array([[c_x, c_y]], dtype=np.float32)
            self.selection_mode = False
            self.match_image()


    def normalize_frame(self, frame):
        frameNormalized = np.zeros(frame.shape)
        frameNormalized = cv2.normalize(frame,  frameNormalized, 0, 255, cv2.NORM_MINMAX)

        return frameNormalized

    # Template-matching methods

    # SSD

    def template_match_ssd(self, image, patch):
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
        print("Image shape ", image.shape)
        k_h, k_w = patch.shape
        
        # Create matrix to store SSD between patch and sub-matrices of the image
        ssd_scores = np.zeros((h - k_h, w - k_w))

        # Convert matrices to arrays
        image = np.array(image, dtype="float")
        patch = np.array(patch, dtype="float")
        
        # Iterate over the image and calculate SSD
        
        for i in range(0, h - k_h):
            for j in range(0, w - k_w):
                score = (image[i:i + k_h, j:j + k_w] - patch)**2
                ssd_scores[i, j] = score.sum()
                
        # Find the minimum points
        min_points = np.unravel_index(ssd_scores.argmin(), ssd_scores.shape)
        
        return (min_points[1], min_points[0])   


    # NCC

    def template_match_ncc(self, image, patch):
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

    def template_match_sad(self, image, patch):
        """
        Performs template matching usind Sum of Absolute Differences
        
        Input:
        image - numpy (w, h) matrix of grayscaled image
        patch - numpy (k_w, k_h) matrix of grayscaled image patch
        
        Output:
        tuple of (p1, p0) of most prominent points of the matched patch
        """
            
        # Get the dimensions of the image and the patch
        h, w = image.shape
        k_h, k_w = patch.shape

        # Convert matrices to arrays
        image = np.array(image, dtype="float")
        patch = np.array(patch, dtype="float")
        
        # Create matrix to store SSD between patch and sub-matrices of the image
        ssd_scores = np.zeros((h - k_h, w - k_w))

        for i in range(0, h - k_h):
            for j in range(0, w - k_w):
                score = np.abs(image[i:i + k_h, j:j + k_w] - patch)
                ssd_scores[i, j] = score.sum()

        # Find the minimum points
        min_points = np.unravel_index(ssd_scores.argmin(), ssd_scores.shape)
        
        return (min_points[1], min_points[0]) 
        

    # General method 

    def perform_template_matching(self, image, patch):
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
        full_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        
        # Dimensions of the image would be used for drawing a rectangle
        patch_image_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        print("Patch shape ", patch.shape)
        h, w = patch_image_gray.shape
        
        # Perform matching
        if(self.method == 'ssd'):
            print("Using SSD method")
            points = self.template_match_ssd(full_image_gray, patch_image_gray)
        elif(self.method == 'ncc'):
            print("Using NCC method")
            points = self.template_match_ncc(full_image_gray, patch_image_gray)
        elif(self.method == 'sad'):
            print("Using SAD method")
            points = self.template_match_sad(full_image_gray, patch_image_gray)
        else:
            raise ValueError("Unknown template matching method. Supported methods: ssd, ncc, sad")
        
        
        # Draw a rectangle
        cv2.rectangle(image, (points[0], points[1] ), (points[0] + w, points[1] + h), (0, 0, 200), 3)
        
        # Return the image with the rectangle
        return image


    def init_detection(self):
        # Create video capture
        if(self.video_file_name == None):
            # Capture video from webcam
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.video_file_name)

        self.cap = cap

        # Read first frame
        ret, frame = cap.read()
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.old_gray = self.normalize_frame(self.old_gray)

        # Set callback for window mouse click
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, 640, 480)
        cv2.setMouseCallback(self.windowName, self.rectangle_select_callback)

        while(self.selection_mode):
            cv2.imshow(self.windowName, frame)
            cv2.startWindowThread()

            if ret is not True:
                break

            key = cv2.waitKey(50)

            if key == 27:
                break


    def match_image(self):
        """
        Performs template-matching using the desired method in 
        """
        # Loop over video frames
        while True:
            # Get the new frame
            _, frame = self.cap.read()
            print("Next frame read")

            if self.box_selected is True:
                # Draw a selected rectangle on the image
                cv2.rectangle(frame, self.boxes[0], self.boxes[1], (255, 0, 0), 2)
                # Define the region of interest
                self.roi = frame[self.boxes[-2][1]:self.boxes[-1][1], self.boxes[-2][0]:self.boxes[-1][0]]
                print("Roi selected with shape ", self.roi.shape)

            
            print("Performing template matching...")
            # Perform template-matching 
            frame = self.perform_template_matching(frame, self.roi)
            print("Template matching performed.")

            cv2.imshow(self.windowName, frame)
            cv2.startWindowThread()

            key = cv2.waitKey(45)

            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Get the name of video file name
    if(len(sys.argv) > 1):
        video_file_name = sys.argv[1]
        method = sys.argv[2]
    else:
        video_file_name = None
        method = 'ssd'

    helper = TemplateMatcher(video_file_name, method)
    helper.init_detection()

        