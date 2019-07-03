import cv2 
import numpy as np
import sys

class MeanShiftTracker:

    def __init__(self, video_source, iteration = 10, points = 1):
        self.video_source = video_source
        self.window_name = "Mean-Shift tracking algorithm"
        self.selection_mode = True
        self.box_selected = False
        self.roi_selected = False
        self.term_criteria = (cv2.TERM_CRITERIA_EPS or cv2.TERM_CRITERIA_COUNT, iteration, points)
        self.init_detection()

    
    def init_detection(self):
        if self.video_source is None:
            self.cap = cv2.VideoCapture(0) # Use webcam as source
        else:
            self.cap = cv2.VideoCapture(self.video_source)

        # Read first frame
        ret, frame = self.cap.read()
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Set callback for window mouse click
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)
        cv2.setMouseCallback(self.window_name, self.rectangle_select_callback)

        while(self.selection_mode):
            cv2.imshow(self.window_name, frame)
            cv2.startWindowThread()

            if ret is not True:
                break

            key = cv2.waitKey(50)

            if key == 27:
                break

    def rectangle_select_callback(self, event, x, y, flags, params):
        """
        Updates the state of drawed rectangle
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Clear the state of the boxes
            self.boxes = []
            self.box_selected = False
            self.roi_selected = False
            self.boxes.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            self.boxes.append((x, y))
            self.box_selected = True
            # Set old points 
            self.selection_mode = False
            self.start_detection()
        

    def start_detection(self):
        while True:
            ret, frame = self.cap.read()

            # Break the loop if we don't have any frames left
            if not ret:
                break

            if self.video_source is None:
                key = cv2.waitKey(1)
            else:
                 key = cv2.waitKey(50)
        
            if self.box_selected is True:
                # Draw a selected rectangle on the image
                cv2.rectangle(frame, self.boxes[0], self.boxes[1], (255, 0, 0), 2)

                if not self.roi_selected:
                    # Define the region of interest
                    self.roi = frame[self.boxes[-2][1]:self.boxes[-1][1], self.boxes[-2][0]:self.boxes[-1][0]]
                    self.roi_selected = True

                    # Convert to HSV in order to process by mean-shift
                    self.roi_hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
                    # Get HSV histogram only from the hue
                    self.roi_hist = cv2.calcHist([self.roi_hsv], [0], None, [180], [0, 180])
                    # Normalize the histogram
                    self.roi_hist = cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)


            # Convert frame to hsv
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Calculate the mask for mean-shift algorithm
            mask = cv2.calcBackProject([frame_hsv], [0], self.roi_hist, [0, 180], 1)

            _, window = cv2.meanShift(mask, (self.boxes[0][0], self.boxes[0][1], self.roi.shape[0], self.roi.shape[1]), self.term_criteria)
            

            x, y, w, h = window
            cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)

            cv2.imshow(self.window_name, frame)
            cv2.imshow("Mask", mask)

            if key == 27:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    # Get the name of video file name
    if(len(sys.argv) > 1):
        video_file_name = sys.argv[1]
    else:
        video_file_name = None

    mean_shift_tracker = MeanShiftTracker(video_file_name)

