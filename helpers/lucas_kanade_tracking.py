import numpy as np
import cv2
import sys


class LucasKanadeObjectTracker:

    def __init__(self, video_file_name = None, lk_window_size = (15, 15)):
        # Setup initial point state
        self.box_selected = False
        self.old_points = np.array([[]])
        self.boxes = []
        self.windowSize = lk_window_size
        self.video_file_name = video_file_name
        self.selection_mode = True

        # Setup constants for OpenCV
        self.windowName = "Pyramidal Lucas-Kanade tracking."

        # The maxLevel describes maximal pyramid level number. If set to 1 - two pyramids are used and so on
        self.lukas_kanade_params = {"winSize": lk_window_size, "maxLevel": 0,
                                    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

    def rectangle_select_callback(self, event, x, y, flags, params):
        """
        Updates the state of drawed rectangle
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Clear the state of the boxes
            self.boxes = []
            self.box_selected = False
            self.boxes.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            self.boxes.append((x, y))
            self.box_selected = True
            # Calculate the relative center of rectangle
            c_x, c_y = (self.boxes[0][0] + self.boxes[1][0])/2, (self.boxes[0][1] + self.boxes[1][1])/2
            # Set old points 
            self.old_points = np.array([[c_x, c_y]], dtype=np.float32)
            self.selection_mode = False
            self.start_detection()

    def normalize_frame(self, frame):
        lmin = float(frame.min())
        lmax = float(frame.max())
        return np.floor((frame - lmin) / (lmax - lmin) * 255.)

    def init_detection(self):
        # Create video capture
        if(self.video_file_name == None):
            # Capture video from webcam
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.video_file_name)

        self.cap = cap

        # Read first frame
        _, frame = cap.read()
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # Set callback for window mouse click
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, 640, 480)
        cv2.setMouseCallback(self.windowName, self.rectangle_select_callback)

        while(self.selection_mode):
            cv2.imshow(self.windowName, frame)
            cv2.startWindowThread()

            key = cv2.waitKey(45)

            if key == 27:
                break
        


    def start_detection(self):
        # Loop over video frames
        while True:
            # Get the new frame
            _, frame = self.cap.read()
            self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.box_selected is True:
                # Draw a selected rectangle on the image
                cv2.rectangle(frame, self.boxes[0], self.boxes[1], (255, 0, 0), 2)

                # Define the region of interest
                roi = frame[self.boxes[-2][1]:self.boxes[-1][1], self.boxes[-2][0]:self.boxes[-1][0]]

                # Update new window shape
                self.windowSize = roi.shape[0:2]

                # Update parameters of Lucas-Kanade method with new window size
                self.lukas_kanade_params = {"winSize":  self.windowSize, "maxLevel": 0,
                                    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
                
                # Get the position of new point using Lukas-Kanade tracking method
                new_points, status, error = cv2.calcOpticalFlowPyrLK(self.old_gray, self.gray_frame,
                                                                     self.old_points, None, **self.lukas_kanade_params)

                # Update old points
                self.old_gray = self.gray_frame.copy()
                self.old_points = new_points

                # Calculate new corner points for rectangle
                half_window_w = self.windowSize[0] / 2
                half_window_h =  self.windowSize[1] / 2

                old_points_unraveled = self.old_points[0]

                upper_corner = (int(self.old_points[0][0] - half_window_h), int(self.old_points[0][1] - half_window_w))
                lower_corner = (int(self.old_points[0][0] + half_window_h), int(self.old_points[0][1] + half_window_w))

                cv2.rectangle(frame, upper_corner, lower_corner, (0, 255, 0), 2)



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
    else:
        video_file_name = None

    helper = LucasKanadeObjectTracker(video_file_name)
    helper.init_detection()
