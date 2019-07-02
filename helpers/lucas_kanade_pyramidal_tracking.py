import numpy as np
import cv2


class LucasKanadePyramidalObjectTracker:

    def __init__(self, lk_window_size = (15, 15), maxPyramidalLevel = 2):
        # Setup initial point state
        self.point = ()
        self.point_selected = False
        self.box_selected = False
        self.old_points = np.array([[]])
        self.boxes = []
        self.windowSize = lk_window_size
        self.pyramidalLevel = maxPyramidalLevel

        # Setup constants for OpenCV
        self.windowName = "Pyramidal Lucas-Kanade tracking."

        # The maxLevel describes maximal pyramid level number. If set to 1 - two pyramids are used and so on
        self.lukas_kanade_params = {"winSize": lk_window_size, "maxLevel": maxPyramidalLevel,
                                    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

    def point_select_callback(self, event, x, y, flags, params):
        """
        Updates the current selected point state from OpenCV Window callback
        """

        # Left button clicked - new tracking point selected
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x, y)
            self.point_selected = True
            self.old_points = np.array([[x, y]], dtype=np.float32)

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
            print("Old points: ", self.old_points)

    def start_detection(self):
        # Create video capture
        cap = cv2.VideoCapture(0)

        # Set initial old frame
        _, frame = cap.read()
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Set callback for window mouse click
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, 640, 480)
        cv2.setMouseCallback(self.windowName, self.rectangle_select_callback)

        # Loop over video frames
        while True:
            # Get the new frame
            _, frame = cap.read()
            self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.point_selected is True:
                # Draw a circle at initial selected position
                cv2.circle(frame, self.point, 10, (0, 0, 255), 2)

                # Get the position of new point using Lukas-Kanade tracking method
                new_points, status, error = cv2.calcOpticalFlowPyrLK(self.old_gray, self.gray_frame,
                                                                     self.old_points, None, **self.lukas_kanade_params)

                # Update old points
                self.old_gray = self.gray_frame.copy()
                self.old_points = new_points

                # Draw new object position
                x, y = new_points.ravel()
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            if self.box_selected is True:
                # Draw a selected rectangle on the image
                cv2.rectangle(frame, self.boxes[0], self.boxes[1], (255, 0, 0), 2)

                # Define the region of interest
                roi = frame[self.boxes[-2][1]:self.boxes[-1][1], self.boxes[-2][0]:self.boxes[-1][0]]

                # Update new window shape
                self.windowSize = roi.shape[0:2]

                # Update parameters of Lucas-Kanade method with new window size
                self.lukas_kanade_params = {"winSize":  self.windowSize, "maxLevel": self.pyramidalLevel,
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

            key = cv2.waitKey(1)

            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    helper = LucasKanadePyramidalObjectTracker()
    helper.start_detection()
