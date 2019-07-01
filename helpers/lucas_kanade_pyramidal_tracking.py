import numpy as np
import cv2


class LucasKanadePyramidalObjectTracker:

    def __init__(self, lk_window_size = (15, 15), maxPyramidalLevel = 2):
        # Setup initial point state
        self.point = ()
        self.point_selected = False
        self.old_points = np.array([[]])

        # Setup constants for OpenCV
        self.windowName = "Classical Lucas-Kanade tracking."

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

    def start_detection(self):
        # Create video capture
        cap = cv2.VideoCapture(0)

        # Set initial old frame
        _, frame = cap.read()
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Set callback for window mouse click
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, 800, 800)
        cv2.setMouseCallback(self.windowName, self.point_select_callback)

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
                                                                     self.old_points, None)

                # Update old points
                self.old_gray = self.gray_frame.copy()
                self.old_points = new_points

                # Draw new object position
                x, y = new_points.ravel()
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)


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
