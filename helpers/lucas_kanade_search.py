import cv2
import numpy as np

class LucasKanadeSearcher:

    def __init__(self):
        self.point = ()
        self.point_selected = False
        self.cap = cv2.VideoCapture(0)
        self.windowName = "Frame"

        cv2.namedWindow(self.windowName)

    def select_mouse_point(self, event, x, y, flags, params):
        """
        Callback to mouse click event to update the state and show
        the selected area.

        Updates the class state of the selected point and boolean value
        """

        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x, y)
            self.point_selected = False

    def reset(self):
        self.cap.release()
        self.point = ()
        self.point_selected = False
        cv2.destroyAllWindows()

    def process_frames(self):
        cv2.setMouseCallback(self.windowName, select_mouse_point)

        while True:

            _, frame = self.cap.read()

            if self.point_selected is True:
                cv2.circle(frame, self.point, 5, (0, 255, 0), 2)

            cv2.imshow(self.windowName, frame)

            key = cv2.waitKey(1)

            if key == 27:
                break

            reset()