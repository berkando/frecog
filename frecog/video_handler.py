from datetime import datetime, timedelta

import cv2
import dlib
import numpy as np
import openface
from image_handler import infer

from frecog.constants import predictor_model


class VideoHandler(object):
    def __init__(self, source=0, height=1080, weight=1024):
        try:
            source = int(source)
        except:
            pass
        self.cap = cv2.VideoCapture(source)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, weight)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.source = source
        self.frame = None  # type: numpy.array
        self.rectangles = []
        self.circles = []
        self.text = []

        if not self.cap.isOpened():
            raise RuntimeError("VideoCapture could not open your source {}!".format(source))

    def add_overlays(self):
        if self.frame is not None:
            if self.rectangles:
                for top_left, bottom_right, color in self.rectangles:
                    cv2.rectangle(self.frame, top_left, bottom_right, color)

            if self.circles:
                for x, y, radius, color in self.circles:
                    cv2.circle(self.frame, (x, y), radius, color)

            if self.text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_size = .4
                color = (0, 0, 255)
                for x, y, text in self.text:
                    cv2.putText(self.frame, text, (x, y), font, font_size, color)

    def calculate_overlays(self):
        pass

    def take_snapshot(self):
        from os.path import dirname, join
        status, frame = self.cap.read()
        if not status:
            return

        shot_dir = join(dirname(__file__), "screenshots")
        filename = "Screenshot_from_{:%Y-%m-%d-%H-%M-%S}.jpg".format(datetime.now())
        abs_filename = join(shot_dir, filename)
        cv2.imwrite(abs_filename, frame)
        print(filename, 'saved')

    def show(self):
        while self.cap.isOpened():
            status, self.frame = self.cap.read()
            self.calculate_overlays()
            self.add_overlays()

            cv2.imshow("Capture source '{}'".format(self.source), self.frame)

            ch = cv2.waitKey(1)
            if ch == 27:  # ESC
                break
            if ch == ord(' '):
                self.take_snapshot()

        self.cap.release()
        cv2.destroyAllWindows()


class FaceRecognizer(VideoHandler):
    def __init__(self, labels, predictor, source=0, height=720, weight=1280):
        super(FaceRecognizer, self).__init__(source, height, weight)

        self.face_detector = dlib.get_frontal_face_detector()
        self.face_pose_predictor = dlib.shape_predictor(predictor_model)
        self.face_aligner = openface.AlignDlib(predictor_model)
        self.next_face_detection = datetime.now() + timedelta(seconds=1)
        self.labels = labels
        self.predictor = predictor

        self.aligned_faces = []  # type: list(np.array)

    def add_overlays(self):
        super(FaceRecognizer, self).add_overlays()
        if self.frame is not None and self.aligned_faces:
            x_offset = 15
            y_offset = 15
            for face in self.aligned_faces:
                x_offset_new = x_offset + face.shape[1]
                y_offset_new = y_offset + face.shape[0]
                self.frame[y_offset:y_offset_new, x_offset:x_offset_new] = face
                x_offset = x_offset_new + 15

    def calculate_overlays(self):
        if self.frame is not None and datetime.now() > self.next_face_detection:
            image = np.copy(self.frame)

            color_rectangular = (255, 255, 0)
            color_circles = (255, 0, 255)
            radius_circles = 3
            detected_faces = self.face_detector(image, 1)
            self.aligned_faces = [self.face_aligner.align(96,
                                                          image,
                                                          face_rect,
                                                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                                  for face_rect in detected_faces
                                  ]
            # Detected face
            self.rectangles = [
                (
                    (face_rect.left(), face_rect.top()),
                    (face_rect.right(), face_rect.bottom()),
                    color_rectangular
                )
                for face_rect in detected_faces
                ]
            # Face landmarks of the detected face
            self.circles = [
                (point.x, point.y, radius_circles, color_circles)
                for face_rect in detected_faces
                for point in self.face_pose_predictor(image, face_rect).parts()
                ]

            predictions = [infer(self.labels, self.predictor, aligned_face)
                           for aligned_face in self.aligned_faces]

            def create_label((face, prediction)):
                if prediction['confidence'] > .5:
                    lbl = "{name} ({confidence:.1%}).".format(**prediction)
                else:
                    lbl = "Unknown person"
                return face.left(), face.bottom() + 20, lbl

            self.text = map(create_label, zip(detected_faces, predictions))
            # self.text = [(face.left(), face.bottom() + 20, lbl)
            #             for face, lbl in zip(detected_faces, labels)]

            self.next_face_detection = datetime.now() + timedelta(seconds=1)
