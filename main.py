#!/usr/bin/env python

import click
import dlib

from frecog.constants import predictor_model
from frecog.video_handler import FaceRecognizer


@click.group(chain=True)
def cli():
    pass


@click.command("video")
@click.argument("classifier_model", type=click.Path(exists=True))
@click.argument("file_name", default="0", type=click.STRING)
def frecog_video_handler(classifier_model, file_name):
    from frecog.image_handler import load_classifier_model
    labels, predictor = load_classifier_model(classifier_model)
    vh = FaceRecognizer(labels, predictor, file_name)
    vh.show()


@click.command("image")
@click.argument("file_name", type=click.STRING)
def frecog_image_handler(file_name):
    import openface
    from skimage import io

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)

    # Load the image into an array
    # bgrImg = cv2.imread(file_name)
    # image = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    image = io.imread(file_name)

    # Run the HOG face detector on the image data.
    # The result will be the bounding boxes of the faces in our image.
    detected_faces = face_detector(image, 1)

    print("Found {} faces in the file {}".format(len(detected_faces), file_name))

    win = dlib.image_window()

    # Open a window on the desktop showing the image
    win.set_image(image)

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(
            i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)

        if True:
            # Draw a box around each face we found
            win.add_overlay(face_rect)

            # Draw the face landmarks on the screen.
            win.add_overlay(pose_landmarks)

        # Use openface to calculate and perform the face alignment
        alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        # Save the aligned image to a file
        # cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace)

    # Wait until the user hits <enter> to close the window
    dlib.hit_enter_to_continue()


@click.command("infer")
@click.argument("classifier_model", type=click.Path(exists=True))
@click.argument("file_name", type=click.Path(exists=True))
def frecog_infer(classifier_model, file_name):
    from frecog.image_handler import infer, read_image, align_face, load_classifier_model
    image = read_image(file_name)
    aligned_face = align_face(image)
    labels, predictor = load_classifier_model(classifier_model)

    prediction = infer(labels, predictor, aligned_face)
    print("Predict {name} with {confidence:.2f} confidence.".format(**prediction))


@click.command("train")
@click.argument("directory", type=click.Path(exists=True))
def frecog_train(directory):
    from frecog.image_handler import train
    train(directory)


cli.add_command(frecog_video_handler)
cli.add_command(frecog_image_handler)
cli.add_command(frecog_infer)
cli.add_command(frecog_train)

if __name__ == "__main__":
    cli()
