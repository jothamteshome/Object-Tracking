import cv2
import numpy as np

def showImage(img, title='image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def checkChanges(diff, line_width, detection_area, ):
    x, y, w, h = detection_area

    # Crop image to only show changes in detection area
    crop_img = diff[y+line_width:y+h-line_width, x+line_width:x+w-line_width]

    contours, _ = cv2.findContours(crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def highlightChanges(curr, contours, line_width, detection_area=(0, 0, 0, 0)):
    # Separate out coordinates of detection area
    x, y, _, _ = detection_area

    # Plot rectangles from contours of detected changes
    for c in contours:
        rect = cv2.boundingRect(c)

        r_x, r_y, r_w, r_h = rect

        # Compute top left and bottom right point of rectangle
        left_x, left_y = x+line_width+r_x, y+line_width+r_y
        right_x, right_y = x-line_width+r_x+r_w, y-line_width+r_y+r_h

        # Plot rectangles
        cv2.rectangle(curr, (left_x, left_y), (right_x, right_y), (0, 255, 0), 1)


def processConsecutiveFrames(prev, curr):
    # Take absolute difference of two frames
    diff = cv2.absdiff(prev, curr)

    # Convert difference of images to grayscale
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)


    # Apply filters for denoising?
    diff = cv2.erode(diff, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)))

    # Show changes between two frames
    showImage(diff*10, 'Difference in Frames')

    return diff


def main():
    # Open image (replace with video feed)
    prev = cv2.imread('data/image.jpg')
    curr = cv2.imread('data/image_cpy.jpg')
    line_width = 2

    # Make sure frames are not none before detecting changes
    if curr is None or prev is None:
        raise TypeError('Attempting to compare two frames but only one provided')

    diff = processConsecutiveFrames(prev, curr)

    # Define bounding box area or object detection
    detection_area = (50, 150, 200, 100)
    x, y, w, h = detection_area

    # Plot rectangle of detection area in red
    cv2.rectangle(curr, (x, y), (x+w, y+h), (0, 0, 255), line_width)

    # Find objects in detection area and put green box around them
    contours = checkChanges(diff, line_width, detection_area)
    highlightChanges(curr, contours, line_width, detection_area)

    # Show detected object within detection area
    showImage(curr, 'Detected differences')


if __name__ == "__main__":
    main()