import cv2
import numpy as np

def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def checkChanges(diff, x, y, w, h, line_width):
    crop_img = diff[y+line_width:y+h-line_width, x+line_width:x+w-line_width]

    # for i in range(crop_img.shape[0]):
    #     for j in range(crop_img.shape[1]):
    #         for chan in range(crop_img.shape[2]):
    #             if crop_img[i][j][chan] > 10:
    #                 crop_img[i][j][chan] = 255

    # showImage(crop_img)

    contours, _ = cv2.findContours(crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    highlightChanges(crop_img, contours, 0, 0, 0, 0)
    
    # showImage(crop_img)

    return contours


def highlightChanges(curr, contours, x, y, w, h):
    for c in contours:
        rect = cv2.boundingRect(c)

        r_x, r_y, r_w, r_h = rect
        cv2.rectangle(curr, (x, y), (r_x+r_w, r_y+r_h), (0, 255, 0), 1)


def main():
    # Open image (replace with video feed)
    prev = cv2.imread('data/image.jpg')
    curr = cv2.imread('data/image - Copy.jpg')
    line_width = 2

    if curr is not None and prev is not None:
    # Take absolute difference between frame 1 and frame 0
        diff = cv2.cvtColor(cv2.absdiff(prev, curr), cv2.COLOR_BGR2GRAY)
        diff = cv2.erode(diff, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        # diff = diff * 10
        showImage(diff)

        # Define bounding box area or object detection
        x, y, w, h = (50, 150, 200, 100)
        cv2.rectangle(diff, (x, y), (x+w, y+h), (0, 0, 255), line_width)

        contours = checkChanges(diff, x, y ,w, h, line_width)
        highlightChanges(curr, contours, x, y, w, h)

        # showImage(curr)


if __name__ == "__main__":
    main()