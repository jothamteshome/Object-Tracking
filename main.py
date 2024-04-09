import cv2
import numpy as np

def showImage(img, title='image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def checkChanges(diff, line_width, detection_area):
    x, y, w, h = detection_area

    # Crop image to only show changes in detection area
    crop_img = diff[y+line_width:y+h-line_width, x+line_width:x+w-line_width]

    contours, heirarchy = cv2.findContours(crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, heirarchy


def highlightChanges(curr, contours, heirarchy, line_width, detection_area=(0, 0, 0, 0), kernel_size=0):
    # Separate out coordinates of detection area
    x, y, _, _ = detection_area
    ks = kernel_size // 2

    # Plot rectangles from contours of detected changes
    for i, c in enumerate(contours):
        # Skip if contour is inside other contour
        if heirarchy[0, i, 3] != -1:
            continue

        rect = cv2.boundingRect(c)

        r_x, r_y, r_w, r_h = rect

        area = r_w * r_h
        # Compute top left and bottom right point of rectangle
        left_x, left_y = x+line_width+r_x-ks, y+line_width+r_y-ks
        right_x, right_y = x-line_width+r_x+r_w+ks, y-line_width+r_y+r_h+ks

        # Plot rectangles
        if area > 700:
            cv2.rectangle(curr, (left_x, left_y), (right_x, right_y), (0, 255, 0), 1)



def getMedianFrame(file_path):
    cap = cv2.VideoCapture(file_path)

    random_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)

    frames = []
    for index in random_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()

        frames.append(frame)

    cap.release()

    return np.median(frames, axis=0).astype(np.uint8)



def processConsecutiveFrames(prev, curr):
    # Take absolute difference of two frames
    diff = cv2.absdiff(prev, curr)

    # Drop all pixel values of 10 or less to 0
    # (assuming differences of 10 or less due to jpeg artifacting)
    diff = np.where(diff <= 10, 0, diff)

    # Convert difference of images to grayscale
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply filters for denoising?
    kernel_size = 9
    diff = cv2.erode(diff, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)))
    diff = cv2.GaussianBlur(diff, ksize=(kernel_size, kernel_size), sigmaX=3, sigmaY=3)

    return diff, kernel_size


def main():
    file_name = 'fruit-and-vegetable-detection'
    # Open image (replace with video feed)
    cap = cv2.VideoCapture(f'data/{file_name}.mp4')
    line_width = 2

    if (cap.isOpened() == False):
        raise ValueError('Video capture has not been opened')
    
    # Set first frame as background
    background = getMedianFrame(f'data/{file_name}.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)

    changes = cv2.VideoWriter(f'outputs/{file_name}-changes.mp4', cv2.VideoWriter.fourcc('m', 'p', '4', 'v'), fps, (background.shape[1], background.shape[0]))
    diffs = cv2.VideoWriter(f'outputs/{file_name}-diffs.mp4', cv2.VideoWriter.fourcc('m', 'p', '4', 'v'), fps, (background.shape[1], background.shape[0]))
    
    while (cap.isOpened()):
        ret, curr = cap.read()

        # Make sure frames are not none before detecting changes
        try:
            diff, kernel_size = processConsecutiveFrames(background, curr)
        except cv2.error:
            break


        # Define bounding box area or object detection
        detection_area = (200, 100, background.shape[1]-400, background.shape[0]-200)
        x, y, w, h = detection_area

        # Plot rectangle of detection area in red
        cv2.rectangle(curr, (x, y), (x+w, y+h), (0, 0, 255), line_width)

        # Find objects in detection area and put green box around them
        contours, heirarchy = checkChanges(diff, line_width, detection_area)
        highlightChanges(curr, contours, heirarchy, line_width, detection_area, kernel_size)

        # Show detected object within detection area
        # showImage(curr, 'Detected differences')
        changes.write(curr)
        diffs.write(cv2.cvtColor(diff * 10, cv2.COLOR_GRAY2BGR))

    cap.release()
    changes.release()
    diffs.release()


if __name__ == "__main__":
    main()