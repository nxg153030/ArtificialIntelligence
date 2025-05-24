"""
Implementation of the paper Silhouette Analysis-Based Gait Recognition
for Human Identification

3 modules to the algorithm
1. Human detection and tracking
    1a. Background modeling
    1b. Motion segmentation
    1c. Human tracking
2. Feature extraction
    2a. Extract binary silhouette from each frame
    2b. Map each silhouette into a 1D normalized distance signal by contour unwrapping
3. Training/Classification using PCA
"""
import cv2 as cv
import argparse


def background_subtraction(bg_model):
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/Users/nishant/Documents/pictures:videos/BryceCanyon.mov')
    parser.add_argument('--algo', default='MOG2')
    args = parser.parse_args()

    bg_subtractor = cv.createBackgroundSubtractorMOG2() if args.algo == 'MOG2' else cv.createBackgroundSubtractorKNN()
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
    if not capture.isOpened():
        print(f'Unable to open: {args.input}')
        exit(0)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fg_mask = bg_subtractor.apply(frame)

        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fg_mask)
        print(fg_mask)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

        # TODO: Implement motion tracking


if __name__ == '__main__':
    main()
