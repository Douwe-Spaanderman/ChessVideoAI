import cv2
import time
import argparse
import functools 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def random_colors(n_colors):
    """
    Randomly selects colors based on hex color coding.

    Input:
        - n_colors = number of colors to return

    Return:
        - list of hex colors
    """
    return ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n_colors)]

def display_video(input_file, labels=False):
    """
    displays mp4 files with or without annotations

    Input:
        - input_file = path to input mp4 file
        - labels = path to label file (yolo coordinate format) if False (no annotations)

    Return:
        - returns video
    """
    cap = cv2.VideoCapture(input_file)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def display_image(input_file, labels=False):
    """
    Main script to run display for images and video's with or without annotations

    Input:
        - input_file = path to input file
        - labels = path to label file (yolo coordinate format) if False (no annotations)

    Return:
        - returns image
    """
    image = cv2.imread(input_file)
    fig, ax = plt.subplots()
    # Matplotlib reads rgb not cv2 bgr
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    dh, dw, _ = image.shape

    # Create annotation
    if labels != False:
        colors = random_colors(12)
        with open(labels, 'r') as f:
            for annotation in f:
                c, x, y, w, h = map(float, annotation.split(' '))

                # get values
                x1 = int((x - w / 2) * dw)
                x2 = int(w * dw)
                y1 = int((y - h / 2) * dh)
                y2 = int(h * dh)  

                rect = patches.Rectangle((x1, y1), x2, y2, linewidth=1, edgecolor=colors[int(c)], facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

    plt.show()

def main_display(input_file, labels=False):
    """
    Main script to run display for images and video's with or without annotations

    Input:
        - input_file = path to input file
        - labels = path to label file (yolo coordinate format) if False (no annotations)

    Return:
        - returns display of image/video with or without annotations
    """
    # Check labels format
    if labels == False:
        pass
    elif labels.upper() == "TRUE":
        # easy replace images with label path (important to have the same folder structure)
        repls = ('images', 'label'), ('.png', '.txt')
        labels = functools.reduce(lambda a, kv: a.replace(*kv), repls, input_file)
    elif input_file.endswith(".txt"):
        pass
    else:
        raise ValueError("Annotation neither False nor file is .txt format, therefore it can't be processed.")

    # Check file format
    if input_file.endswith(".mp4"):
        display_video(input_file, labels)
    elif input_file.endswith(".png"):
        display_image(input_file, labels)
    else:
        raise ValueError("Input file is neither a video in .mp4 or a image in .png, therefore it can't be processed.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from google bucket")
    parser.add_argument("Path", help="path to image or video (.png/.mp4)")
    parser.add_argument("-a", dest="Annotation", nargs='?', default=False, help="path to annotations or True (.txt yolo format, default=False)")

    args = parser.parse_args()
    start = time.time()
    main_display(input_file=args.Path, labels=args.Annotation)
    end = time.time()
    print('completed in {} seconds'.format(end-start))