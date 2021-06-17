import numpy as np

def read_yolo(file, image, format_out="ml"):
    """
    read annotation files and return transformed x and y coordinates

    Input:
        - file = path to yolo file
        - image = image which is coupled to the annotations file for shape of image
        - format_out = type of output required (options = plot and ml)

    Return:
        - returns coordinates in desired format
    """
    if type(image) is not np.ndarray:
        image = np.array(image)
        
    dh, dw, _ = image.shape

    with open(file, 'r') as f:
        annotations = []
        for annotation in f:
            c, x, y, w, h = map(float, annotation.split(' '))

            # get values
            x1 = int((x - w / 2) * dw)
            y1 = int((y - h / 2) * dh)

            if format_out == "plot":
                x2 = int(w * dw)
                y2 = int(h * dh) 
            elif format_out == "ml":
                x2 = int((x + w / 2) * dw)
                y2 = int((y + h / 2) * dh)
            else:
                raise ValueError(f"format_out in read_yolo.py must be set to either plot and ml, while {format_out} was provided")
             
            if x2 <= x1 or y2 <= y1:
                print(f"the file at {file} has a weird annoated box which could cause problems")
            
            annotations.append([c, x1, y1, x2, y2])

        return np.array(annotations)