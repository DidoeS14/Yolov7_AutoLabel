from multiprocessing import Process
import os.path
import os
import time
import torch
# import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


# upgrade loading model with autoparser
def Load_weights():
        """
    Loads the path and filename for model weights.

    Returns:
        tuple: A tuple containing:
            - full_path (str): Full path to the model weights file.
            - path_to_file (str): Directory path to the model weights.
    """
    full_path = "" # you must add path here like: C:\Projects\Yolov7_AutoLabel\modeium-7-1-6.p
    path_to_file = "" # and here like: C:\Projects\Yolov7_AutoLabel
    return full_path, path_to_file


class Yolo:
    def init(self, source,output):
          """
    Initializes the Yolo class with source and output directories and sets up a separate process for processing images.

    Args:
        source (str): Directory path to the source images.
        output (str): Directory path to output the processed label files.
    """
        self.source = source  # location
        self.output = output
        self.cn = cn
        self.ci = ci


        p = Process(target=self.process)
        p.start()

    def bounding_boxes(self):
          """
    Detects objects in images using a YOLO model and saves the bounding box annotations in YOLO format.

    The function:
        1. Loads weights and initializes the model.
        2. Loops through images in the specified source directory.
        3. For each image, runs object detection and extracts bounding box coordinates.
        4. Writes bounding box annotations to a .txt file in YOLO format, with class IDs and normalized coordinates.
        5. Creates a classes.txt file containing the class names.

    Raises:
        Exception: If there is an issue processing an image file, it skips the file and continues.
    """

        weights, path_to_weights = Load_weights()
        weights = torch.load(weights, map_location=torch.device('cpu'))
        model = torch.hub.load(path_to_weights, 'custom', weights, source='local', force_reload=True)

        #model.classes = [int(self.ci)]  # 0 detects only people

        if not os.path.exists(self.output): os.makedirs((self.output))

        for filename in tqdm.tqdm(os.listdir(self.source)):
            try:
                f = os.path.join(self.source, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    # print(f)
                    # Images
                    cap = cv2.imread(f)

                    def MakeClassesTxt():
                         """
    Creates a 'classes.txt' file in the output directory with the class names.

    Notes:
        - Writes the contents of `self.cn` to 'classes.txt'.
    """
                        with open(self.output + '/classes.txt', 'w') as c:
                            c.write(self.cn)
                            c.close()


                    # str(results) #shows the size and classes
                    results = model(cap)

                    det = results.xyxy[0]
                    label_name = f.replace('.jpg', '.txt')
                    label_name = label_name.replace(self.source, self.output)
                    # print(f'label file name {label_name}')

                    with open(label_name, 'w') as l:
                        for bbox in det:
                            y = cap.shape[0]
                            x = cap.shape[1]
                            cls = int(bbox[5])

                            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                            p1 = format(((x1 + x2) / 2) / x, ".6f")  # (x1+x2/2)
                            p2 = format(((y1 + y2) / 2) / y, ".6f")
                            p3 = format((x2 - x1) / x, ".6f")
                            p4 = format((y2 - y1) / y, ".6f")
                            # print(f'0 {p1} {p2} {p3} {p4}')
                            # print(cap.shape)
                            # print(f'bbox{bbox}')
                            if p1.startswith('1') or p2.startswith('1') or p3.startswith('1') or p4.startswith('1'):
                                print('Invalid detection!')
                                continue
                            print(f'{cls} {p1} {p2} {p3} {p4}')
                            l.write(f'{cls} {p1} {p2} {p3} {p4}')
                            l.write('\n')
                        l.close()
                MakeClassesTxt() 
            except Exception as e:
                    print(e)
                    continue
    def process(self):
           """
    Starts the process of generating bounding box annotations by calling `bounding_boxes`.

    Runs in a separate process when initialized, enabling parallel image processing.
    """
            self.bounding_boxes()

if __name__ == '__main__':
    import argparse
    yinput = 'images' # folder you want to get images from
    youtput = 'labels' # folder you want to generate the labels


    Yolo(yinput, youtput)
