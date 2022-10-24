# import threading # not good
from multiprocessing import Process
import os.path
import time
import torch
# import torch.nn as nn
import cv2
import numpy as np
import tqdm



def Load_weights():
    full_path = "D:\Projects\Yolov7_AutoLabel\yolov7.pt"
    path_to_file = "D:\Projects\Yolov7_AutoLabel"
    return full_path, path_to_file


class Yolo:
    def __init__(self, source, show):
        self.source = source  # list
        self.show = show  # boolean
        p = Process(target=self.process)
        p.start()

    def process(self):
        # colors
        red = (0, 0, 200)
        green = (0, 200, 0)
        blue = (200, 0, 0)
        yellow = (127, 255, 212)
        white = (255, 255, 255)

        bg_color = white
        text_color = red

        # detected people
        tScreenPplCount = ''
        lScreenPplCount = ''

        # counter of corrupted frames in a row
        corrFramesInARow = 0

        weights, path_to_weights = Load_weights()
        # good detection: ('ultralytics/yolov5', 'yolov5m6') yolov7: ('WongKinYiu/yolov7', 'yolov7')

        # model = torch.hub.load('ultralytics/yolov5', 'yolov5n',
        #                        force_reload=True)  # or yolov5m, yolov5l, yolov5x, custom !-yolov5s6 or yolov5m6 were the best-!

        # model = torch.hub.load(repo_or_dir='C:/Projects/mq3-counter', model='yolov5m6.pt', force_reload=True)  # for custom weights
        # model = torch.jit.load('model.pth')
        # todo try using a smaller model :
        weights = torch.load(weights, map_location=torch.device('cpu'))
        model = torch.hub.load(path_to_weights, 'custom', weights, source='local', force_reload=True)

        model.classes = [0]  # detects only people

        for filename in tqdm.tqdm(os.listdir(self.source)):
            f = os.path.join(self.source, filename)
            # checking if it is a file
            if os.path.isfile(f):
                # print(f)
                # Images
                cap = cv2.imread(f)  # cap = cv2.VideoCapture("in.avi")  # or file, Path, PIL, OpenCV, numpy, list, rtsp

                # codec = cv2.VideoWriter_fourcc(*'mp4v')  ##(*'XVID')

                def CheckForNewPeople(ret, frame):

                        # str(results) #shows the size and classes
                        results = model(frame)
                        # prepClass = str(results) #use results to check if there are people
                        # if 'persons' not in prepClass:
                        # print('error')
                        det = results.xyxy[0]
                        #print(det)
                        return det

                if self.show:
                    cv2.namedWindow(f, cv2.WINDOW_NORMAL)  # makes window resizable


                start_time = time.time()  # start time of the loop for the fps calculator


                corrFramesInARow = 0

                    # Apply the mask
                mask = np.zeros((cap.shape[0], cap.shape[1]), dtype=np.uint8)
                    # pts = np.array([[400, 10], [1400, 10],  # left up, right up                 for regular videos from shop
                    #                 [2000, 1100], [1390, 1100]], np.int32)  # right down, left down

                pts = np.array([[100,30],[1200,30],#left up, right up                   for other types of videos
                                    [1000,1100],[10,1100]],np.int32)#right down, left down

                pts = pts.reshape((-1, 1, 2))

                cv2.fillPoly(mask, [pts], 255)
                cv2.polylines(cap, [pts], True, yellow, 4)

                masked_frame = cv2.bitwise_and(cap, cap, mask=mask)

                    # data from tracker
                sortBoxes = CheckForNewPeople(cap, masked_frame)


                    # prevents the counter from going crazy, but at the same time it's not a good way to count objects
                lpplcount = 0
                tpplcount = 0

                    # draw each detected/tracked box
                # print(f'file name {f}')
                # print(f'image size:{cap.shape[0]} {cap.shape[1]}')
                label_name = f.replace('.jpg','.txt')
                label_name = label_name.replace('images','labels')
                # print(f'label file name {label_name}')

                with open(label_name,'w')as l:
                    for bbox in sortBoxes:

                            point_1 = (int(bbox[0]), int(bbox[1]))
                            point_2 = (int(bbox[2]), int(bbox[3]))
                            # print(point_1, point_2)
                            y = cap.shape[0]
                            x = cap.shape[1]

                            x1,y1,x2,y2 = bbox[0],bbox[1],bbox[2],bbox[3]
                            p1 = format(((x1+x2)/2)/x,".6f")  #(x1+x2/2)
                            p2 = format(((y1+y2)/2)/y,".6f")
                            p3 = format((x2-x1) / x, ".6f")
                            p4 = format((y2-y1) / y, ".6f")
                            # print(f'0 {p1} {p2} {p3} {p4}')
                            # print(cap.shape)
                            # print(f'bbox{bbox}')
                            l.write(f'0 {p1} {p2} {p3} {p4}')
                            l.write('\n')
                    l.close()

        # shows the actual fps
                fps = 1.0 / (time.time() - start_time)
                fps = "{:.2f}".format(fps)
                    # print("FPS: ",fps) # FPS = 1 / time to process loop
                cv2.putText(cap, ('FPS: ' + fps), (10, 1070), cv2.FONT_HERSHEY_PLAIN, 2, green, 3)

                    # cv2.imshow('mask',masked_frame)
                if self.show:
                        cv2.imshow(f, cap)
                        cv2.resizeWindow(f, 720, 480)  # remove this to resize the frame with mouse freely
                        key = cv2.waitKey(1)
                        if key == 27:  # quit the program if esc is pressed
                            cap.release()
                            cv2.destroyAllWindows()
                else:
                        pass


if __name__ == '__main__':
    Yolo("images",False).process()