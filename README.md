# YOLOv7 Auto-Labeling Tool 🏷️

This repository builds upon the original [**YOLOv7**](https://github.com/WongKinYiu/yolov7) framework, with an additional script, [`autolabel.py`](https://github.com/DidoeS14/Yolov7_AutoLabel/blob/main/autolabel.py), developed to automatically label new images using a pre-trained model. The purpose of this tool is to expand and enhance your dataset by generating labels based on the model's detections, allowing you to iteratively improve the model's performance.

## 📄 Overview

The [`autolabel.py`](https://github.com/DidoeS14/Yolov7_AutoLabel/blob/main/autolabel.py) script automates the labeling of images using a trained YOLOv7 model. By feeding new images into the model, bounding box annotations are generated automatically in [YOLO format](https://docs.ultralytics.com/datasets/detect/), saving significant time and effort in manual labeling. This can be useful if you have already trained a reliable model and want to boost its dataset with new samples.

## 🚀 Key Features

- **Automated Label Generation**: Utilizes a trained YOLOv7 model to create bounding box annotations for new images.
- **Format Compatibility**: Saves bounding boxes in YOLO's format (`class x_center y_center width height`).
- **Process Management**: Runs image processing in parallel to maximize efficiency.
- **Dynamic Class File Creation**: Automatically generates a `classes.txt` file listing class names.

## 🛠️ Dependencies

Ensure that the following libraries are installed:
- `torch`
- `cv2` (OpenCV)
- `numpy`
- `tqdm`
- `detectron2`
- `yaml`
- `matplotlib`

If you have any trouble with installing the framework, simply refer to its [original documentation](https://github.com/WongKinYiu/yolov7?tab=readme-ov-file#official-yolov7).

## ⚙️ Usage Instructions
**1. Clone the Repository:**

`git clone https://github.com/yourusername/Yolov7_AutoLabel.git`

**2. Set Up Paths:** Update full_path and path_to_file in Load_weights() with the location of your trained YOLOv7 weights.

**3. Run the Script:** Place the images to be labeled in a source folder and specify it when creating the Yolo instance.

```python
if __name__ == '__main__':
    yinput = 'images' # folder you want to get images from
    youtput = 'labels' # folder you want to generate the labels

    Yolo(yinput, youtput)
 ```
**4. Output:**

.txt files for each image will be created in the output folder, containing bounding box annotations.
classes.txt will be created in the output folder with class names.
## 📚 References

This project builds on the original [YOLOv7 framework](https://github.com/WongKinYiu/yolov7). Refer to the YOLOv7 [documentation](https://docs.ultralytics.com/models/yolov7/#how-do-i-install-and-run-yolov7-for-a-custom-object-detection-project) for more details on model training and deployment.

This tool aims to simplify the data-labeling process for iterative training in computer vision projects. Feel free to adapt and modify for your own workflows!
