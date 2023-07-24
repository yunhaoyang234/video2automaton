!nvidia-smi


import os
HOME = os.getcwd()
print("HOME:", HOME)

!git clone https://github.com/IDEA-Research/GroundingDINO.git
!git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
!pip install -q -e .

import sys
!{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'

!pip uninstall -y supervision
!pip install -q supervision==0.6.0

import supervision as sv
print(sv.__version__)

!pip install -q roboflow

import os

GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))

!mkdir -p {HOME}/weights
!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

import os

GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))

!mkdir -p {HOME}/weights
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

import os

SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))

"""## Load models"""

import numpy as np
import random
from google.colab.patches import cv2_imshow

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""### Load Grounding DINO Model"""

from groundingdino.util.inference import Model

grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

"""### Load Segment Anything Model (SAM)"""

SAM_ENCODER_VERSION = "vit_h"

from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

"""## Object Detection"""

import warnings
warnings.filterwarnings("ignore")

from typing import List

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

import cv2
import supervision as sv

def detect(image, classes):
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=classes),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    return detections

BOX_TRESHOLD = 0.4
TEXT_TRESHOLD = 0.25
CLASSES=[]

"""## Read Video"""

# import cv2
# from google.colab.patches import cv2_imshow
import numpy as np

def read_video(path, scale=5, second_per_frame=2):
    cap = cv2.VideoCapture(path)

    video_fps = cap.get(cv2.CAP_PROP_FPS),
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    video = []
    frame_counter = 0
    frame_per_sec = int(round(video_fps[0]))*second_per_frame

    while True:
        ret, frame = cap.read()
        if not ret: break # break if no next frame

        if frame_counter % frame_per_sec == 0:
            res_frame = cv2.resize(frame, dsize=(int(width//scale), int(height//scale)), interpolation=cv2.INTER_CUBIC)
            # cv2_imshow(res_frame) # show frame
            video.append(res_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # on press of q break
            break
        frame_counter += 1

    # release and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    return np.array(video)

def annotate(image, detections, classes):
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{classes[class_id] if class_id is not None else None} {confidence:0.2f}"
        for _, _, confidence, class_id, _ in detections
    ]

    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    sv.plot_image(annotated_frame, (16, 16))
