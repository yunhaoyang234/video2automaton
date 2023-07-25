import os
HOME = os.getcwd()
print("HOME:", HOME)

os.system('git clone https://github.com/IDEA-Research/GroundingDINO.git')
os.system('git checkout -q 57535c5a79791cb76e36fdb64975271354f10251')
os.system('pip install -q -e .')

import sys
os.system('{sys.executable} -m pip install \'git+https://github.com/facebookresearch/segment-anything.git\'')
os.system('pip uninstall -y supervision')
os.system('pip install -q supervision==0.6.0')

import supervision as sv
print(sv.__version__)

os.system('pip install -q roboflow')

GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))

os.system('mkdir -p {HOME}/weights')
os.system('wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth')

GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))

os.system('mkdir -p {HOME}/weights')
os.system('wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')

SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))

