
PATH_TO_MIMIC_CXR = "/home/data/DIVA/mimic" #TODO set your own path to MIMIC-CXR-JPG dataset (should point to a folder containing "mimic-cxr-jpg" folder)
PATH_TO_MIMIC_NLE = "/home/data/DIVA/mimic" #TODO set your own path to MIMIC-NLE dataset (should point to a folder containing "mimic-nle" folder)
VIS_ROOT = f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.0.0"

CHEXBERT_ENV_PATH = '/home/guests/chantal_pellegrini/miniconda3/envs/chexbert/bin/python'
#CHEXBERT_ENV_PATH = '<PATH_TO_ENVS>/miniconda3/envs/chexbert/bin/python' #replace with path to chexbert environment

CHEXBERT_PATH = '/home/guests/chantal_pellegrini/RaDialog/chexbert/src'
#CHEXBERT_PATH = '<PATH_TO_PROJECT>/RaDialog/chexbert/src' #replace with path to chexbert project in RaDialog folder

PT_MODE = 'train' #set to train for QFormer pre-training, to "precompute" for pre-computing QFormer embeddings for all images