## RaDialog: A Large Vision-Language Model for Radiology Report Generation and Conversational Assistance
**Authors:** [Chantal Pellegrini*][cp], [Ege Özsoy*][eo], [Benjamin Busam][bb], [Nassir Navab][nn], [Matthias Keicher][mk]

[cp]:https://www.cs.cit.tum.de/camp/members/chantal-pellegrini/
[eo]:https://www.cs.cit.tum.de/camp/members/ege-oezsoy/
[mk]:https://www.cs.cit.tum.de/camp/members/matthias-keicher/
[nn]:https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/
[bb]:https://www.cs.cit.tum.de/camp/members/benjamin-busam-1/

[![](https://img.shields.io/badge/Project_Page-green)](https://chantalmp.github.io/RaDialog/) [![](https://img.shields.io/badge/Arxiv-2307.05766-blue)](https://arxiv.org/abs/2311.18681) [![](https://img.shields.io/badge/PhysioNet-Dataset-lightgrey)](https://physionet.org/content/radialog-instruct-dataset/1.0.0/)

**✨ News ✨**
- 26 March 2024: RaDialog Instruct Dataset now available on [PhysioNet](https://physionet.org/content/radialog-instruct-dataset/1.0.0/)! 
---

<img align="right" src="figs/example.png" alt="teaser" width="50%" style="margin-left: 20px">

Conversational AI tools that can generate and discuss clinically correct radiology reports for a given medical image have the potential to transform radiology. Such a human-in-the-loop radiology assistant could facilitate a collaborative diagnostic process, thus saving time and improving the quality of reports. Towards this goal, we introduce RaDialog, the first thoroughly evaluated and publicly available large vision-language model for radiology report generation and interactive dialog. RaDialog effectively integrates visual image features and structured pathology findings with a large language model (LLM) while simultaneously adapting it to a specialized domain using parameter-efficient fine-tuning. To keep the conversational abilities of the underlying LLM, we propose a comprehensive, semi-automatically labeled, image-grounded instruct dataset for chest X-ray radiology tasks. By training with this dataset, our method achieves state-of-the-art clinical correctness in report generation and shows impressive abilities in interactive tasks such as correcting reports and answering questions, serving as a foundational step toward clinical dialog systems.

## Installation

### Environment Setup:

#### 1) RaDialog Environment
- clone this repository and move to the radialog directory with `cd RaDialog`
- Install the RaDialog environment with `conda create --name radialog python=3.7`
- Activate the environment with `conda activate radialog`
- Install the requirements with `pip install -r requirements.txt`
- Install hl-ml-multimodal with `pip install hi-ml-multimodal==0.2.0`
- Reinstall correct versions of torch and transformers with `pip install torch==1.13.0 transformers==4.28.1`
- Install java and set JAVA_HOME and PATH in local_config.py (we used jre1.8.0)

#### 2) CheXbert Environment
- Install the CheXbert environment with `conda create --name chexbert python=3.7`
- Activate the environment with `conda activate chexbert`
- Move to the chexbert directory with `cd chexbert`
- Install the requirements with `pip install -r requirements.txt`
- Set the absolute path to the chexbert env and folder in `RaDialog/local_config.py`

### Prepare the Data and Models:

#### 1) Download pretrained models
- Download the pretrained models from [here](https://github.com/ChantalMP/RaDialog/releases/tag/weights)
- place chexbert.pth in RaDialog/chexbert/src/checkpoint/
- unzip vicuna-7b-img-instruct.zip and vicuna-7b-img-report.zip and place folders into RaDialog/checkpoints/
- unzip chexpert_train and place folder into RaDialog/findings_classifier/checkpoints/
- unzip embs and place folder into RaDialog/pretraining/
- unzip checkpoint_4.pth and place it into outputs/stage1_pt_instruct_blip_origlr_img448/


#### 2) Download MIMIC-CXR
- Download the MIMIC-CXR-JPG dataset from [here](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/)
- The dataset should be saved in .../physionet.org/files/mimic-cxr-jpg
- Go to physionet.org/files/mimic-cxr-jpg/files/ and unzip mimic-cxr-2.0.0-split.csv.gz
- from [here](https://physionet.org/content/mimic-cxr/2.0.0/), dowload mimic-cxr-reports.zip
- unzip it and place the folder in the same directory as the MIMIC-CXR-JPG dataset (e.g. physionet.org/files/)
- in local_config.py set the path to the MIMIC-CXR dataset (e.g. .../physionet.org/files/)
- in model/lavis/defaults_report.yaml set the path to the MIMIC-CXR-JPG dataset (e.g. .../physionet.org/files/mimic-cxr-jpg/2.0.0 )

#### 3) Create sectioned report data
- go to the mimic-cxr folder in the code with `cd mimic-cxr`
- run `python create_section_files.py` to prepare the report data
- go back to the RaDialog directory with `cd ..`

#### 4) Prepare the instruct dataset

- As MIMIC-CXR needs a certified PhysioNet account to be accessed, we can not publish our instruct dataset directly.
- We are working on publishing the instruct dataset on PhysioNet. In the meantime, you can create an instruct dataset yourself by following the steps below or just use our pre-trained model.
- The MIMIC-NLE data has to be generated first, as it also contains protected data. Follow the instructions [here](https://github.com/maximek3/MIMIC-NLE/tree/main) to generate the MIMIC-NLE data and set the path to the MIMIC-NLE data in `local_config.py`.
- For the correction task, you can write us, then we can share the used incorrect predictions with you.
- To generate data without Correction or Reasoning (MIMIC-NLE), please comment our line 335 or 336 in "create_data.py" accordingly.

Data for RaDialog-RG:
- run `python -m data.create_data --mode "RG"` to generate the report generation dataset in the required format (no instruct data)

Data for RaDialog-INS:
- run `python -m data.create_data --mode "INS"` to generate the instruct dataset


### Run Demo:
- run `python demo.py --cfg-path pretraining/configs/blip2_pretrain_stage1_emb.yaml` to start the demo
- connect to the demo with a browser at `http://127.0.0.1:7860` and start chatting with RaDialog

### Evaluate RaDialog on MIMIC-CXR test set:
- RaDialog-RG: run `python test.py --prompt img_matching_examples_ig2_noexamples_IMG_findings --use_embs --num_workers 0 --lora_model checkpoints/vicuna-7b-img-report/checkpoint-11200`
- RaDialog-INS: run `python test.py --prompt img_matching_examples_ig2_noexamples_IMG_findings --use_embs --num_workers 0 --lora_model checkpoints/vicuna-7b-img-instruct/checkpoint-4800`
- RaDialog-INS (correction): run `python test.py --prompt img_matching_examples_ig2_noexamples_IMG_findings --use_embs --num_workers 0 --lora_model checkpoints/vicuna-7b-img-instruct/checkpoint-4800 --do_corr` 
- RaDialog-INS (findings QA): run `python test.py --prompt img_matching_examples_ig2_noexamples_IMG_findings --use_embs --num_workers 0 --lora_model checkpoints/vicuna-7b-img-instruct/checkpoint-4800 --do_cp_all_qa` (or --do_cp_bin_qa)

### Train RaDialog:
#### 1) CheXbert classifier Training
- run `python -m findings_classifier.chexpert_train --train --run_name "train_chexbert"`
- in chexpert_train.py set ckpt_path (line 152) to the path of the trained model you just trained
- then run `python -m findings_classifier.chexpert_train --run_name "save_preds"` to save the predictions of the trained model

#### 2) Alignment Module Pretraining
- run `python -m pretraining.train --cfg-path pretraining/configs/blip2_pretrain_stage1.yaml`, we used the 4th epoch checkpoint
- run `python -m pretraining.train --cfg-path pretraining/configs/blip2_pretrain_stage1_emb.yaml`, to save the embeddings of the trained model

#### 3) LLM Training
Train RaDialog-RG:
- run `python finetune.py --use_embs True --base_model 'vicuna_v7' --output_dir 'checkpoints/lora-vicuna-7b-report' --wandb_run_name lora-vicuna-7b-report --prompt_template_name vicuna_v11 --data_path "data/data_files/mimic_cxr_reports_stratified.json" --cutoff_len 600 --num_epochs 10`
- we used checkpoint-11200

Train RaDialog-INS:
- run `python finetune.py --use_embs True --base_model 'vicuna_v7' --output_dir 'checkpoints/lora-vicuna-7b-instruct' --wandb_run_name lora-vicuna-7b-instruct --prompt_template_name vicuna_v11 --data_path "data/data_files/mimic_cxr_instruct_stratified.json" --cutoff_len 800 --num_epochs 10`
- we used checkpoint-4800

To use a model from a checkpoint, you'll need to perform the following steps:
- make a copy of "pytorch_model.bin" and rename it to "adapter_model.bin"
- copy adapter_config.json to the checkpoint folder (it will be generated after the last epoch or you can copy it from the checkpoints we provide) 
