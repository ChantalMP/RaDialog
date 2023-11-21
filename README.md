## RaDialog: A Large Vision-Language Model for Radiology Report Generation and Conversational Assistance
**Authors**: [Chantal Pellegrini*][cp], [Ege Özsoy*][eo], [Benjamin Busam][bb], [Nassir Navab][nn], [Matthias Keicher][mk]

[cp]:https://www.cs.cit.tum.de/camp/members/chantal-pellegrini/
[eo]:https://www.cs.cit.tum.de/camp/members/ege-oezsoy/
[mk]:https://www.cs.cit.tum.de/camp/members/matthias-keicher/
[nn]:https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/
[bb]:https://www.cs.cit.tum.de/camp/members/benjamin-busam-1/

## [Paper](https://arxiv.org/abs/2106.02009) | [Demo](https://www.youtube.com/watch?v=8Z3QX6Q4Zq4) | Dataset - Coming Soon

<img align="right" src="figs/example.png" alt="teaser" width="50%" style="margin-left: 20px">

Conversational AI tools that can generate and discuss clinically correct radiology reports for a given medical image have the potential to transform radiology. Such a human-in-the-loop radiology assistant could facilitate a collaborative diagnostic process, thus saving time and improving the quality of reports. Towards this goal, we introduce RaDialog, the first thoroughly evaluated and publicly available large vision-language model for radiology report generation and interactive dialog. RaDialog effectively integrates visual image features and structured pathology findings with a large language model (LLM) while simultaneously adapting it to a specialized domain using parameter-efficient fine-tuning. To keep the conversational abilities of the underlying LLM, we propose a comprehensive, semi-automatically labeled, image-grounded instruct dataset for chest X-ray radiology tasks. By training with this dataset, our method achieves state-of-the-art clinical correctness in report generation and shows impressive abilities in interactive tasks such as correcting reports and answering questions, serving as a foundational step toward clinical dialog systems.

## Installation

### Environment Setup:
#### 1) RaDialog Environment
- Install the RaDialog environment with `conda create --name radialog python=3.7`
- Activate the environment with `conda activate radialog`
- Install the requirements with `pip install -r requirements.txt`
- Reinstall correct versions of torch and transformers with `pip install torch==1.13.1 transformers==4.28.1`

#### 2) CheXbert Environment
- Install the CheXbert environment with `conda create --name chexbert python=3.7`
- Activate the environment with `conda activate chexbert`
- Move to the chexbert directory with `cd chexbert`
- Install the requirements with `pip install -r requirements.txt`
- Set the absolute path to the chexbert env and folder in `local_config.py`

### Prepare the Data and Models:

#### 1) Download MIMIC-CXR
- Download the MIMIC-CXR dataset from [here](https://physionet.org/content/mimic-cxr/2.0.0/)
- in local_config.py set the path to the MIMIC-CXR dataset
- in model/lavis/defaults_report.yaml set the path to the MIMIC-CXR dataset

#### 2) Create sectioned report data
- go to the mimic-cxr folder with `cd mimic-cxr`
- run `python create_section_files.py` to prepare the report data

#### 3) Prepare the instruct dataset

- As MIMIC-CXR needs a certified PhysioNet account to be accessed, we can not publish our instruct dataset directly.
- We are working on publishing the instruct dataset on PhysioNet. In the meantime, you can create an instruct dataset yourself by following the steps below.

- The MIMIC-NLE data has to be generated first, as it also contains protected data. Follow the instructions [here](https://github.com/maximek3/MIMIC-NLE/tree/main) to generate the MIMIC-NLE data and set the path to the MIMIC-NLE data in `local_config.py`.
- For the correction task, you can write us, then we can share the used incorrect predictions with you.
- To generate data without Correction or Reasoning (MIMIC-NLE), please comment our line 335 or 336 in "create_data.py" accordingly.

Data for RaDialog-RG:
- run `python create_data.py --mode "RG"` to generate the report generation dataset in the required format (no instruct data)

Data for RaDialog-INS:
- run `python create_data.py --mode "INS"` to generate the instruct dataset

4) Download pretrained models
- Download the pretrained models from [here](TODO) and place them in the checkpoints folder

### Run Demo:
- run `python demo.py --cfg-path configs/blip2_pretrain_stage1_emb.yaml` to start the demo
- connect to the demo with a browser at `http://127.0.0.1:7860` (check terminal for address) and start chatting with RaDialog

### Evaluate RaDialog on MIMIC-CXR test set:
- RaDialog-RG: run `python test.py --prompt img_matching_examples_ig2_noexamples_IMG_findings --use_embs --num_workers 0 --lora_model checkpoints/vicuna-7b-img-report/checkpoint-11200`
- RaDialog-INS: run `python test.py --prompt img_matching_examples_ig2_noexamples_IMG_findings --use_embs --num_workers 0 --lora_model checkpoints/vicuna-7b-img-instruct/checkpoint-4800`

### Train RaDialog:
#### 1) CheXbert classifier Training
- run `python -m findings_classifier.train --train --run_name "train_chexbert" `
- then run `python -m findings_classifier.train --run_name "save_preds" ` to save the predictions of the trained model

#### 2) Image Encoder Pretraining
- run `python -m pretraining.train`

#### 3) LLM Training
Train RaDialog-RG:
- run `python finetune.py --use_embs True --base_model 'vicuna_v7' --output_dir './lora-cxr-vicuna-specific-7b-noexamples-imgemb-findings-rightpadding-stratified_32imgtokens_600tokens' --wandb_run_name lora-cxr-vicuna-specific-7b-noexamples-imgemb-findings-rightpadding-stratified_32imgtokens_600tokens --prompt_template_name vicuna_v11 --data_path "data/data_files/mimic_cxr_reports_stratified.json" --cutoff_len 600`

Train RaDialog-INS:
- run `python finetune.py --use_embs True --base_model 'vicuna_v7' --output_dir './lora-cxr-vicuna-specific-7b-noexamples-imgemb-findings-rightpadding-stratified_32imgtokens_600tokens_reversed2' --wandb_run_name lora-cxr-vicuna-specific-7b-noexamples-imgemb-findings-rightpadding-stratified_32imgtokens_600tokens_reversed2 --prompt_template_name vicuna_v11 --data_path "data/data_files/instruct_data_stratified.json" --cutoff_len 600`

# TODO fix all epochs etc etc