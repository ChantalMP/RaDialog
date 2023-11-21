"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
import wandb

from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm


import model.lavis.tasks as tasks
from model.lavis.common.config import Config
from model.lavis.common.dist_utils import get_rank
from model.lavis.common.logger import setup_logger


from model.lavis.common.registry import registry
from model.lavis.common.utils import now

# imports modules for registration
from model.lavis.common.optims import (
   LinearWarmupCosineLRScheduler,
   LinearWarmupStepLRScheduler,
)
from model.lavis.datasets.builders import *
from model.lavis.models import *
from model.lavis.processors import *
from model.lavis.runners import *
from model.lavis.tasks import *
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
from local_config import PT_MODE, PATH_TO_MIMIC_CXR


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    registry.mapping['paths']['cache_root'] = '.'
    cfg = Config(parse_args())

    job_id = now()

    # init_distributed_mode(cfg)
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    wandb_run = wandb.init(
        project=cfg.run_cfg.project_name,
        entity=cfg.run_cfg.wandb_entity,
        name=cfg.run_cfg.run_name
    )

    cfg.pretty_print()

    task = tasks.setup_task(cfg)

    # my report dataset
    datasets = {}
    datasets['mimic_cxr'] = {}
    datasets['mimic_cxr']['train'] = MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.0.0",
                                                       split="train", cfg=cfg, truncate=None)
    datasets['mimic_cxr']['train_val'] = MIMIC_CXR_Dataset(vis_processor=None, text_processor=None,
                                                           vis_root=f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.0.0", split="train", cfg=cfg,
                                                           truncate=1000)  # 1000
    datasets['mimic_cxr']['val'] = MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.0.0",
                                                     split="validate", cfg=cfg, truncate=None)
    datasets['mimic_cxr']['test'] = MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.0.0",
                                                      split="test", cfg=cfg, truncate=None)

    model = task.build_model(cfg)
    print(summary(model, input_size=None, device='cpu'))


    if PT_MODE == 'train':
        ''' training code '''
        runner = RunnerBase(
            cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
        )

        runner.train(wandb_run)
        # save the last checkpoint
        # create folder
        if not os.path.exists(f"models/{cfg.run_cfg.run_name}"):
            os.makedirs(f"models/{cfg.run_cfg.run_name}")
        torch.save(model.state_dict(), os.path.join("models", cfg.run_cfg.run_name, "model.pth"))


    else:
        ''' precompute Q-Former output embeddings for all images '''
        model.cuda()
        model.eval()
        dataloader = DataLoader(datasets['mimic_cxr']['test'], batch_size=256, shuffle=False, num_workers=cfg.run_cfg.num_workers)
        embeddings = {}
        for i, batch in enumerate(tqdm(dataloader)):
            qformer_embs, _ = model.forward_image(batch['image'].cuda())
            for j, id in enumerate(batch['image_id']):
                dicom = datasets['mimic_cxr']['train'].id_to_dicom[id.item()]
                embeddings[dicom] = qformer_embs[j].cpu().detach().numpy()

        # save embeddings
        with open(f"pretraining/embs/{cfg.run_cfg.run_name}_embeddings_test.pkl", "wb") as f: #TODO ME has to be done for train, val and test
            pickle.dump(embeddings, f)


if __name__ == "__main__":
    main()
