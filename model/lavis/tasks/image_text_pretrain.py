"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from model.lavis.common.registry import registry
from model.lavis.tasks.base_task import BaseTask
from model.lavis.datasets.data_utils import move_to_cuda


@registry.register_task("image_text_pretrain_eval")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        loss = 0.0
        for batch in data_loader:
            if cuda_enabled:
                batch = move_to_cuda(batch)
            loss_dict = model(batch)
            loss += loss_dict["loss"].item()

        return loss / len(data_loader)
