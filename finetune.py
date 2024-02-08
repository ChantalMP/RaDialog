import os
import pickle
import random
import sys
from typing import List, Optional

import fire
import torch
import transformers
from datasets import load_dataset
import wandb

from torch import nn
from torch.utils.data import Sampler
from transformers.modeling_utils import unwrap_model

from local_config import WANDB_ENTITY
from utils.datacollator import MyDataCollatorForSeq2Seq
from model.lavis.models.blip2_models.modeling_llama_imgemb import LlamaForCausalLM

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, PreTrainedModel

from utils.prompter import Prompter

import logging
logger = logging.getLogger(__name__)

#how are input and instruction put together:
'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:

or

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
'''

class BalancedSampler(Sampler):
    def __init__(self, true_indices, false_indices):
        self.true_indices = true_indices
        self.false_indices = false_indices
        self.num_samples = 2 * min(len(self.true_indices), len(self.false_indices))

    def __iter__(self):
        # Randomly sample from true_indices
        sampled_true_indices = random.sample(self.true_indices, len(self.false_indices))
        # Merge and shuffle the two lists of indices
        indices = sampled_true_indices + self.false_indices
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

class InstructTrainer(transformers.Trainer):
    def __init__(self, *args, rep_idxs=None, inst_idxs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rep_idxs = rep_idxs
        self.inst_idxs = inst_idxs

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return BalancedSampler(self.rep_idxs, self.inst_idxs)

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_NAME_FINAL = "adapter_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"
class ImgTrainer(transformers.Trainer): #also save img projector
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if state_dict is None:
                state_dict = self.model.state_dict()
                base_state_dict = self.model.base_model.state_dict()
                if 'model.model.img_proj_layer.weight' in base_state_dict:
                    state_dict['base_model.model.model.img_proj_layer.weight'] = base_state_dict['model.model.img_proj_layer.weight']
                    state_dict['base_model.model.model.img_proj_layer.bias'] = base_state_dict['model.model.img_proj_layer.bias']

            if isinstance(unwrap_model(self.model), PreTrainedModel):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def save_pretrained(model, save_directory, **kwargs):
    r"""
    This function saves the adapter model and the adapter configuration files to a directory, so that it can be
    reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
    method.

    Args:
        save_directory (`str`):
            Directory where the adapter model and configuration files will be saved (will be created if it does not
            exist).
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments passed along to the `push_to_hub` method.
    """
    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
    os.makedirs(save_directory, exist_ok=True)

    # save only the trainable weights
    output_state_dict = get_peft_model_state_dict(model, kwargs.get("state_dict", None))
    base_state_dict = model.base_model.state_dict()
    if 'model.model.img_proj_layer.weight' in base_state_dict:
        output_state_dict['base_model.model.model.img_proj_layer.weight'] = base_state_dict['model.model.img_proj_layer.weight']
        output_state_dict['base_model.model.model.img_proj_layer.bias'] = base_state_dict['model.model.img_proj_layer.bias']

    torch.save(output_state_dict, os.path.join(save_directory, WEIGHTS_NAME_FINAL))

    inference_mode = model.peft_config.inference_mode
    model.peft_config.inference_mode = True
    model.peft_config.save_pretrained(save_directory)
    model.peft_config.inference_mode = inference_mode


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    lora_weights: str = None,
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-cxr",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 2,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024, #256 -> need much more with examples in prompt (1024), 512 for without examples but long IG labels
    val_set_size: int = 5,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [ #default is for llama models
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "lora_training",
    wandb_run_name: str = "lora_mimic_cxr",
    wandb_entity: str = WANDB_ENTITY,
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    use_embs=False,
    use_instruct_data=False
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"lora_weights: {lora_weights}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_entity: {wandb_entity}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model


    if base_model == 'vicuna_v13':
        model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.3", torch_dtype=torch.float16, device_map='auto', load_in_8bit=False)
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3", use_fast=False, truncation_side="right", padding_side="right")
    else: #7b
        model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3", torch_dtype=torch.float16, device_map='auto', load_in_8bit=False)
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3", use_fast=False, truncation_side="right", padding_side="right")

    tokenizer.pad_token = tokenizer.unk_token

    if use_embs:
        model.base_model.img_proj_layer = nn.Linear(768, model.base_model.config.hidden_size).to(model.base_model.device)

    # add special token to tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": ["<IMG>"]})
    model.resize_token_embeddings(len(tokenizer))


    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config) #this sets requires_grad for all params to False
    # unfreeze the img_proj_layer
    model.model.base_model.img_proj_layer.weight.requires_grad = True
    model.model.base_model.img_proj_layer.bias.requires_grad = True

    print("Loading data from ", data_path)
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if use_instruct_data:
        report_indices = [i for i, item in enumerate(train_data) if item['is_report']][:5]
        instruct_indices = [i for i, item in enumerate(train_data) if not item['is_report']][:5]

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name
    )

    if use_instruct_data:
        trainer = InstructTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            rep_idxs = report_indices,
            inst_idxs = instruct_indices,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=10,
                optim="adamw_torch",
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=200 if val_set_size > 0 else None,
                save_steps=200,
                output_dir=output_dir,
                save_total_limit=None,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
                max_steps=-1,
                dataloader_num_workers=8,
                remove_unused_columns=False if use_embs else True,
            ),
            data_collator=MyDataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ) if use_embs else
            transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
    else:
        trainer = ImgTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=10,
                optim="adamw_torch",
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=400 if val_set_size > 0 else None,
                save_steps=400,
                output_dir=output_dir,
                save_total_limit=None,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
                max_steps=-1,
                dataloader_num_workers=8,
                remove_unused_columns=False if use_embs else True
            ),
            data_collator=MyDataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ) if use_embs else
            transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    save_pretrained(model, output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
