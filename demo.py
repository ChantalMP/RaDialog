import argparse
import os
import random
import numpy as np
import torch
from torch.backends import cudnn

from chexpert_train import LitIGClassifier
from local_config import JAVA_HOME, JAVA_PATH

# Activate for deterministic demo, else comment
SEED = 16
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True

# set java path
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_PATH + os.environ["PATH"]
os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.getcwd(), "gradio_tmp")

import dataclasses
import json
import time
from enum import auto, Enum
from typing import List, Any


import gradio as gr
from PIL import Image
from peft import PeftModelForCausalLM
from skimage import io
from torch import nn
from transformers import LlamaTokenizer
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, transforms

from model.lavis import tasks
from model.lavis.common.config import Config
from model.lavis.data.ReportDataset import create_chest_xray_transform_for_inference, ExpandChannels
from model.lavis.models.blip2_models.modeling_llama_imgemb import LlamaForCausalLM


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

    return args

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    # Used for gradio server
    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += self.sep + " " + role + ": " + message
                else:
                    ret += self.sep + " " + role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def clear(self):
        self.messages = []
        self.offset = 0
        self.skip_next = False

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


cfg = Config(parse_args())
vis_transforms = create_chest_xray_transform_for_inference(512, center_crop_size=448)
use_img = False
gen_report = True
pred_chexpert_labels = json.load(open('findings_classifier/predictions/structured_preds_chexpert_log_weighting_test_macro.json', 'r'))

def init_blip(cfg):
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model = model.to(torch.device('cpu'))
    return model

def init_chexpert_predictor():
    ckpt_path = f"findings_classifier/checkpoints/chexpert_train/ChexpertClassifier-epoch=06-val_f1=0.36.ckpt"
    chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                          "Cardiomegaly", "Lung Opacity",
                          "Lung Lesion", "Edema",
                          "Consolidation", "Pneumonia",
                          "Atelectasis", "Pneumothorax",
                          "Pleural Effusion", "Pleural Other",
                          "Fracture", "Support Devices"]
    model = LitIGClassifier.load_from_checkpoint(ckpt_path, num_classes=14, class_names=chexpert_cols, strict=False)
    model.eval()
    model.cuda()
    model.half()
    cp_transforms = Compose([Resize(512), CenterCrop(488), ToTensor(), ExpandChannels()])

    return model, np.asarray(model.class_names), cp_transforms


def remap_to_uint8(array: np.ndarray, percentiles=None) -> np.ndarray:
    """Remap values in input so the output range is :math:`[0, 255]`.

    Percentiles can be used to specify the range of values to remap.
    This is useful to discard outliers in the input data.

    :param array: Input array.
    :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
        Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
    :returns: Array with ``0`` and ``255`` as minimum and maximum values.
    """
    array = array.astype(float)
    if percentiles is not None:
        len_percentiles = len(percentiles)
        if len_percentiles != 2:
            message = (
                'The value for percentiles should be a sequence of length 2,'
                f' but has length {len_percentiles}'
            )
            raise ValueError(message)
        a, b = percentiles
        if a >= b:
            raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
        if a < 0 or b > 100:
            raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
        cutoff: np.ndarray = np.percentile(array, percentiles)
        array = np.clip(array, *cutoff)
    array -= array.min()
    array /= array.max()
    array *= 255
    return array.astype(np.uint8)


def load_image(path) -> Image.Image:
    """Load an image from disk.

    The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

    :param path: Path to image.
    :returns: Image as ``Pillow`` ``Image``.
    """
    # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
    image = io.imread(path)

    image = remap_to_uint8(image)
    return Image.fromarray(image).convert("L")


def init_vicuna():
    use_embs = True

    vicuna_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3", use_fast=False, truncation_side="left", padding_side="left")
    lang_model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3", torch_dtype=torch.float16, device_map='auto')
    vicuna_tokenizer.pad_token = vicuna_tokenizer.unk_token

    if use_embs:
        lang_model.base_model.img_proj_layer = nn.Linear(768, lang_model.base_model.config.hidden_size).to(lang_model.base_model.device)
        vicuna_tokenizer.add_special_tokens({"additional_special_tokens": ["<IMG>"]})

    lang_model = PeftModelForCausalLM.from_pretrained(lang_model,
                                                      f"checkpoints/vicuna-7b-img-instruct/checkpoint-4800",
                                                      torch_dtype=torch.float16, use_ram_optimized_load=False).half()
    # lang_model = PeftModelForCausalLM.from_pretrained(lang_model, f"checkpoints/vicuna-7b-img-report/checkpoint-11200", torch_dtype=torch.float16, use_ram_optimized_load=False).half()
    return lang_model, vicuna_tokenizer

blip_model = init_blip(cfg)
lang_model, vicuna_tokenizer = init_vicuna()
blip_model.eval()
lang_model.eval()

cp_model, cp_class_names, cp_transforms = init_chexpert_predictor()

def get_response(input_text, dicom):
    global use_img, blip_model, lang_model, vicuna_tokenizer

    if input_text[-1].endswith(".png") or input_text[-1].endswith(".jpg"):
        image = load_image(input_text[-1])
        cp_image = cp_transforms(image)
        image = vis_transforms(image)
        dicom = input_text[-1].split('/')[-1].split('.')[0]
        if dicom in pred_chexpert_labels:
            findings = ', '.join(pred_chexpert_labels[dicom]).lower().strip()
        else:
            logits = cp_model(cp_image[None].half().cuda())
            preds_probs = torch.sigmoid(logits)
            preds = preds_probs > 0.5
            pred = preds[0].cpu().numpy()
            findings = cp_class_names[pred].tolist()
            findings = ', '.join(findings).lower().strip()

        if gen_report:
            input_text = (
                f"Image information: <IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG>. Predicted Findings: {findings}. You are to act as a radiologist and write the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. "
                "Write in the style of a radiologist, write one fluent text without enumeration, be concise and don't provide explanations or reasons.")
        use_img = True

        blip_model = blip_model.to(torch.device('cuda'))
        qformer_embs = blip_model.forward_image(image[None].to(torch.device('cuda')))[0].cpu().detach()
        blip_model = blip_model.to(torch.device('cpu'))
        # save image embedding with torch
        torch.save(qformer_embs, 'current_chat_img.pt')
        if not gen_report:
            return None

    else:  # free chat
        input_text = input_text
        findings = None

    '''Generate prompt given input prompt'''
    conv.append_message(conv.roles[0], input_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    '''Call vicuna model to generate response'''
    inputs = vicuna_tokenizer(prompt, return_tensors="pt")  # for multiple inputs, use tokenizer.batch_encode_plus with padding=True
    input_ids = inputs["input_ids"].cuda()
    # lang_model = lang_model.cuda()
    generation_output = lang_model.generate(
        input_ids=input_ids,
        dicom=[dicom] if dicom is not None else None,
        use_img=use_img,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=300
    )
    # lang_model = lang_model.cpu()

    preds = vicuna_tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    new_pred = preds[0].split("ASSISTANT:")[-1]
    # remove last message in conv
    conv.messages.pop()
    conv.append_message(conv.roles[1], new_pred)
    return new_pred, findings


'''Conversation template for prompt'''
conv = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant."
           "The assistant gives professional, detailed, and polite answers to the user's questions.",
    roles=["USER", "ASSISTANT"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

# Global variable to store the DICOM string
dicom = None


# Function to update the global DICOM string
def set_dicom(value):
    global dicom
    dicom = value


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history


# Function to clear the chat history
def clear_history(button_name):
    global chat_history, use_img, conv
    chat_history = []
    conv.clear()
    use_img = False
    return []  # Return empty history to the Chatbot


def bot(history):
    # You can now access the global `dicom` variable here if needed
    response, findings = get_response(history[-1][0], None)
    print(response)

    # show report generation prompt if first message after image
    if len(history) == 1:
        input_text = f"You are to act as a radiologist and write the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. Write in the style of a radiologist, write one fluent text without enumeration, be concise and don't provide explanations or reasons."
        if findings is not None:
            input_text = f"Image information: (img_tokens) Predicted Findings: {findings}. {input_text}"
        history.append([input_text, None])

    history[-1][1] = ""
    if response is not None:
        for character in response:
            history[-1][1] += character
            time.sleep(0.01)
            yield history


if __name__ == '__main__':
    with gr.Blocks() as demo:


        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
        )

        with gr.Row():
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
                container=False,
            )

        with gr.Row():
            btn = gr.UploadButton("📁 Upload image", file_types=["image"], scale=1)
            clear_btn = gr.Button("Clear History", scale=1)

        clear_btn.click(clear_history, [chatbot], [chatbot])

        txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
            bot, chatbot, chatbot
        )
        txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
        file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

    demo.queue()
    demo.launch()
