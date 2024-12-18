"""
Instruct BLIP for Scene Graph Generation, codes are modified from CCoT
"""

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, InstructBlipConfig,  AutoModelForVision2Seq
import torch
from PIL import Image
import requests
from accelerate import init_empty_weights, infer_auto_device_map
import json
import os
from tqdm import tqdm


# Determine if CUDA (GPU) is available.
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the model configuration.
config = InstructBlipConfig.from_pretrained("Salesforce/instructblip-vicuna-13b")

# Initialize the model with the given configuration.
with init_empty_weights():

    model = AutoModelForVision2Seq.from_config(config)
    model.tie_weights()

# Infer device map based on the available resources.
device_map = infer_auto_device_map(model, max_memory={7: "20GiB", 8: "20GiB", 9: "20GiB"},
                                   no_split_module_classes=['InstructBlipEncoderLayer', 'InstructBlipQFormerLayer',
                                                            'LlamaDecoderLayer'])

device_map['language_model.lm_head'] = device_map['language_projection'] = device_map[('language_model.model'
                                                                                       '.embed_tokens')]

offload = ""
# Load the processor and model for image processing.
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b", device_map="auto")
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b",
                                                             device_map=device_map,
                                                             offload_folder=offload, offload_state_dict=True)


sgPrompt='''
For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Object relationships that are relevant to answering the question.

Scene Graph:
'''


qs_path = ""  #Path to question
ans_path = ""  #Path to store result
img_dir = ""  #Path to image
ans_file = open(ans_path, 'w')


with open(qs_path, 'r') as json_file:
    json_list = list(json_file)


count = 0
for json_str in tqdm(json_list):
    result = json.loads(json_str)
    try:
        cur_image = img_dir + result["image"]
        image = Image.open(cur_image).convert("RGB")
        prompt = "<Image> " +  result["text"].split("?")[0] + "?" + sgPrompt

        
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=0.5,
            temperature=0,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        

        answerPrompt="Use the image and scene graph as context and answer the following question: "
        prompt_score = "<Image> Scene Graph: " + generated_text + '\n\n' + answerPrompt + result["text"] + ". The correct letter is"
        inputs2 = processor(images=image, text=prompt_score, return_tensors="pt").to("cuda")
        outputs2 = model.generate(
            **inputs2,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=0.5,
            temperature=0,
        )
        generated_text = processor.batch_decode(outputs2, skip_special_tokens=True)[0].strip()
    except:
        generated_text = "None"

    temp_result = {"question_id":result["question_id"], "text":generated_text}
    ans_file.write(json.dumps(temp_result) + "\n")
ans_file.close()