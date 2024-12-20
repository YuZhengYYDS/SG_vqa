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


# Determine if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model configuration
config = InstructBlipConfig.from_pretrained("Salesforce/instructblip-vicuna-13b")

# Initialize the model with the given configuration
with init_empty_weights():
    model = AutoModelForVision2Seq.from_config(config)
    model.tie_weights()

# Infer device map based on the single GPU
device_map = infer_auto_device_map(
    model,
    max_memory={0: "16GiB"},  # Assign memory for the single GPU
    no_split_module_classes=[
        'InstructBlipEncoderLayer',
        'InstructBlipQFormerLayer',
        'LlamaDecoderLayer'
    ]
)

# Load the processor and model for image processing
offload = "offload"  # Directory for offloading model components to CPU if needed
processor = InstructBlipProcessor.from_pretrained(
    "Salesforce/instructblip-vicuna-13b", device_map="auto"
)
model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-vicuna-13b",
    device_map=device_map,
    offload_folder=offload,  # Enables offloading to CPU
    offload_state_dict=True
)
# Save the initialized model locally (first time only)
model.save_pretrained("D:\\AiProjects\\SG_vqa\\saved_model")
processor.save_pretrained("D:\\AiProjects\\SG_vqa\\saved_processor")

# Load the model and processor from local files
#model = InstructBlipForConditionalGeneration.from_pretrained("D:\\AiProjects\\SG_vqa\\saved_model")
#processor = InstructBlipProcessor.from_pretrained("D:\\AiProjects\\SG_vqa\\saved_processor")

sgPrompt='''
For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Object relationships that are relevant to answering the question.

Scene Graph:
'''


import json
from PIL import Image
from tqdm import tqdm
import torch

# Paths to input and output
qs_path = "D:\\AiProjects\\SG_vqa\\scene_graph_questions.json"  # Path to the questions JSON file
ans_path = "D:\\AiProjects\\SG_vqa\\ans.json"  # Path to store the result as a JSON file
img_dir = "D:\\AiProjects\\SG_vqa\\images\\"  # Path to the directory containing images

# Open the answer file for writing
ans_file = open(ans_path, 'w')

# Load questions from the JSON file
with open(qs_path, 'r') as json_file:
    questions = json.load(json_file)  # Load JSON directly as a list

# Process each question
for result in tqdm(questions):
    try:
        # Load the image associated with the current question
        cur_image = img_dir + result["image"]
        image = Image.open(cur_image).convert("RGB")
        
        # Construct the prompt for the scene graph
        prompt = "<Image> " + result["text"].split("?")[0] + "?" + sgPrompt
        
        # Process the image and text through the model
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
    except Exception as e:
        print(f"Error processing question {result['question_id']}: {e}")
        generated_text = "None"  # Fallback in case of an error

    # Save the result for the current question
    temp_result = {"question_id": result["question_id"], "text": generated_text}
    ans_file.write(json.dumps(temp_result) + "\n")

# Close the answer file
ans_file.close()
