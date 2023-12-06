import os
import sys
import pandas as pd

import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import decord
from decord import cpu, gpu
import numpy as np
import torchvision.transforms as T
import ast
import sys
import os
import openai

sys.path.append("models/LAVIS")
from lavis.models import load_model_and_preprocess

from utils import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



val = pd.read_csv('/shared/shang/datasets/nextqa/metadata/val.csv')
val = val[val['type'] == 'TP'].reset_index(drop=True)
print(val.shape[0])

video_path = '/shared/shang/datasets/nextqa/videos/'
questions = []
statements = []
true_answers = []
pred_answers = []
true_descriptions = []
pred_descriptions = []
accuracy = []
infos = []
outputs = []

# for i in range(0, val.shape[0]):
for i in range(0, 100):
    video_id = val.loc[i, 'video']
    video_file = str(video_id) + '.mp4'
    print(f'video {i}: {video_file}')
    
    # load video / preprocess
    video = get_video(video_path + video_file)
    question = val.loc[i, 'question']
    options = val.loc[i, 'a0':'a4'].to_list()
    
    print(question)
    print(options)
    
    statement = question_to_statement(question)
    print(statement)
    
    questions.append(question)
    statements.append(statement)
    
    # eval
    try:
        images, caption, relevant_idx = answer_question_half(video_id, question, options, statement)
        final_output, info, output = get_answer(images, question, options, relevant_idx, len(images)-1, caption, None)
        
        true_answer = val.loc[i, 'answer']
        true_answers.append(true_answer)
        
        # if isinstance(final_output, int):
        #     pred_answer = final_output
        # else:
        #     pred_answer = 0
        pred_answer = final_output
        pred_answers.append(pred_answer)

        if true_answer == pred_answer:
            accuracy.append(1)
        else:
            accuracy.append(0)

        print(f'pred answer: {pred_answer}')
        print(f'true answer: {true_answer}')
        print(f'current accuracy: {np.mean(accuracy)}')
        print()
        
        infos.append(info)
        outputs.append(output)
    
        true_description = options[true_answer] 
        pred_description = options[pred_answer] 
        true_descriptions.append(true_description)
        pred_descriptions.append(pred_description)
        print(f'pred description: {pred_description}')
        print(f'true description: {true_description}')
    
    except Exception as e:
        print(e)
        true_answers.append('error')
        pred_answers.append('error')
        accuracy.append(0)
        true_descriptions.append('error')
        pred_descriptions.append('error')
        infos.append('error')
        outputs.append('error')
        
        
       
    if i % 10 == 0:
        evaluation = pd.DataFrame({
                'questions': questions,
                'statements': statements,
                'true_answers': true_answers,
                'pred_answers': pred_answers,
                'true_descriptions': true_descriptions,
                'pred_descriptions': pred_descriptions,
                'accuracy': accuracy,
                'infos': infos,
                'outputs' : outputs,
        })
        evaluation.to_csv('experiment.csv', index=False)


evaluation = pd.DataFrame({
        'questions': questions,
        'statements': statements,
        'true_answers': true_answers,
        'pred_answers': pred_answers,
        'true_descriptions': true_descriptions,
        'pred_descriptions': pred_descriptions,
        'accuracy': accuracy,
        'infos': infos,
        'outputs' : outputs,
})
evaluation.to_csv('experiment.csv', index=False)