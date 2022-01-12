import gradio as gr
import numpy as np
from preprocess import preprocess
from predict import predict

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def classify(filepath):
    input_batch = preprocess(filepath)
    result = predict(input_batch)
    result = softmax(result).ravel()
    result_ind = np.argmax(result)
    confidence_score = round(result[result_ind]*100,10)
    return f'Pneumonia {confidence_score}%' if result_ind == 1 else f'Normal {confidence_score}%' 

title = "Pnuemonia Detection from chest X-Ray using PyTorch"
description = "Detecting Pnuemonia using chest X-Ray ||"
article = """<p style='text-align: center'>
        <a href='https://nbviewer.org/github/kingjuno/Pneumonia-Detection/blob/master/notebook/pneumonia-det.ipynb?flush_cache=true' target='_blank'>Notebook Link</a>
        <br><a href='https://github.com/kingjuno/Pneumonia-Detection' target='_blank'>Github Repo</a></p>
    """
examples = [['samples/Normal-1.jpeg'], 
            ['samples/Normal-2.jpeg'],
            ['samples/PN-1.jpeg'],
            ['samples/PN-2.jpeg'],
            ['samples/PN-3.jpeg']
        ]

gr.Interface(classify,
            inputs=gr.inputs.Image(type="filepath"),
            outputs= "text", title=title,
            description=description,
            article=article,
            examples=examples).launch(debug=True, enable_queue=True)