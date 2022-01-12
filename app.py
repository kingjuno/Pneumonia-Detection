import gradio as gr
import numpy as np
from preprocess import preprocess
from predict import predict

def classify(filepath):
    input_batch = preprocess(filepath)
    result = np.argmax(predict(input_batch))
    return 'Pneumonia' if result == 1 else 'Normal'


title = "Pnuemonia Detection from chest X-Ray using PyTorch"
description = "Detecting Pnuemonia using chest X-Ray "
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
            examples=examples).launch(debug=False, enable_queue=True)