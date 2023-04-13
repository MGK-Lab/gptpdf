import gradio as gr
import gptpdf
import os
import openai
import shutil

key_file = "api.key"
with open(key_file, "r") as f:
    openai.api_key = f.readline().strip()

if os.path.exists("embedded_files"):
    shutil.rmtree("embedded_files")

title = 'PDF GPT'
description = """ The returned response cites the page number in square brackets([]) where the information is located."""

with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(f'<center><h3>{description}</h3></center>')

    with gr.Row():

        with gr.Group():
            gr.Markdown("<center><h4>OR<h4></center>")
            file = gr.File(
                label='Upload your PDF/ Research Paper / Book here', file_types=['.pdf'])
            question = gr.Textbox(label='Enter your question here')
            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='The answer to your question is :')

        btn.click(gptpdf.question_answer, inputs=[
                  file, question], outputs=[answer])
demo.launch(share=True)
