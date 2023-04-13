import gptpdf
import os
import openai
import shutil

key_file = "api.key"
question_file = "question.qry"
pdfs_path = "input_pdfs"
output_folder = "output_files"

with open(key_file, "r") as f:
    openai.api_key = f.readline().strip()

if os.path.exists("embedded_files"):
    shutil.rmtree("embedded_files")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

files = [f for f in os.listdir(pdfs_path) if os.path.isfile(
    os.path.join(pdfs_path, f))]

with open(question_file) as file:
    questions = [line.strip() for line in file.readlines()]

for file in files:
    output_file = os.path.join(output_folder, file.replace(".pdf", ".txt"))
    with open(output_file, "w") as f:
        for question in questions:
            answer = gptpdf.question_answer_loop(
                os.path.join(pdfs_path, file), question)
            f.write("\n" + question + "\n")
            f.write(answer + "\n")
