import fitz
import re
import numpy as np
import openai
import os
import semantic_search as ss
import os


recommender = ss.SemanticSearch()
engine = "text-davinci-003"


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                    len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[{idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


def load_recommender(path, folder_name="embedded_files", start_page=1):

    global recommender

    pdf_file = os.path.basename(path)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    embeddings_file = os.path.join(folder_name, f"{pdf_file}_{start_page}.npy")

    if os.path.isfile(embeddings_file):
        embeddings = np.load(embeddings_file)
        recommender.embeddings = embeddings
        recommender.fitted = True
        return "Embeddings loaded from file"

    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    np.save(embeddings_file, recommender.embeddings)
    return 'Corpus Loaded.'


def generate_text(prompt, engine="text-davinci-003"):
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message


def generate_answer(question):
    global engine
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
              "Cite each reference using [ Page Number] notation (every result has this number at the beginning). "\
              "Citation should be done at the end of each sentence. If the search results mention multiple subjects "\
              "with the same name, create separate answers for each. Only include information found in the results and "\
              "don't add any additional information. Make sure the answer is correct and don't output false content. "\
              "If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier "\
              "search results which has nothing to do with the question. Only answer what is asked. The "\
              "answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "

    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(prompt, engine)
    return answer


def question_answer(file, question):
    old_file_name = file.name
    file_name = file.name
    file_name = file_name[:-12] + file_name[-4:]
    os.rename(old_file_name, file_name)
    load_recommender(file_name)

    return generate_answer(question)


def question_answer_loop(file, question):
    load_recommender(file)

    return generate_answer(question)
