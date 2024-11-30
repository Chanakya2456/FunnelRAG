import os

# Install necessary libraries
os.system('pip install torch')
os.system('pip install faiss-cpu')
os.system('pip install rank_bm25')
os.system('pip install sentence-transformers PyPDF2')
os.system('pip install pandas')
os.system('pip install numpy')
os.system('pip install scikit-learn')
os.system('pip install fitz')
os.system('pip install transformers')
os.system('pip install ujson')
os.system('pip install groq')

from PyPDF2 import PdfReader
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import json
import gc
import csv
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel
import torch
import re
import sys
import ujson as json
import string
from collections import Counter
from groq import Groq
from rank_bm25 import BM25Okapi

os.environ['GROQ_API_KEY'] = "Your Groq API Key"

# Set directory path
directory_path = r"full_contract_pdf"

with open(r"cuad_lbrag_100_ans.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

ground_truth_path = r"cuad_lbrag_100_ans.json"

# Iterate through directory and rename .PDF files to .pdf
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.PDF'):
            old_file_path = os.path.join(root, file)
            new_file_name = file.replace('.PDF', '.pdf')
            new_file_path = os.path.join(root, new_file_name)
            os.rename(old_file_path, new_file_path)



# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(file_path):
    text = ""
    # Open the PDF file in binary read mode
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        # Iterate over all pages and extract text
        for page in reader.pages:
            if page is not None:
                text += page.extract_text() or ""
    return text
import os

# Specify the path to your main folder containing subfolders with PDFs
folder_path = directory_path

# Create a list to store all the PDF file paths
pdf_files = []

# Walk through the main folder and its subfolders
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.pdf'):
            # Add the full path of each PDF file to the list
            pdf_files.append(os.path.join(root, file))


# Extract text from each PDF and store it with filenames
documents = []
document_names = []
document_map = {}
i=1
for file_path in pdf_files:
    i+=1
    text = extract_text_from_pdf(file_path)
    if text.strip():  # Only include non-empty documents
        documents.append(text)
        document_names.append(os.path.basename(file_path))
        document_map[os.path.basename(file_path)] = text.strip()


# Convert documents to embeddings using SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a small, fast model
document_embeddings = model.encode(documents, convert_to_tensor=True)



# Perform clustering using KMeans
if len(pdf_files)>30:
   num_clusters = int(len(pdf_files)/15)  # Adjust this number based on your needs
elif len(pdf_files)>10:
    num_clusters = int(len(pdf_files)/5) 
elif len(pdf_files>5):
    num_clusters = int(len(pdf_files)/3) 
else:
    num_clusters=len(pdf_files)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(document_embeddings.cpu().numpy())

# Organize documents by cluster
clusters = {i: [] for i in range(num_clusters)}
for i, label in enumerate(kmeans.labels_):
    clusters[label].append(document_names[i])



cluster_documents = []
cluster_ids = []

# Iterate over clusters to prepare text and IDs for each cluster
for cluster_id, doc_names in clusters.items():  
    # Combine the content of documents in the cluster into a single string
    cluster_text = " ".join([document_map[doc_name] for doc_name in doc_names if doc_name in document_map])
    cluster_documents.append(cluster_text)
    cluster_ids.append(cluster_id)

# Tokenize the text of each cluster for BM25 model usage
tokenized_clusters = [cluster.split() for cluster in cluster_documents]

# Initialize the BM25 model with tokenized clusters
bm25 = BM25Okapi(tokenized_clusters)

# Function to find the best matching cluster for a given query
def retrieve_best_cluster(query):
    # Regular expression to check if a document name is included in the query
    pattern = r"\b[\w\-. ]+\.pdf\b"  
    doc_name_in_query = None
    match = re.search(pattern, query)

    if match:
        # Extract document name if found in the query
        doc_name_in_query = match.group()

    cluster_found = None

    # Check if the document name found in the query belongs to a cluster
    if doc_name_in_query is not None:
        for cluster_id, doc_names in clusters.items():
            if doc_name_in_query in doc_names:
                cluster_found = cluster_id
                break

    # Return the cluster if found directly using the document name
    if cluster_found is not None:
        return cluster_found, None  

    # If no document name is found, use BM25 scoring to find the best cluster
    tokenized_query = query.rsplit(' ', 1)[0]  # Exclude the last word for better match
    tokenized_query = tokenized_query.split()
    scores = bm25.get_scores(tokenized_query)

    # Get the index of the best-scoring cluster
    best_cluster_index = np.argmax(scores)
    best_cluster_id = cluster_ids[best_cluster_index]

    return best_cluster_id, scores[best_cluster_index]

# Prepare a list of document names from the specified cluster
document_names = list(clusters[cluster_id])

# Load the SentenceTransformer model for document similarity scoring
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to find the best matching document in a given cluster for a query
def retrieve_best_document(query, cluster_id):
    document_names = list(clusters[cluster_id])
    pattern = r"\b[\w\-. ]+\.pdf\b" 
    doc_name_in_query = None
    match = re.search(pattern, query)

    if match:
        # Extract document name if found in the query
        doc_name_in_query = match.group()

    document_found = None

    # Check if the document name matches any document in the cluster
    if doc_name_in_query is not None:
        for doc_names in clusters[cluster_id]:
            if doc_name_in_query in doc_names:
                document_found = doc_name_in_query
                break

    # Return the document if found directly using the name
    if document_found is not None:
        return document_found, None  

    # If no document name is found, use sentence embeddings to find the best document
    tokenized_query = query.rsplit(' ', 1)[0]  # Exclude the last word for better match
    query_embedding = embed_model.encode(tokenized_query, convert_to_tensor=True)
    doc_embeddings = embed_model.encode([document_map[doc_name] for doc_name in document_names], convert_to_tensor=True)

    # Calculate cosine similarity scores between the query and document embeddings
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]

    # Get the index of the best-scoring document
    best_doc_idx = scores.argmax().item()

    best_document_name = document_names[best_doc_idx]
    best_document_score = scores[best_doc_idx].item()

    return best_document_name, best_document_score



# Function to retrieve a document's full content from the document map using its name
def document_maker(document_name):
   document = document_map[document_name]
   return document

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to split a document into chunks of a specified word size
def chunk_document(doc, chunk_size=130):
    # Split the document into sentences using a regular expression that looks for punctuation followed by a space
    sentences = re.split(r'(?<=[.!?]) +', doc)  

    chunks = []  # List to store chunks of the document
    current_chunk = []  # Temporary list to hold the current chunk of sentences

    # Loop through each sentence and add it to the current chunk
    for sentence in sentences:
        current_chunk.append(sentence)
        
        # If the current chunk reaches the desired size (in words), add it to the chunks list
        if len(' '.join(current_chunk).split()) >= chunk_size:  
            chunks.append(' '.join(current_chunk))  # Join sentences to form a chunk
            current_chunk = []  # Reset the current chunk

    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Function to find the best matching chunk from a document using BERT embeddings
def retrieve_best_chunk_with_bert(query, document, chunk_size=150):
    # Split the document into chunks of the specified size
    chunks = chunk_document(document, chunk_size)

    # Prepare the query for BERT (excluding the last word for better context matching)
    query = query.rsplit(' ', 1)[0]
    query_inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    # Get the BERT embedding for the query (using mean pooling over the last hidden state)
    query_embedding = model(**query_inputs).last_hidden_state.mean(dim=1)

    chunk_embeddings = []  # List to store embeddings for each chunk

    # Loop through each chunk, tokenize, and get its BERT embedding
    for chunk in chunks:
        chunk_inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
        chunk_embedding = model(**chunk_inputs).last_hidden_state.mean(dim=1)
        chunk_embeddings.append(chunk_embedding)

    scores = []  # List to store cosine similarity scores between the query and each chunk

    # Calculate cosine similarity between the query embedding and each chunk embedding
    for chunk_embedding in chunk_embeddings:
        cosine_sim = torch.cosine_similarity(query_embedding, chunk_embedding).item()
        scores.append(cosine_sim)

    # Find the index of the chunk with the highest similarity score
    best_chunk_idx = scores.index(max(scores))

    # Get the best matching chunk and its similarity score
    best_chunk = chunks[best_chunk_idx]
    best_chunk_score = scores[best_chunk_idx]

    return best_chunk, best_chunk_score



def create_and_append_to_json(file_path, query, reply):
    # Initialize the data with the first query and reply
    data = [{
        "query": query,
        "reply": reply
    }]

    # Write the initial data to the JSON file (this will create the file)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage:
file_path = 'queries_and_replies.json'


def append_to_existing_json(file_path, query, reply):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Append the new query and reply
    data.append({
        "query": query,
        "reply": reply
    })

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

output_list = []
i=0
for entry in data:
    query = entry["question"]
    cluster_id, score = retrieve_best_cluster(query)
    document_name,score=retrieve_best_document(query,cluster_id)
    document=document_maker(document_name)
    best_chunk, score = retrieve_best_chunk_with_bert(query, document, chunk_size=150)
    i+=1
    if i==1:
        create_and_append_to_json(file_path, query, best_chunk)
    else:
        append_to_existing_json(file_path, query, best_chunk)
    if i==100:
        break






def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


# Load JSON data containing questions and their retrieved chunks
json_file_path = r"queries_and_replies.json"  # Replace with your file path


with open(json_file_path, 'r',encoding='utf-8') as file:
    question_chunks_data = json.load(file)

with open(ground_truth_path, 'r',encoding='utf-8') as file:
    ground_truth_data = json.load(file)


ground_truth_map = {entry['question']: entry['chunk'] for entry in ground_truth_data}

# Initialize variables to hold cumulative F1, precision, recall, and count
total_f1 = 0
total_precision = 0
total_recall = 0
count = 0
fp=0
fn=0
tp=0
# Loop through question_chunks_data, ensuring we align with the ground_truth_map
for question_entry in question_chunks_data:
    question = question_entry['query']
    ground_truth = ground_truth_map.get(question, "")
    chunk1_text = question_entry['reply']
    if chunk1_text:  # Proceed only if "chunk1" exists
        f1, precision, recall = f1_score(chunk1_text, ground_truth)
        total_f1 += f1
        total_precision += precision
        total_recall += recall
        count += 1

# Calculate average F1, precision, and recall
avg_f1 = total_f1 / count if count else 0
avg_precision = total_precision / count if count else 0
avg_recall = total_recall / count if count else 0
print(f"Average F1 Score on context : {avg_f1:.4f}")
print(f"Average Precision on context : {avg_precision:.4f}")
print(f"Average Recall on context : {avg_recall:.4f}")


json_file_path = r"queries_and_replies.json"  # Replace with your file path
with open(json_file_path, 'r', encoding='utf-8') as file:
    question_chunks_data = json.load(file)
llm = Groq()
file_path = 'llm_outputs.json'
i=0
for question_entry in question_chunks_data:
    query=question_entry['query']
    reply=question_entry['reply']
    prompt = query + "\n" + "Use the following information to answer the above query" + reply
    chat_completion = llm.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-70b-versatile",
        stream=False,
    )
    answer_predicted = chat_completion.choices[0].message.content
    i+=1
    if i==1:
        create_and_append_to_json(file_path, query, reply)
    else:
        append_to_existing_json(file_path, query, answer_predicted)

i=0

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


# Load JSON data containing questions and their answers generated using LLM
json_file_path = r"llm_outputs.json"  # Replace with your file path


with open(json_file_path, 'r',encoding='utf-8') as file:
    question_chunks_data = json.load(file)

with open(ground_truth_path, 'r',encoding='utf-8') as file:
    ground_truth_data = json.load(file)


ground_truth_map = {entry['question']: entry['answer'] for entry in ground_truth_data}

# Initialize variables to hold cumulative F1, precision, recall, and count
total_f1 = 0
total_precision = 0
total_recall = 0
count = 0
fp=0
fn=0
tp=0
# Loop through question_chunks_data, ensuring we align with the ground_truth_map
for question_entry in question_chunks_data:
    question = question_entry['query']
    ground_truth = ground_truth_map.get(question, "")
    chunk1_text = question_entry['reply']
    if chunk1_text:  # Proceed only if "chunk1" exists
        f1, precision, recall = f1_score(chunk1_text, ground_truth)
        total_f1 += f1
        total_precision += precision
        total_recall += recall
        count += 1

# Calculate average F1, precision, and recall
avg_f1 = total_f1 / count if count else 0
avg_precision = total_precision / count if count else 0
avg_recall = total_recall / count if count else 0
print(f"Average F1 Score on answer : {avg_f1:.4f}")
print(f"Average Precision on answer : {avg_precision:.4f}")
print(f"Average Recall on answer : {avg_recall:.4f}")

# LLM as a judge
LLM_Score = 0
json_file_path_2=ground_truth_path
with open(json_file_path_2, 'r', encoding='utf-8') as file:
    chunks_data_1 = json.load(file)
json_file_path = r"llm_outputs.json"  
with open(json_file_path, 'r', encoding='utf-8') as file:
    question_chunks_data = json.load(file)
ground_truth_map = {entry['question']: entry['answer'] for entry in chunks_data_1}
for question_entry in question_chunks_data:
    i+=1
    query=question_entry['query']
    reply=question_entry['reply']
    answer=ground_truth_map.get(query, "")
    # Perform the API call
    chat_completion = llm.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""Evaluate the semantic similarity between the following two answers with respect to some question. Output only a single floating-point number between 0 and 1, where 0 indicates no similarity and 1 indicates identical meaning. Respond with only the number and no other text, explanations, or symbols:
Question: {query}

Text A: {answer}

Text B: {reply}""",
            }
        ],
        model="llama-3.1-70b-versatile",
        stream=False,
    )
    
    # Extract the similarity score and accumulate it
    score = float(chat_completion.choices[0].message.content.strip())
    LLM_Score += score

# Calculate and print the average LLM similarity score
average_llm_score = LLM_Score / 100
print(f"Average LLM Similarity Score: {average_llm_score}")
