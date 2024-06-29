# -*- coding: utf-8 -*-
"""
# Job Description Keyword Analyzer

This program analyzes job descriptions to extract and rank keywords and phrases
using natural language processing techniques. It performs the following tasks:

1. Loads job descriptions from a CSV file (column name = full_description)
2. Processes the text using NLTK for tokenization and stopword removal
3. Uses a pre-trained language model for semantic analysis
4. Extracts keywords and phrases based on their relevance to the job description
5. Performs word count analysis
6. Aggregates results and exports them to an Excel file

Key features:
- Semantic analysis using pre-trained language models
- Word count and frequency analysis
- Extraction of both single words and phrases
- Comparison of full job descriptions
- Export of results to Excel for further analysis

Dependencies: transformers, torch, pandas, openpyxl, nltk, sentence-transformers, sklearn
"""

#MODEL_NAME = 'roberta-large'
# or
MODEL_NAME = 'xlnet-base-cased'
# or
#MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


!pip install transformers torch pandas openpyxl nltk sentence-transformers
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import re
import torch.nn.functional as F
from collections import defaultdict
from collections import Counter
import csv
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch


# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def load_model(model_name):
    if model_name == 'sentence-transformers/all-MiniLM-L6-v2':
        model = SentenceTransformer(model_name)
        tokenizer = None
    elif model_name == 'xlnet-base-cased':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

# Load the model
model, tokenizer = load_model(MODEL_NAME)

# Combine common job terms with standard English stopwords
stop_words = set(stopwords.words('english'))

def export_to_excel(data_dict, filename):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, data in data_dict.items():
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Data exported to {filename}")

def perform_word_count_analysis(descriptions, top_n=1000):
    word_counts = Counter()
    bigram_counts = Counter()

    for desc in descriptions:
        if desc:
            words = nltk.word_tokenize(desc.lower())
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
            word_counts.update(filtered_words)

            bigrams = list(nltk.bigrams(filtered_words))
            bigram_counts.update(' '.join(bigram) for bigram in bigrams)

    combined_counts = word_counts + bigram_counts
    return combined_counts.most_common(top_n)

def chunk_text(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def get_bert_embedding(text, model_name=MODEL_NAME):
    if model_name == 'sentence-transformers/all-MiniLM-L6-v2':
        return model.encode([text], convert_to_tensor=True)
    else:
        chunks = chunk_text(text)
        chunk_embeddings = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # For XLNet, use the last hidden state
            last_hidden_state = outputs.last_hidden_state
            # Use mean pooling
            mean_pooling = torch.mean(last_hidden_state, dim=1)
            chunk_embeddings.append(mean_pooling)
        # Average the embeddings of all chunks
        return torch.mean(torch.cat(chunk_embeddings, dim=0), dim=0).unsqueeze(0)

def extract_phrases(text):
    words = nltk.word_tokenize(text)
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words)
    finder.apply_freq_filter(2)  # Remove pairs that appear less than 2 times
    return finder.nbest(bigram_measures.pmi, 10)  # Top 10 collocations

def extract_keywords_and_phrases(text, top_n=10, model_name=MODEL_NAME):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

    # Extract phrases
    phrases = extract_phrases(text)
    filtered_phrases = [' '.join(phrase) for phrase in phrases if all(word.lower() not in stop_words for word in phrase)]

    # Combine single tokens and phrases
    candidates = filtered_tokens + filtered_phrases

    text_embedding = get_bert_embedding(text, model_name)

    token_scores = []
    for candidate in set(candidates):
        candidate_embedding = get_bert_embedding(candidate, model_name)
        if model_name == 'sentence-transformers/all-MiniLM-L6-v2':
            similarity = torch.nn.functional.cosine_similarity(text_embedding, candidate_embedding)
        else:
            similarity = torch.nn.functional.cosine_similarity(text_embedding, candidate_embedding)
        token_scores.append((candidate, similarity.item()))

    sorted_tokens = sorted(token_scores, key=lambda x: x[1], reverse=True)
    return sorted_tokens[:top_n]

def load_job_descriptions(file_path):
    df = pd.read_csv(file_path)
    full_descriptions = df['full_description'].tolist()

    def clean_text(text_list):
        cleaned_list = []
        for text in text_list:
            if isinstance(text, str):
                text = text.lower()
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                text = ' '.join(text.split())
                cleaned_list.append(text)
            else:
                cleaned_list.append('')  # Handle non-string entries
        return cleaned_list

    cleaned_full_descriptions = clean_text(full_descriptions)

    return cleaned_full_descriptions

def process_job_columns(descriptions):
    all_keywords = defaultdict(lambda: {'total_score': 0, 'count': 0})
    for desc in descriptions:
        if desc:
            keywords = extract_keywords_and_phrases(desc, top_n=1000)  # Increased from 50 to 1000
            for keyword, score in keywords:
                all_keywords[keyword]['total_score'] += score
                all_keywords[keyword]['count'] += 1
    return all_keywords

def aggregate_keywords(keywords):
    aggregated = []
    for keyword, data in keywords.items():
        total_score = data['total_score']
        count = data['count']
        avg_score = total_score / count if count > 0 else 0
        aggregated.append((keyword, total_score, avg_score, count))
    return sorted(aggregated, key=lambda x: x[1], reverse=True)  # Sort by total score

def analyze_job_columns(full_descriptions, skills_responsibilities):
    full_desc_keywords = process_job_columns(full_descriptions)
    skills_resp_keywords = process_job_columns(skills_responsibilities)
    return full_desc_keywords, skills_resp_keywords

def compare_keywords(full_desc_sorted, skills_resp_sorted, top_n=50):
    full_desc_set = set(keyword for keyword, _ in full_desc_sorted[:top_n])
    skills_resp_set = set(keyword for keyword, _ in skills_resp_sorted[:top_n])

    common_keywords = full_desc_set.intersection(skills_resp_set)
    unique_full_desc = full_desc_set - skills_resp_set
    unique_skills_resp = skills_resp_set - full_desc_set

    return common_keywords, unique_full_desc, unique_skills_resp

def print_keyword_list(title, keywords, limit=50):
    print(f"\n{title}:")
    for keyword, score in keywords[:limit]:
        print(f"{keyword}: {score:.4f}")

# Load and process job descriptions
full_descriptions = load_job_descriptions('job_listings.csv')
full_desc_keywords = process_job_columns(full_descriptions)
full_desc_sorted = aggregate_keywords(full_desc_keywords)

# Word count analysis
full_desc_word_counts = perform_word_count_analysis(full_descriptions, top_n=1000)

# Calculate average word counts
full_desc_avg_counts = [(word, count / len(full_descriptions)) for word, count in full_desc_word_counts]

# Prepare data for export
export_data = {
    'Description Semantic Sum': [{'Keyword': k, 'Sum Score': s, 'Count': c} for k, s, _, c in full_desc_sorted[:1000]],
    'Description Semantic Average': [{'Keyword': k, 'Avg Score': a, 'Count': c} for k, _, a, c in sorted(full_desc_sorted, key=lambda x: x[2], reverse=True)[:1000]],
    'Description Count Sum': [{'Word/Phrase': w, 'Count': c} for w, c in full_desc_word_counts[:1000]],
    'Description Count Average': [{'Word/Phrase': w, 'Avg Count': c} for w, c in full_desc_avg_counts[:1000]],
}

# Export results to a single Excel file
export_to_excel(export_data, f'full_description_analysis_{MODEL_NAME.replace("/", "_")}.xlsx')

"""## Check Keyword Counts"""

# Import necessary libraries
!pip install PyPDF2
import PyPDF2
import csv
import re
import os
import nltk
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Initialize stemmer
stemmer = PorterStemmer()

# Check if the required files are present
if not os.path.exists('resume.pdf') or not os.path.exists('keywords.csv'):
    raise FileNotFoundError("Please ensure both 'resume.pdf' and 'keywords.csv' are present in the current directory.")

print("Setup complete. Required files found and NLTK initialized.")

def read_pdf(file_name):
    """
    Read the content of a PDF file.
    """
    with open(file_name, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text.lower()  # Convert to lowercase for case-insensitive matching

def read_keywords(file_name):
    """
    Read keywords from a CSV file.
    """
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        keywords = [row[0].lower() for row in reader]  # Assuming one keyword per row
    return keywords

def match_keywords(text, keywords):
    """
    Find which keywords are present in the text, accounting for word variations.
    """
    present = []
    missing = []
    text_tokens = nltk.word_tokenize(text)
    text_stems = [stemmer.stem(word) for word in text_tokens]

    for keyword in keywords:
        keyword_stem = stemmer.stem(keyword)
        if keyword_stem in text_stems:
            present.append(keyword)
        else:
            missing.append(keyword)
    return present, missing

def display_results(present, missing, total_keywords):
    """
    Display the results of keyword matching, including percentage score.
    """
    score_percentage = (len(present) / total_keywords) * 100

    print(f"Keyword Match Score: {score_percentage:.2f}%")
    print(f"Keywords found: {len(present)}/{total_keywords}")

    print("\nKeywords found in the resume:")
    for keyword in present:
        print(f"- {keyword}")

    print("\nKeywords missing from the resume:")
    for keyword in missing:
        print(f"- {keyword}")

def main():
    """
    Main function to orchestrate the keyword checking process.
    """
    resume_text = read_pdf('resume.pdf')
    keywords = read_keywords('keywords.csv')
    present, missing = match_keywords(resume_text, keywords)
    display_results(present, missing, len(keywords))

# Run the main function
if __name__ == "__main__":
    main()
