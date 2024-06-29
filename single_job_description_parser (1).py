# -*- coding: utf-8 -*-
"""
# Job Description and Resume Analyzer

## Overview

This program analyzes a job description and compares it with your resume to help you identify important keywords and potential gaps in your resume. It uses natural language processing techniques to extract and rank keywords, and then checks if these keywords are present in your resume.

## How It Works

1. **Job Description Analysis**:
   - Extracts keywords using semantic analysis (based on word embeddings)
   - Performs a word count analysis

2. **Resume Analysis**:
   - Reads your resume (PDF format)
   - Compares the extracted job description keywords with your resume content

3. **Matching Process**:
   - Uses fuzzy matching to account for slight variations in spelling and word forms
   - Handles hyphenated words and their non-hyphenated variants
   - Normalizes words to account for plurals and some common tenses

4. **Output**:
   - Displays top 50 keywords (semantic score and word count) with indicators for presence in resume
   - Shows lists of keywords missing from your resume
   - Exports detailed results to an Excel file

## How to Use

1. **Setup**:
   - Ensure your resume is saved as 'resume.pdf' in the same directory as this notebook.
   - Replace the placeholder text in the `job_description` variable with the actual job description you want to analyze.

2. **Run the Program**:
   - Execute all cells in the notebook.

3. **Review Results**:
   - Console output will show:
     - Top 50 keywords (semantic score) with resume match indicators
     - Top 50 words (count) with resume match indicators
     - List of missing keywords (semantic score)
     - List of missing words (count)
   - Check the generated Excel file for detailed results

## File Naming

- **Input**:
  - Job Description: Entered directly into the `job_description` variable in the code
  - Resume: Should be named 'resume.pdf' and placed in the same directory as the notebook

- **Output**:
  - Excel File: 'job_description_resume_analysis.xlsx'
    - Contains sheets for Semantic Analysis, Word Count, Missing Keywords, and Missing Words

## Important Notes

- The program uses the 'sentence-transformers/all-MiniLM-L6-v2' model for semantic analysis.
- Fuzzy matching is used with a threshold of 80% similarity to catch variations in words.
- The Excel file provides a comprehensive view of all analyzed data for further review and action.

## Customization

You can modify the following parameters in the code:
- `MODEL_NAME`: Change the semantic analysis model
- Fuzzy matching threshold in the `fuzzy_match` function
- Number of top keywords displayed (default is 50)
"""

job_description = "placeholder, paste job descriptoin here..."

!pip install sentence_transformers
!pip install PyPDF2 fuzzywuzzy
import nltk
import torch
import pandas as pd
import PyPDF2
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Constants
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
FUZZY_MATCH_THRESHOLD = 80
MAX_KEYWORDS = 50

# Load the SentenceTransformer model
def load_model(model_name=MODEL_NAME):
    return SentenceTransformer(model_name)

model = load_model()

# Clean and normalize the input text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join(text.split())
    return ''

# Split text into chunks of maximum length
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

# Get BERT embedding for the input text
def get_bert_embedding(text):
    return model.encode([text], convert_to_tensor=True)

# Extract bigram phrases from the text
def extract_phrases(text):
    words = nltk.word_tokenize(text)
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words)
    finder.apply_freq_filter(2)
    return finder.nbest(bigram_measures.pmi, 10)

# Extract keywords and phrases from the text using semantic analysis
def extract_keywords_and_phrases(text, top_n=MAX_KEYWORDS):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

    phrases = extract_phrases(text)
    filtered_phrases = [' '.join(phrase) for phrase in phrases if all(word.lower() not in stop_words for word in phrase)]

    candidates = filtered_tokens + filtered_phrases

    text_embedding = get_bert_embedding(text)
    token_scores = []
    for candidate in set(candidates):
        candidate_embedding = get_bert_embedding(candidate)
        similarity = torch.nn.functional.cosine_similarity(text_embedding, candidate_embedding)
        token_scores.append((candidate, similarity.item()))

    return sorted(token_scores, key=lambda x: x[1], reverse=True)[:top_n]

# Perform word count analysis on the text
def perform_word_count_analysis(text, top_n=MAX_KEYWORDS):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    word_counts = Counter(filtered_words)

    bigrams = list(nltk.bigrams(filtered_words))
    bigram_counts = Counter(' '.join(bigram) for bigram in bigrams)

    combined_counts = word_counts + bigram_counts
    return combined_counts.most_common(top_n)

# Analyze the job description using semantic and word count methods
def analyze_job_description(description):
    cleaned_description = clean_text(description)
    semantic_keywords = extract_keywords_and_phrases(cleaned_description)
    word_counts = perform_word_count_analysis(cleaned_description)
    return semantic_keywords, word_counts

# Read text content from a PDF file
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Perform fuzzy matching between two words
def fuzzy_match(word1, word2, threshold=FUZZY_MATCH_THRESHOLD):
    return fuzz.ratio(word1.lower(), word2.lower()) >= threshold

# Normalize a word by removing common suffixes
def normalize_word(word):
    word = word.lower()
    if word.endswith('s'):
        word = word[:-1]
    if word.endswith('ed'):
        word = word[:-2]
    if word.endswith('ing'):
        word = word[:-3]
    return word

# Compare job keywords with resume text
def compare_with_resume(job_keywords, resume_text):
    resume_words = set(word.lower() for word in nltk.word_tokenize(resume_text))
    matched_keywords = []

    for keyword, score in job_keywords:
        normalized_keyword = normalize_word(keyword)
        if '-' in keyword:
            variants = [keyword, keyword.replace('-', ' ')]
        else:
            variants = [keyword]

        for variant in variants:
            if any(fuzzy_match(normalize_word(variant), resume_word) for resume_word in resume_words):
                matched_keywords.append((keyword, score))
                break

    return matched_keywords

# Print a list of keywords with match status
def print_keyword_list(title, keywords, matched_keywords, limit=MAX_KEYWORDS):
    print(f"\n{title}:")
    for keyword, score in keywords[:limit]:
        match_status = "✓" if keyword in [k for k, _ in matched_keywords] else "✗"
        print(f"{match_status} {keyword}: {score:.4f}")

# Print a list of missing keywords
def print_missing_keywords(title, keywords):
    print(f"\n{title}:")
    for keyword, score in keywords:
        print(f"{keyword}: {score:.4f}")

# Export data to an Excel file
def export_to_excel(data_dict, filename):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, data in data_dict.items():
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Data exported to {filename}")

# Analyze job description and compare with resume
def analyze_job_description_and_resume(job_description, resume_path):
    cleaned_description = clean_text(job_description)
    semantic_keywords = extract_keywords_and_phrases(cleaned_description)
    word_counts = perform_word_count_analysis(cleaned_description)

    resume_text = read_pdf(resume_path)
    matched_semantic_keywords = compare_with_resume(semantic_keywords, resume_text)
    matched_word_counts = compare_with_resume(word_counts, resume_text)

    missing_semantic = [item for item in semantic_keywords if item not in matched_semantic_keywords]
    missing_counts = [item for item in word_counts if item not in matched_word_counts]

    return semantic_keywords, word_counts, matched_semantic_keywords, matched_word_counts, missing_semantic, missing_counts

# Main execution
resume_path = 'resume.pdf'

semantic_results, count_results, matched_semantic, matched_counts, missing_semantic, missing_counts = analyze_job_description_and_resume(job_description, resume_path)

# Print results
print_keyword_list("Top 50 Keywords (Semantic Score)", semantic_results, matched_semantic)
print_keyword_list("Top 50 Words (Count)", count_results, matched_counts)

# Print missing keywords
print_missing_keywords("Missing Keywords (Semantic Score)", missing_semantic)
print_missing_keywords("Missing Words (Count)", missing_counts)

# Export results to Excel
export_data = {
    'Semantic Analysis': [{'Keyword': k, 'Score': s, 'In Resume': k in [m[0] for m in matched_semantic]} for k, s in semantic_results],
    'Word Count': [{'Word': w, 'Count': c, 'In Resume': w in [m[0] for m in matched_counts]} for w, c in count_results],
    'Missing Keywords (Semantic)': [{'Keyword': k, 'Score': s} for k, s in missing_semantic],
    'Missing Words (Count)': [{'Word': w, 'Count': c} for w, c in missing_counts]
}
export_to_excel(export_data, 'job_description_resume_analysis.xlsx')
