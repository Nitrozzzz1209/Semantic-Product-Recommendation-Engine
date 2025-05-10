import pandas as pd
import numpy as np
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from tabulate import tabulate
from difflib import get_close_matches

# Step 1: Load and clean data
def load_data(path):
    df = pd.read_csv(path)
    df['description'] = df['description'].fillna('')
    df['product_category_tree'] = df['product_category_tree'].fillna('')
    
    # Parse category tree
    df['product_category_tree'] = df['product_category_tree'].apply(lambda x: ast.literal_eval(x)[0] if x.startswith('[') else x)
    
    # Combine features
    df['text_features'] = df['product_name'].fillna('') + ' ' + df['description'] + ' ' + df['product_category_tree']
    df['text_features'] = df['text_features'].apply(clean_text)
    
    return df

# Step 2: Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Step 3: Vectorize text with TF-IDF
def vectorize_text(df, text_column='text_features'):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)
    matrix = vectorizer.fit_transform(df[text_column])
    return matrix, vectorizer

# # Step 3.1: Correct spelling errors in the query
# def correct_query_word_by_word(query, corpus_vocab):
#     corrected_words = []
#     for word in query.lower().split():
#         match = get_close_matches(word, corpus_vocab, n=1, cutoff=0.7)
#         corrected_words.append(match[0] if match else word)
#     return ' '.join(corrected_words)


#Step 3.2: Corrected Typos
def correct_query_typo(query, vectorizer):
    query_words = query.split()
    vocab = vectorizer.get_feature_names_out()
    
    corrected_words = []
    for word in query_words:
        matches = get_close_matches(word, vocab, n=1, cutoff=0.8)
        corrected_words.append(matches[0] if matches else word)
        
    return ' '.join(corrected_words)


# Step 4: Fuzzy matching to handle partial inputs
def find_best_match(query, df, vectorizer, tfidf_matrix, corpus_vocab):
    corrected_query = correct_query_typo(query, vectorizer)
    query_clean = clean_text(corrected_query)
    query_vec = vectorizer.transform([query_clean])

    if not query_vec.nnz:
        return None  # no known words in corrected query

    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sim_scores.argsort()[::-1][0]

    print(f"🔍 Top similarity score: {sim_scores[top_idx]:.3f}")
    return df.iloc[top_idx]['product_name'] if sim_scores[top_idx] > 0 else None



# Step 5: Recommendation engine
def get_recommendations(query, df, tfidf_matrix, vectorizer, top_n=5):
    query_clean = clean_text(query)
    corrected_query = correct_query_typo(query_clean, vectorizer)
    
    query_vec = vectorizer.transform([corrected_query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][:top_n]
    
    if sim_scores[top_indices[0]] == 0:
        print("\nNo relevant products found.")
        return None

    print(f"\n🔍 Top similarity score: {sim_scores[top_indices[0]]:.3f}")
    print(f"\n🔍 Query after typo correction: '{corrected_query}'\n")

    recommended = df.iloc[top_indices].copy()
    recommended['similarity_score'] = [round(sim_scores[i], 3) for i in top_indices]
    recommended['description'] = recommended['description'].apply(lambda x: x[:80] + '...' if len(str(x)) > 80 else x)

    # Fill missing ratings and prices with fallback
    recommended['overall_rating'] = recommended['overall_rating'].replace('No rating available', 'N/A')
    recommended['discounted_price'] = recommended['discounted_price'].fillna('N/A')

    output = recommended[['product_name', 'description', 'product_category_tree',
                          'discounted_price', 'overall_rating', 'similarity_score']]

    output.columns = ['Product Name', 'Short Description', 'Category', 'Price', 'Rating', 'Score']

    print(tabulate(output.values.tolist(), headers=output.columns, tablefmt="fancy_grid",
                   colalign=("left", "left", "left", "center", "center", "center")))



# ========== Run Everything ==========
df = load_data('flipkart_com-ecommerce_sample.csv')
tfidf_matrix, tfidf_vectorizer = vectorize_text(df)

# ✅ Build vocabulary for fuzzy correction
all_text = ' '.join(df['text_features'].tolist())
corpus_vocab = list(set(all_text.split()))

# Example usage
results = get_recommendations("tshirt for man", df, tfidf_matrix, tfidf_vectorizer)
print(results)