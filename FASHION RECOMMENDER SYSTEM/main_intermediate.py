import pandas as pd
import numpy as np
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from tabulate import tabulate

# Step 1: Load and clean data
def load_data(path):
    df = pd.read_csv(path)
    df['description'] = df['description'].fillna('')
    df['product_category_tree'] = df['product_category_tree'].fillna('')

    # Parse category tree
    df['product_category_tree'] = df['product_category_tree'].apply(lambda x: ast.literal_eval(x)[0] if x.startswith('[') else x)

    # Combine features with more weight to category
    df['text_features'] = (
        df['product_name'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        (df['product_category_tree'] + ' ') * 3  # Triple weight to category
    )
    df['text_features'] = df['text_features'].apply(clean_text)
    return df

# Step 2: Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Step 3: Vectorize text with TF-IDF (including bigrams & trigrams)
def vectorize_text(df, text_column='text_features'):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 3))
    matrix = vectorizer.fit_transform(df[text_column])
    return matrix, vectorizer

# Step 4: Typo correction for query
def correct_query_typo(query, vectorizer):
    query_words = query.split()
    vocab = vectorizer.get_feature_names_out()

    corrected_words = []
    for word in query_words:
        matches = get_close_matches(word, vocab, n=1, cutoff=0.82)
        corrected_words.append(matches[0] if matches else word)

    return ' '.join(corrected_words)

# Step 5: Filter products by category match in query
def filter_by_category(query, df):
    query_lower = query.lower()
    filtered = df[df['product_category_tree'].str.lower().str.contains(query_lower)]
    return filtered if not filtered.empty else df

# Step 6: Boost exact keyword matches
def boost_exact_matches(query, df, sim_scores):
    boosted_scores = sim_scores.copy()
    query_tokens = set(query.lower().split())
    for i, text in enumerate(df['text_features']):
        if query_tokens & set(text.split()):
            boosted_scores[i] += 0.1  # manual boost
    return boosted_scores

# Step 7: Main recommendation engine
def get_recommendations(query, df, top_n=5):
    query_clean = clean_text(query)
    filtered_df = filter_by_category(query_clean, df)

    tfidf_matrix, vectorizer = vectorize_text(filtered_df)
    corrected_query = correct_query_typo(query_clean, vectorizer)
    query_vec = vectorizer.transform([corrected_query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    sim_scores = boost_exact_matches(corrected_query, filtered_df, sim_scores)
    top_indices = sim_scores.argsort()[::-1][:top_n]

    if sim_scores[top_indices[0]] == 0:
        print("\n❌ No relevant products found.")
        return None

    print(f"\n🔍 Top similarity score: {sim_scores[top_indices[0]]:.3f}")
    print(f"\n🔍 Query after typo correction: '{corrected_query}'\n")

    recommended = filtered_df.iloc[top_indices].copy()
    recommended['similarity_score'] = [round(sim_scores[i], 3) for i in top_indices]
    recommended['description'] = recommended['description'].apply(lambda x: x[:80] + '...' if len(str(x)) > 80 else x)
    recommended['overall_rating'] = recommended['overall_rating'].replace('No rating available', 'N/A')
    recommended['discounted_price'] = recommended['discounted_price'].fillna('N/A')

    output = recommended[['product_name', 'description', 'product_category_tree',
                          'discounted_price', 'overall_rating', 'similarity_score']]

    output.columns = ['Product Name', 'Short Description', 'Category', 'Price', 'Rating', 'Score']

    print(tabulate(output.values.tolist(), headers=output.columns, tablefmt="fancy_grid",
                   colalign=("left", "left", "left", "center", "center", "center")))

    return output


# ========== Run Everything ==========
if __name__ == "__main__":
    df = load_data('flipkart_com-ecommerce_sample.csv')
    get_recommendations("tshirt for man", df)
