import pandas as pd
import numpy as np
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from tabulate import tabulate

class FlipkartRecommender:
    def __init__(self, data_path):
        """Optimized initialization without heavy dependencies"""
        self.df = self.load_and_preprocess_data(data_path)
        self.vectorizer = TfidfVectorizer(stop_words='english', 
                                        max_features=15000,
                                        ngram_range=(1, 2))  # Added bigrams
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['text_features'])
        
    def load_and_preprocess_data(self, path):
        """More efficient data loading"""
        df = pd.read_csv(path)
        
        # Vectorized cleaning operations
        text_cols = ['product_name', 'description', 'product_category_tree']
        df[text_cols] = df[text_cols].fillna('')
        
        # Faster category tree parsing
        df['product_category_tree'] = df['product_category_tree'].apply(
            lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith('[') else x
        )
        
        # Optimized text feature combination
        df['text_features'] = (df['product_name'] + ' ' + 
                             df['description'] + ' ' + 
                             df['product_category_tree'])
        df['text_features'] = df['text_features'].apply(self.clean_text)
        
        return df
    
    def clean_text(self, text):
        """Optimized text cleaning"""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Removed unnecessary A-Z
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def correct_query_typo(self, query):
        """More robust typo correction"""
        query_words = query.lower().split()
        vocab = self.vectorizer.get_feature_names_out()
        corrected_words = []
        
        for word in query_words:
            # Only correct words above certain length
            if len(word) > 3:  
                matches = get_close_matches(word, vocab, n=1, cutoff=0.7)
                corrected_words.append(matches[0] if matches else word)
            else:
                corrected_words.append(word)
                
        return ' '.join(corrected_words)
    
    def get_recommendations(self, query, top_n=5, show_explanation=False):
        """Optimized recommendation engine"""
        # Typo correction and cleaning
        corrected_query = self.correct_query_typo(query)
        query_clean = self.clean_text(corrected_query)
        
        # Vectorize and calculate similarity
        query_vec = self.vectorizer.transform([query_clean])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Filter out zero-similarity results
        valid_indices = np.where(sim_scores > 0.1)[0]  # Threshold adjustment
        if len(valid_indices) == 0:
            print("\nNo relevant products found.")
            return None
            
        # Get top recommendations
        top_indices = valid_indices[np.argsort(sim_scores[valid_indices])[::-1][:top_n]]
        
        # Prepare results
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = np.round(sim_scores[top_indices], 3)
        
        # Enhanced output formatting
        output = results[[
            'product_name', 'description', 'product_category_tree',
            'retail_price', 'overall_rating', 'similarity_score'
        ]].sort_values('similarity_score', ascending=False)
        
        output.columns = ['Product Name', 'Description', 'Category', 'Price', 'Rating', 'Score']
        
        # Display results
        print(f"\n🔍 Original Query: '{query}'")
        print(f"🔄 Corrected Query: '{corrected_query}'")
        print(f"🏆 Best Match Score: {results['similarity_score'].max():.3f}\n")
        
        display_df = output.copy()
        display_df['Description'] = display_df['Description'].apply(
            lambda x: (x[:100] + '...') if len(str(x)) > 100 else str(x)
        )
        
        print(tabulate(
            display_df.values.tolist(),
            headers=display_df.columns,
            tablefmt="fancy_grid",
            colalign=("left", "left", "left", "center", "center", "center")
        ))
        
        return output

# Example Usage
if __name__ == "__main__":
    # Initialize with faster loading
    print("🚀 Initializing optimized recommender...")
    recommender = FlipkartRecommender("flipkart_com-ecommerce_sample.csv")
    
    # Get recommendations
    query = "tshirts for men"  # Intentional typo
    print(f"\n🔎 Getting recommendations for: '{query}'")
    results = recommender.get_recommendations(query, top_n=5)