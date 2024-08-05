import streamlit as st
import anthropic
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set page config
st.set_page_config(page_title="Article Comments Analysis", layout="wide")

# Initialize the Anthropic client
client = anthropic.Client(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Load and preprocess the CSV data
@st.cache_data
def load_and_process_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'heathercomments.csv')
    df = pd.read_csv(csv_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['embedding'] = df.apply(lambda row: model.encode(f"{row['Article']}: {row['Comment']}"), axis=1)
    return df

# Function to find most relevant comments
def find_relevant_comments(query, df, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([x], [query_embedding])[0][0])
    return df.nlargest(top_k, 'similarity')

# Load data
df = load_and_process_data()

st.title("Article Comments Analysis")

# Article selection
article_titles = df['Article'].unique()
selected_article = st.selectbox("Select an article:", article_titles)

# User input
user_question = st.text_input("Ask a question about the comments on the selected article:")

# Process user input
if user_question:
    # Filter comments for the selected article
    article_df = df[df['Article'] == selected_article]
    relevant_comments = find_relevant_comments(user_question, article_df)
    context = "\n\n".join([f"Article: {row['Article']}\nComment: {row['Comment']}" for _, row in relevant_comments.iterrows()])
    
    prompt = f"""You are a helpful assistant that analyzes comments on various articles. The current article is "{selected_article}".
    Be concise and use bullet points whenever possible.

    Context:\n{context}\n\nQuestion: {user_question}\n\nPlease answer the question based on the provided comments for the article "{selected_article}"."""
    
    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        st.subheader("AI Response:")
        
        if response.content:
            for content in response.content:
                if hasattr(content, 'text'):
                    st.markdown(content.text)
        else:
            st.write("No content in the response.")
        
    except Exception as e:
        st.error(f"Error processing AI response: {str(e)}")

st.sidebar.markdown("### About")
st.sidebar.write("This AI assistant analyzes comments on various articles, powered by Claude AI.")
