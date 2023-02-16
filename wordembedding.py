import streamlit as st
import pandas as pd
import numpy as np
import gensim
# Load text data
@st.cache
def load_text_data(file):
    text = file.getvalue().decode("utf-8")
    return text

# Pre-process text data
@st.cache
def preprocess_text_data(text):
    # Clean text data
    text = text.lower().replace('\n', ' ')
    
    # Tokenize text data
    words = text.split()
    
    # Build word2vec model
    model = gensim.models.Word2Vec([words], window=5, min_count=1, workers=4)
    
    return model

def main():
    st.title('Text Summarization Generative Model')
    
    # Upload text file
    file = st.file_uploader('Upload Text File', type=['txt'])
    if file is not None:
        text = load_text_data(file)
        model = preprocess_text_data(text)
        
        # Show word embeddings
        st.write('Word Embeddings:')
        st.write(model.wv.vectors)
        save_model = st.button('Save Model')
        if save_model:
             model.save('word2vec_model')
             st.success('Model saved successfully!')

if __name__ == '__main__':
    main()  
