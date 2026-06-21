import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

# Load data once and cache it so it doesn't reload on every interaction
@st.cache_data
def load_data():
    df = pd.read_csv("movie_data.csv")
    x = np.load("text_embeddings.npy")
    hold = np.load("image_embeddings.npy")
    similarity = cosine_similarity(x)
    image_similarity = cosine_similarity(hold)
    return df, similarity, image_similarity

df, similarity, image_similarity = load_data()

def recommend(movie_title, mode="hybrid"):
    matches = df[df['title'] == movie_title]
    if matches.empty:
        return []
    movie_index = matches.index[0]

    if mode == "text":
        posit = similarity[movie_index]
    elif mode == "image":
        posit = image_similarity[movie_index]
    else:
        posit = (0.6 * similarity[movie_index]) + (0.4 * image_similarity[movie_index])

    sorted_sc = sorted(enumerate(posit), key=lambda x: x[1], reverse=True)
    results = []
    for i in sorted_sc[1:11]:
        results.append({
            "title": df['title'].iloc[i[0]],
            "vote_average": df['vote_average'].iloc[i[0]],
            "poster_path": df['poster_path'].iloc[i[0]]
        })
    return results

def display_row(results):
    cols = st.columns(len(results)) if results else []
    for col, movie in zip(cols, results):
        with col:
            if pd.notna(movie['poster_path']):
                poster_url = "https://image.tmdb.org/t/p/w500" + movie['poster_path']
                st.image(poster_url, use_container_width=True)
            title = movie['title']
            if len(title) > 22:
                title = title[:20] + "..."
            st.caption(f"**{title}**  \n⭐ {movie['vote_average']:.1f}")

# ---------------- UI ----------------

st.title("🎬 Movie Recommendation System")
st.write("Pick a movie and get recommendations based on content (plot, genre, cast, director) and visual style (poster).")

movie_list = sorted(df['title'].dropna().unique().tolist())
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Find Similar Movies", type="primary"):
    st.subheader("Similar by Content")
    content_results = recommend(selected_movie, mode="text")
    display_row(content_results)

    st.subheader("Similar by Visual Style")
    visual_results = recommend(selected_movie, mode="image")
    display_row(visual_results)