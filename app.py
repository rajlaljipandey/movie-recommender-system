import streamlit as st
import joblib
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")

# --------------------------------------------------
# TMDB CONFIG (SAFE)
# --------------------------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # ✅ CORRECT
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://via.placeholder.com/300x450?text=No+Poster"

def fetch_poster(movie_title):
    if not TMDB_API_KEY:
        return PLACEHOLDER_POSTER

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": movie_title},
            timeout=5
        )
        data = response.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return TMDB_IMAGE_BASE + poster_path

        return PLACEHOLDER_POSTER
    except Exception:
        return PLACEHOLDER_POSTER

# --------------------------------------------------
# THEME TOGGLE
# --------------------------------------------------
theme = st.toggle("🌙 Dark Mode")

# --------------------------------------------------
# NETFLIX-INSPIRED THEME
# --------------------------------------------------
if theme:
    st.markdown("""
    <style>
    .stApp { background-color: #141414; color: white; }
    h1 { color: #E50914; text-align: center; font-weight: 900; }
    p { color: #b3b3b3; text-align: center; }
    .movie-card {
        background-color: #181818;
        color: white;
        padding: 12px;
        border-radius: 6px;
        text-align: center;
        font-weight: 600;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background-color: #f5f5f5; }
    h1 { color: #E50914; text-align: center; font-weight: 900; }
    p { color: #444; text-align: center; }
    .movie-card {
        background-color: white;
        color: #111;
        padding: 12px;
        border-radius: 6px;
        text-align: center;
        font-weight: 600;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MOVIES
# --------------------------------------------------
@st.cache_resource
def load_movies():
    return joblib.load("models/movies.pkl")

movies = load_movies()

# --------------------------------------------------
# BUILD SIMILARITY
# --------------------------------------------------
@st.cache_resource
def build_similarity(df):
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = tfidf.fit_transform(df["tags"]).toarray()
    return cosine_similarity(vectors)

similarity = build_similarity(movies)

# --------------------------------------------------
# RECOMMENDER
# --------------------------------------------------
movie_index = {title: idx for idx, title in enumerate(movies["title"])}

def recommend(movie_name, top_n=5):
    idx = movie_index[movie_name]
    scores = sorted(
        list(enumerate(similarity[idx])),
        key=lambda x: x[1],
        reverse=True
    )
    return [movies.iloc[i[0]].title for i in scores[1:top_n+1]]

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🎬 Movie Recommendation System")
st.markdown("<p>Select a movie and get similar recommendations instantly</p>", unsafe_allow_html=True)

selected_movie = st.selectbox("Choose a movie", movies["title"].values)
top_n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("Recommend"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend(selected_movie, top_n)

    cols = st.columns(3)
    for i, movie in enumerate(recommendations):
        with cols[i % 3]:
            st.image(fetch_poster(movie), use_container_width=True)
            st.markdown(f"<div class='movie-card'>🎥 {movie}</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<hr>
<p style="text-align:center; font-size:14px; color:gray;">
Built with ❤️ by <b>Raj Lalji Pandey</b>
</p>
""", unsafe_allow_html=True)
