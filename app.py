import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    layout="centered"
)

# --------------------------------------------------
# THEME TOGGLE
# --------------------------------------------------
theme = st.toggle("🌙 Dark Mode")

# --------------------------------------------------
# NETFLIX-INSPIRED THEME (CSS ONLY)
# --------------------------------------------------
if theme:
    st.markdown("""
    <style>
    .stApp {
        background-color: #141414;
        color: white;
    }
    h1 {
        color: #E50914;
        text-align: center;
        font-weight: 900;
        letter-spacing: -1px;
    }
    h3 {
        color: white;
        text-align: center;
    }
    p {
        color: #b3b3b3;
        text-align: center;
    }
    .stButton > button {
        background-color: #E50914;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 10px 22px;
        border: none;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #f6121d;
        transform: scale(1.05);
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    }
    .movie-card {
        background-color: #181818;
        color: white;
        padding: 16px;
        margin-bottom: 16px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .movie-card:hover {
        transform: scale(1.12);
        box-shadow: 0 20px 40px rgba(0,0,0,0.8);
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    h1 {
        color: #E50914;
        text-align: center;
        font-weight: 900;
    }
    h3 {
        color: #111;
        text-align: center;
    }
    p {
        color: #444;
        text-align: center;
    }
    .stButton > button {
        background-color: #E50914;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 10px 22px;
        border: none;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #f6121d;
        transform: scale(1.05);
        box-shadow: 0 6px 14px rgba(0,0,0,0.25);
    }
    .movie-card {
        background-color: white;
        color: #111;
        padding: 16px;
        margin-bottom: 16px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .movie-card:hover {
        transform: scale(1.12);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MOVIES DATA
# --------------------------------------------------
@st.cache_resource
def load_movies():
    return joblib.load("models/movies.pkl")

movies = load_movies()

# --------------------------------------------------
# BUILD SIMILARITY MATRIX (AUTO-GENERATED)
# --------------------------------------------------
@st.cache_resource
def build_similarity(movies_df):
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = tfidf.fit_transform(movies_df["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

similarity = build_similarity(movies)

# --------------------------------------------------
# RECOMMENDER LOGIC
# --------------------------------------------------
movie_index = {title: idx for idx, title in enumerate(movies["title"])}

def recommend(movie_name, top_n=5):
    idx = movie_index[movie_name]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    return [movies.iloc[i[0]].title for i in distances[1:top_n+1]]

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🎬 Movie Recommendation System")
st.markdown(
    "<p>Select a movie and get similar recommendations instantly</p>",
    unsafe_allow_html=True
)

selected_movie = st.selectbox(
    "Choose a movie",
    movies["title"].values
)

top_n = st.slider(
    "Number of recommendations",
    3, 10, 5
)

# --------------------------------------------------
# SHOW RECOMMENDATIONS
# --------------------------------------------------
if st.button("Recommend"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend(selected_movie, top_n)

    st.subheader("🎬 Recommended Movies")

    cols = st.columns(3)
    for idx, movie in enumerate(recommendations):
        with cols[idx % 3]:
            st.markdown(
                f"<div class='movie-card'>🎥 {movie}</div>",
                unsafe_allow_html=True
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <hr style="margin-top:40px; margin-bottom:10px;">
    <p style="text-align:center; font-size:14px; color:gray;">
        Built with ❤️ by <b>Raj Lalji Pandey</b>
    </p>
    """,
    unsafe_allow_html=True
)
