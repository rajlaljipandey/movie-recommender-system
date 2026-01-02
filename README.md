
# 🎬 Movie Recommendation System (Netflix-Inspired)

A **Netflix-inspired Movie Recommendation System** built using **Machine Learning and Streamlit**.  
This web app recommends similar movies based on content similarity using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 🔥 Features

- 🎥 Content-based movie recommendations  
- 🧠 Machine Learning with TF-IDF + Cosine Similarity  
- 🌙 Dark Mode toggle  
- 🎨 Netflix-inspired UI (black & red theme)  
- ⚡ Fast recommendations using caching  
- ☁️ Streamlit Cloud deployable (no large files)

---

## 🧠 How It Works

1. Movie metadata is combined into a single feature space  
2. TF-IDF converts text into vectors  
3. Cosine similarity finds similar movies  
4. Top-N recommendations are displayed instantly  

---

## 📁 Project Structure

```
movie-recommender-system/
│
├── data/
├── models/
│   └── movies.pkl
├── notebooks/
│   └── eda.ipynb
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- Pandas, NumPy  
- Joblib  

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push repository to GitHub  
2. Go to https://share.streamlit.io  
3. Select repository  
4. Set main file as `app.py`  
5. Deploy  

---

## 🎯 Future Enhancements

- Movie posters using TMDB API  
- Collaborative filtering  
- Advanced search & filters  

---

## 👨‍💻 Author

**Raj Lalji Pandey**  
Built with ❤️ for portfolio and learning.

---

## 📜 License

Open-source for educational and portfolio use.
