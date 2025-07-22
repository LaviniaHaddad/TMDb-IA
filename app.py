import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Carregando os dados
movies = pd.read_csv(r'C:\Users\Cliente\Desktop\Python\TMDb\data\tmdb_5000_movies.csv')

# Parse genres
def parse_genres(x):
    try:
        genres = ast.literal_eval(x)
        return [d['name'] for d in genres]
    except:
        return []

movies['genres'] = movies['genres'].apply(parse_genres)
movies['overview'] = movies['overview'].fillna('')
movies['metadata'] = movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + movies['overview']

# TF-IDF e similaridade
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['metadata'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recomendar_filmes(titulo, num_recomendacoes=5, min_votos=50):
    if titulo not in indices:
        return pd.DataFrame()
    idx = indices[titulo]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recomendados = []
    for i, score in sim_scores[1:]:
        if movies.loc[i, 'vote_count'] >= min_votos:
            recomendados.append(i)
        if len(recomendados) == num_recomendacoes:
            break

    return movies.loc[recomendados, ['title', 'release_date', 'vote_average', 'vote_count']]

# Streamlit UI
st.title("Recomendador de Filmes TMDb")

titulos = movies['title'].tolist()

entrada = st.text_input("Digite o nome do filme:")

filme_escolhido = None
if entrada:
    sugestoes = [t for t in titulos if entrada.lower() in t.lower()]
else:
    sugestoes = []

if sugestoes:
    filme_escolhido = st.selectbox("Selecione o filme:", sugestoes)
else:
    st.write("Comece a digitar para ver sugestões")

num_recs = st.slider("Número de recomendações", 1, 10, 5)

if filme_escolhido:
    recomendacoes = recomendar_filmes(filme_escolhido, num_recs)
    if recomendacoes.empty:
        st.write("Filme não encontrado ou sem recomendações suficientes.")
    else:
        st.write(f"Filmes recomendados para: **{filme_escolhido}**")
        for idx, row in recomendacoes.iterrows():
            st.markdown(f"**{row['title']}** ({row['release_date'][:4] if pd.notna(row['release_date']) else 'N/A'}) — Nota: {row['vote_average']} ⭐️ — {row['vote_count']} votos")
