import streamlit as st
import pandas as pd
import numpy as np
import ast
import tensorflow as tf
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender",layout="wide")

def parse_genres(val):
    try:
        result=ast.literal_eval(val)
        return result if isinstance(result,list) else []
    except:
        return []

@st.cache_data
def load_data():
    df=pd.read_csv("movie_data.csv")
    x=np.load("text_embeddings.npy")
    hold=np.load("image_embeddings.npy")
    sim=cosine_similarity(x)
    img_sim=cosine_similarity(hold)
    df['genres']=df['genres'].apply(parse_genres)
    m=df['vote_count'].quantile(0.70)
    C=df['vote_average'].mean()
    df['weighted_score']=((df['vote_count']/(df['vote_count']+m))*df['vote_average']+(m/(df['vote_count']+m))*C)
    return df,sim,img_sim

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ncf_model.keras")

@st.cache_data
def get_genres(df,n=5):
    c=Counter()
    for g in df['genres']:
        c.update(g)
    return [g for g,_ in c.most_common(n)]

df,sim,img_sim=load_data()
model=load_model()
all_genres=get_genres(df)

def recommend(title,mode="hybrid"):
    matches=df[df['title']==title]
    if matches.empty:
        return []
    idx=matches.index[0]
    if mode=="text":
        scores=sim[idx]
    elif mode=="image":
        scores=img_sim[idx]
    else:
        scores=(0.6*sim[idx])+(0.4*img_sim[idx])
    sorted_sc=sorted(enumerate(scores),key=lambda x:x[1],reverse=True)
    results=[]
    for i in sorted_sc[1:11]:
        results.append({"title":df['title'].iloc[i[0]],"vote_average":df['vote_average'].iloc[i[0]],"poster_path":df['poster_path'].iloc[i[0]]})
    return results

def recommend_ncf(user_id,top_n=10):
    all_ids=np.arange(len(df))
    user_arr=np.full(len(df),user_id)
    preds=model.predict([user_arr,all_ids],verbose=0)
    pred_ratings=preds.flatten()*5.0
    top_idx=np.argsort(pred_ratings)[::-1][:top_n]
    results=[]
    for idx in top_idx:
        results.append({"title":df['title'].iloc[idx],"vote_average":round(float(pred_ratings[idx]),2),"poster_path":df['poster_path'].iloc[idx]})
    return results

if "current_movie" not in st.session_state:
    st.session_state.current_movie=None
if "history" not in st.session_state:
    st.session_state.history=[]
if "page" not in st.session_state:
    st.session_state.page="Get Recommendations"

def select_movie(title):
    if st.session_state.current_movie and st.session_state.current_movie!=title:
        st.session_state.history.append(st.session_state.current_movie)
    st.session_state.current_movie=title
    st.session_state.page="Get Recommendations"

def go_back():
    if st.session_state.history:
        st.session_state.current_movie=st.session_state.history.pop()

def poster_url(path):
    return "https://image.tmdb.org/t/p/w500"+path if pd.notna(path) else None

def display_row(results,key_prefix):
    if not results:
        return
    cols=st.columns(len(results))
    for idx,(col,movie) in enumerate(zip(cols,results)):
        with col:
            url=poster_url(movie['poster_path'])
            if url:
                st.image(url,use_container_width=True)
            st.caption(f"Rating: {movie['vote_average']:.1f}")
            title=movie['title']
            short=title if len(title)<=22 else title[:20]+"..."
            st.button(short,key=f"{key_prefix}_{idx}_{title}",on_click=select_movie,args=(title,),use_container_width=True)

def display_grid(rows,key_prefix,per_row=6):
    for i in range(0,len(rows),per_row):
        display_row(rows[i:i+per_row],f"{key_prefix}_{i}")

st.sidebar.title("Navigation")
st.sidebar.radio("Choose a section:",["Get Recommendations","Browse by Genre","Trending Now","Personalized"],key="page")

st.title("Movie Recommendation System")

if st.session_state.page=="Get Recommendations":
    st.write("Pick a movie to get recommendations, or click any title to keep exploring.")
    movie_list=sorted(df['title'].dropna().unique().tolist())
    selected=st.selectbox("Choose a movie:",movie_list)
    if st.button("Find Similar Movies",type="primary"):
        select_movie(selected)
    if st.session_state.current_movie:
        c1,c2=st.columns([1,6])
        with c1:
            if st.session_state.history:
                st.button("Back",on_click=go_back)
        with c2:
            st.write(f"**Now showing movies similar to:** {st.session_state.current_movie}")
        st.subheader("Similar by Content")
        display_row(recommend(st.session_state.current_movie,mode="text"),"content")
        st.subheader("Similar by Visual Style")
        display_row(recommend(st.session_state.current_movie,mode="image"),"visual")

elif st.session_state.page=="Browse by Genre":
    st.write("Top rated movies by genre, ranked by quality score.")
    selected_genre=st.selectbox("Choose a genre:",all_genres)
    subset=df[df['genres'].apply(lambda g:selected_genre in g)]
    subset=subset.sort_values('weighted_score',ascending=False).head(18)
    display_grid(subset.to_dict('records'),f"genre_{selected_genre}")

elif st.session_state.page=="Trending Now":
    st.write("Most popular movies right now based on TMDB popularity.")
    trending=df.sort_values('popularity',ascending=False).head(18)
    display_grid(trending.to_dict('records'),"trending")

elif st.session_state.page=="Personalized":
    st.write("Get personalized recommendations from our trained Neural Collaborative Filtering model.")
    user_id=st.slider("Select your user ID:",min_value=0,max_value=4999,value=0,step=1)
    if st.button("Get My Recommendations",type="primary"):
        with st.spinner("Running model predictions..."):
            results=recommend_ncf(user_id,top_n=10)
        st.subheader(f"Top picks for user {user_id}")
        display_row(results,"ncf")