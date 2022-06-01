import streamlit as st
from make_dataset import get_rec_data,generate_matrix,base_split,get_rating,get_anime
import pandas as pd

#### Functions ####
def info_to_data(anime_data,anime_array = [],rating_array = [],list=False):
    info_data_df = pd.DataFrame({'name':anime_array,'rating':rating_array})
    info_data_df = info_data_df.join(anime_data.set_index('name')['anime_id'], on='name')
    return info_data_df if list==False else info_data_df['rating'].to_list(),info_data_df['anime_id'].to_list()


def get_image(data):
    url_array = []
    anime_array = data['anime_id'].tolist()
    from jikanpy import Jikan
    jikan = Jikan()

    for anime in anime_array:
        search_result = jikan.anime(anime)
        url_array.append(search_result.get('image_url'))

    data['url'] = url_array

    return data

### Datasets ###
anime = get_anime()
rating = get_rating()
rec_data = get_rec_data()
all_data = rec_data.join(anime.set_index('anime_id')['name'], on='anime_id')

### Resources ###
anime_list = all_data['name'].drop_duplicates().sort_values().tolist()[3:]
anime_array = []
rating_array = []

#Streamlit app
st.title("Welcome to my first Recommendation System!")

st.write("In this model, we're going to use collaborative filltering, wich means, "
          "I'gonna need you to give me some information about your taste:")

st.write("Could you please, select and rate (from 1-10), five animes?")

#Columns and select boxes

with st.form(key='form1'):

    col1, col2 = st.columns(2)

    with col1:
        col1_values = [st.selectbox('Anime', (anime_list), key=1),
                    st.selectbox('Anime', (anime_list), key=2),
                    st.selectbox('Anime', (anime_list), key=3),
                    st.selectbox('Anime', (anime_list), key=4),
                    st.selectbox('Anime', (anime_list), key=5)]

    with col2:
        col2_values = [st.selectbox('Rating',(range(1,11)),key=11),
        st.selectbox('Rating',(range(1,11)),key=12),
        st.selectbox('Rating',(range(1,11)),key=13),
        st.selectbox('Rating',(range(1,11)),key=14),
        st.selectbox('Rating',(range(1,11)),key=15)]

    submitted = st.form_submit_button('Predict')

##Gerar e printar recomendações



if submitted:

    for value in col1_values:
        anime_array.append(value)
    for value in col2_values:
        rating_array.append(value)

    new_rating_array, new_anime_array = info_to_data(anime, anime_array, rating_array, True)

    from predict_model import retrain_model

    results = retrain_model(new_anime_array, new_rating_array)
    
    print('Array de animes: {x}'.format(x=new_anime_array))
    print('Array de rating: {x}'.format(x=new_rating_array))
    print(results)

    result_image = get_image(results)

    st.subheader('You may like:')

    for name_anime,url_anime in zip(result_image['name'].to_list(),result_image['url'].to_list()):
        st.markdown(name_anime)
        st.image(url_anime)

            


else:
     st.write('Click to predict!')
