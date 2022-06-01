import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import barplot
import requests
import pickle

#----------------------------------------------------------------------------------------------------------------------#
                # Importing the Pickle file to get the dataframe of the movies #

pickle_in = open('Movie_data.pkl', 'rb') 
movies = pickle.load(pickle_in)

pickle_in = open('countvectorizer.pkl', 'rb') 
bagofwords = pickle.load(pickle_in)

pickle_in = open('Tfidf.pkl', 'rb') 
tfidf = pickle.load(pickle_in)

pickle_in = open('hashvec.pkl', 'rb') 
hashvec = pickle.load(pickle_in)

#----------------------------------------------------------------------------------------------------------------------#
                  #                     Set of fuctions to use in the app                         #
img=[] # to store images links 
captions=[]  # to store images captions
temp=[]
def movie_display(popular_movies):
 getList_name = {}
 
 for x, xRows in popular_movies.iterrows():

    getResponse = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=c0bda0be71f7815fd6ba2eb5f5c86fd8'.format(xRows['movie_id']) ) # every movie has a unique ID 
    getData = getResponse.json() # we request the data from the API and convert it to json
    
    # a bug fixed because sometimes there are is no poster so it returrns error
    if getData['poster_path']== None:
     continue
    else:   
     getPath = "http://image.tmdb.org/t/p/w500" + getData['poster_path']    # get the path of the poster
     getList_name[xRows['title']] = getPath

     temp.append(getPath)
 
 html_string=f""" <div style="font-size:34px; font-weight:Bold; color:#fff; text-align:center; padding-top:8px; height:12%; width: 100%;  margin-top:10px; background-color:#580b0f;">{movie}</div> """
 st.write(html_string, unsafe_allow_html=True)
 # the poster of the movie
 col1, col2, col3 = st.columns([2,1,2])
 with col1:
    st.write("")

 with col2:
    st.image(temp[0], width=250)

 with col3:
    st.write("")  


 for i in range(1,popular_movies.shape[0],5):

    img.append(list(getList_name.values())[i])
    img.append(list(getList_name.values())[i+1])
    img.append(list(getList_name.values())[i+2]) 
    img.append(list(getList_name.values())[i+3])
    img.append(list(getList_name.values())[i+4])

    captions.append(list(getList_name.keys())[i])
    captions.append(list(getList_name.keys())[i+1])
    captions.append(list(getList_name.keys())[i+2])
    captions.append(list(getList_name.keys())[i+3])
    captions.append(list(getList_name.keys())[i+4])
      

 
def recommend(movie,option):
    movie_index = movies[movies['title'] == movie].index[0]
    #('Bag of Words', 'TF-IDF', 'Hash Vectorizer'),)
    if option == 'Bag of Words':
        distances = bagofwords[movie_index]
    elif option == 'TF-IDF':
        distances = tfidf[movie_index]
    else:
        distances = hashvec[movie_index]        
    movies_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[0:21] # number of movies you want to display(note it have to be odd to avoid errors)
    mov=[]
    id=[]
    scores=[]
    for i in movies_list:
        mov.append(movies.iloc[i[0]].title)
        id.append(movies.iloc[i[0]].movie_id)
        scores.append(i[1])
    dic={'movie_id':id,'title':mov,'Similarity Score':scores}
    return pd.DataFrame(dic) 

# a function that puts images next to eachother

def paginator(items, items_per_page=26):


    # Display a pagination selectbox in the specified location.
    items = list(items)

    # Iterate over the items in the page to let the user display them.
   
    import itertools
    return itertools.islice(enumerate(items), 0, 26)
#----------------------------------------------------------------------------------------------------------------------#
                            #     Adjust the title  and configration of the app                          #
movlst=[]
for i in movies['title']:
    movlst.append(i)

st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎥",
        layout="wide",
    )
    
# st.markdown("<h1 style='text-align: center; color: red;'>Movie Recommendation System</h1>", unsafe_allow_html=True) ## this is a title

col1, col2, col3 = st.columns([1,5,1])
with col1:
    st.write("")

with col2:
    st.image("https://heartoflongmont.org/wp-content/uploads/2019/02/Movie-Recommendation.jpg", width=800)

with col3:
    st.write("")
#----------------------------------------------------------------------------------------------------------------------#
        #                                              main app                                           #

option = st.selectbox(
     'Choose the Vectorizer',
     ('Bag of Words', 'TF-IDF', 'Hash Vectorizer'),)

st.write("You selected: ", option)


movie = st.selectbox(
     '',
     movlst,help='Select a movie to get recommendations', index=1)


popular_movies=recommend(movie,option)

plt.rcParams['figure.figsize'] = (10, 4)

fig, ax = plt.subplots()
barplot(popular_movies['Similarity Score'],popular_movies['title'][1:],palette='flare')
ax.set_ylabel('Title')
ax.set_xlabel('Similarity Score')
st.pyplot(fig)

                     
with st.spinner('الصبر طيب'):
    movie_display(popular_movies)
    image_iterator = paginator(img)
    indices_on_page, images_on_page = map(list, zip(*image_iterator))
    st.success('Done, hope you like the movies! 😊')


st.image(images_on_page, width=250, caption=captions)



