#!/usr/bin/env python
# coding: utf-8

# # Genre Extraction from Spotify API
# 
# ### Objective: To access spotify API and extract genres of each artist

# ## 1) Installation of `spotipy` package
# - Only for first-time installation
# 
# 

# In[ ]:


#pip install spotipy


# ## 2)Importing of essential libraries in Python 3
# Libraries imported: 
# 1. `spotipy`, 
# 2. `itertools`, 
# 3. `pandas`, 
# 4. `time`, 
# 5. `requests`

# In[ ]:


#importing packages

import spotipy
import itertools
import pandas as pd
from time import sleep
import requests


# ## 3) Accessing Spotify API
# - Extract track audio features

# In[ ]:


# Get access to Spotify API to get track audio features (updated June 5, 2017). 
# Plug in the client ID and client secret you get from the registering with the Spotify API
from spotipy.oauth2 import SpotifyClientCredentials
client_credentials_manager = SpotifyClientCredentials(client_id='enter_your_client_id_here',
                                                      client_secret='enter_your_client_secret_here')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace = False


# ## 4) Removing Confounders
# - Christmas songs have been known to cause confounding of clusters
# - A function `RemoveChristmas` was written to remove any songs associated with Christmas

# In[ ]:


#function to remove christmas-related songs\n",
def RemoveChristmas(y):\n",
    y = [x for x in y if not ('christmas' in x)]\n",
    y = [x for x in y if not ('pop christmas' in x)]\n",
    y = [x for x in y if not ('soul christmas' in x)]\n",
    y = [x for x in y if not ('jazz christmas' in x)]\n",
    y = [x for x in y if not ('classical christmas' in x)]\n",
    y = [x for x in y if not ('world christmas' in x)]\n",
    y = [x for x in y if not ('folk christmas' in x)]\n",
    y = [x for x in y if not ('country christmas' in x)]\n",
    y = [x for x in y if not ('celtic christmas' in x)]\n",
    y = [x for x in y if not ('christmas product' in x)]\n",
    y = [x for x in y if not ('latin christmas' in x)]\n",
    y = [x for x in y if not (\"children's christmas\" in x)]\n",
    y = [x for x in y if not ('christian christmas' in x)]\n",
    y = [x for x in y if not ('heavy christmas' in x)]\n",
    y = [x for x in y if not ('indie christmas' in x)]\n",
    y = [x for x in y if not ('punk christmas' in x)]\n",
    return y"


# ## 5) Genre finder function
# - searches for genres `find_genres`

# In[ ]:


#genre finder function
def find_genres(x):
    if '+' in x:  # remove all the "+" characters. this was breaking the search
        x = x.replace('+', '')
    else:
        x = x
    result0 = sp.search(q='artist:' + x, type='artist')
    
    try:
        artist_1_genre = result0['artists']['items'][0]['genres']
    except IndexError:
        artist_1_genre = ['Index Error None']

    bwj0 = RemoveChristmas(artist_1_genre)
    artist_1_genre = bwj0

    if artist_1_genre == []:
        artist_1_genre = ['None']
    return artist_1_genre


# ## 6) Storing audio features data extracted from Spotify API
# - reading extracted data into a csv file for further analysis
# - first 5 rows of dataset printed below

# In[ ]:


#reading dataset
audio_set = pd.read_csv('SpotifyAudioFeaturesApril2019.csv', sep=',', index_col=[0])
audio_set.head()


# ## 7) Finding genres according to *artist_name* 
# 
# - Initializing an empty list for storage
# - Iterating through the dataframe using a *for* loop
# - Loops though *artist_name* column in dataframe and appends corresponding genres based on *artist_name*
# 

# In[ ]:


#execute function for each song in the dataframe
new_list = []
for artist in comparison_df['artist_name']:
    while True:
        #the following try/except statements deal with the Connection Error that
        #emerges when we try to request too much info from Spotify's endpoint
        try:
            new_list.append(find_genres(artist))
        except:
            continue
        break


# ## 8) Combining newly generated list of genre features to main dataframe

# In[ ]:


audio_set['genre'] = new_list


# ## 9) Saving the extracted dataset as a csv file

# In[ ]:


#saving the dataset
audio_set.to_csv('audioset_g.csv',index=True)

