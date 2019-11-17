import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity

# A few lines of code before importing the collaborative filtering function
anime = pd.read_csv('anime.csv')
rating = pd.read_csv('rating.csv')
rating.loc[rating.rating == -1, 'rating'] = np.NaN
combined = rating.merge(anime, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_abc', ''])
combined.rename(columns = {'rating_abc':'user_rating'}, inplace = True)
combined=combined[['user_id', 'name', 'user_rating']]
combined_small= combined[combined.user_id <= 20000]
collab = combined_small.pivot_table(index=['user_id'], columns=['name'], values='user_rating')
collab_normalize = collab.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
collab_normalize.fillna(0, inplace=True)
collab_normalize = collab_normalize.T
collab_normalize = collab_normalize.loc[:, (collab_normalize != 0).any(axis=0)]
collab_sparse = sp.sparse.csr_matrix(collab_normalize.values)
item_similarity = cosine_similarity(collab_sparse)
user_similarity = cosine_similarity(collab_sparse.T)
collab_user_sim = pd.DataFrame(user_similarity, index = collab_normalize.columns, columns = collab_normalize.columns)
collab_item_sim = pd.DataFrame(item_similarity, index = collab_normalize.index, columns = collab_normalize.index)

# Top 20 similar users are displayed 

def similar_movies(movie_names):
    count = 1
    print('Similar anime to {} are \n'.format(movie_names))
    
    for item in collab_item_sim.sort_values(by = movie_names, ascending = False).index[1:21]:
        
        print('Anime {} is {}'.format(count, item))
        count +=1  
        
        
# Top 10 similar users are displayed 

def similar_users(sim_user):
    print('Users with similar tastes are \n')
    
    sim_user_values = collab_user_sim.sort_values(by=sim_user, ascending=False).loc[:,sim_user].tolist()[1:11]
    
    sim_users = collab_user_sim.sort_values(by=sim_user, ascending=False).index[1:11]
    
    combine = zip(sim_users, sim_user_values,)
    for sim_user, sim in combine:
        print('Other users are {0}, How Similar = {1:.2f}'.format(sim_user, sim)) 
        
        
# This function calculates the weighted average of similar users
# to determine a potential rating for an input user.

def user_anime_rating(anime, user):
    sim_users = collab_user_sim.sort_values(by=user, ascending=False).index[1:1000]
    
    user_values = collab_user_sim.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:1000]
    weight_list = []
    rating_list = []
    
    for j, i in enumerate(sim_users):
        
        rating = collab.loc[i, anime]
        
        similarity = user_values[j]
        if np.isnan(rating):
            
            continue
        elif not np.isnan(rating):
            
            rating_list.append(rating*similarity)
            weight_list.append(similarity)
    return sum(rating_list)/sum(weight_list)    