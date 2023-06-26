import pandas as pd

import pickle

from scipy.sparse import csr_matrix, save_npz
from pandas.api.types import CategoricalDtype

from surprise import SVD, Reader, Dataset 
from surprise.model_selection import cross_validate

from sklearn.neighbors import NearestNeighbors

from MovieRecommender.constants import *
from MovieRecommender.logging import logger

from MovieRecommender.entity.config_entity import CollaborativeFilteringConfig

class CollaborativeFiltering:
    def __init__(self, config = CollaborativeFilteringConfig):
        self.config = config

    def get_df(self, data_path):
        """
        Read a CSV file into a pandas DataFrame.

        Args:
            data_path (str): File path of the CSV file.

        Returns:
            pandas.DataFrame: DataFrame containing the data from the CSV file.
        """
        df = pd.read_csv(data_path)
        return df
    
    def filter_user(self,data):
        filtered_users = data['userId'].value_counts() > self.config.params_min_user_rating
        # getting the index of these users
        ind = filtered_users[filtered_users].index
        data = data[data['userId'].isin(ind)]
        return data

    def filter_movie(self,data):
        # figure out which movie got how much rating
        num_rating = data.groupby('title')['rating'].count().reset_index()
        num_rating.rename(columns={"rating":"num_of_rating"},inplace=True)

        final = data.merge(num_rating, on = 'title')
        final = final[final['num_of_rating'] >= self.config.params_min_movie_rating]
        return final
        # data.drop_duplicates(['userId','title'], inplace=True)

    def user_movie_matrix(self,data):
        rcLabel, vLabel = ('title','userId'), 'rating'
        rcCat = [CategoricalDtype(sorted(data[col].unique()), ordered=True) for col in rcLabel]
        rc = [data[column].astype(aType).cat.codes for column, aType in zip(rcLabel, rcCat)]
        mat = csr_matrix((data[vLabel], rc), shape=tuple(cat.categories.size for cat in rcCat))
        return mat, dict(zip(rcCat[0].categories, range(len(rcCat[0].categories))))
    
    def NearestNeighbors_model(self,mat):
        model = NearestNeighbors(algorithm = 'brute')
        model.fit(mat)

        return model
    
    def SVD_model(self,data):
        svd = SVD()
        reader = Reader()

        data = Dataset.load_from_df(data[['userId', 'title', 'rating']], reader)
        cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

        #sample full trainset
        trainset = data.build_full_trainset()
        # Get the user IDs and movie IDs in the same sequence as the SVD is calculated
        user_ids = trainset._raw2inner_id_users
        movie_ids = trainset._raw2inner_id_items
        # Train the algorithm on the trainset
        svd.fit(trainset)

        return svd,user_ids,movie_ids
    
    ## STEPS 
    def load_data_from_db(self):        
        self.ratings_df = self.get_df(data_path=self.config.ratings_data)
        self.movies_df = self.get_df(data_path = self.config.movies_data)
        self.movies_df  = self.movies_df[['title', 'movieId']]
        logger.info( f'Data loaded from DB' )

    def process_data(self):
        self.ratings_df = self.filter_user(self.ratings_df)
        self.ratings_df = self.ratings_df.merge(self.movies_df, on = "movieId")
        # print (self.rating_with_movies.shape)
        self.ratings_df = self.filter_movie(self.ratings_df)
        # print (self.ratings_df.shape)
        logger.info( f"Data processed # of {self.ratings_df[['userId','title']].nunique()}" )
    
    
    def save_user_movie_matrix(self,user_movie_matrix):
        # user_movie_matrix = self.create_user_movie_matrix()
        # pickle.dump(csr_matrix(cosine_sim), open(self.config.content_matrix,'wb'))
        save_npz(self.config.user_movie_matrix,user_movie_matrix)
        logger.info( f'user_movie_matrix saved in {self.config.user_movie_matrix}' )

    def save_model(self,model,model_path):
        pickle.dump(model, open(model_path,'wb'))
        logger.info( f'{model} saved in {model_path}' )

    def train_NearestNeighbors_model(self):
        user_movie_matrix, nn_item_indices = self.user_movie_matrix(self.ratings_df)
        model = self.NearestNeighbors_model(user_movie_matrix)
        print (f'shape of input martix: {user_movie_matrix.shape}')
        self.save_user_movie_matrix(user_movie_matrix)   
        self.save_model(model, self.config.nearest_neighbors_model) 
        pickle.dump(nn_item_indices, open(self.config.nn_item_indices,'wb'))
    
    def train_SVD_model(self):
        model,svd_user_indices,svd_item_indices = self.SVD_model(self.ratings_df)
        self.save_model(model, self.config.svd_model)
        pickle.dump(svd_user_indices, open(self.config.svd_user_indices,'wb'))
        pickle.dump(svd_item_indices, open(self.config.svd_item_indices,'wb'))
    
