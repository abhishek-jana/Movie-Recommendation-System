import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from MovieRecommender.constants import *
from MovieRecommender.logging import logger
from scipy.sparse import csr_matrix, save_npz
from MovieRecommender.entity.config_entity import ContentBasedModelConfig
import numpy as np

class ContentBasedModel:
    def __init__(self, config = ContentBasedModelConfig):
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

    def apply_literal_eval(self, data, features = ['genres','keywords']):
        for feature in features:
            data[feature] = data[feature].apply(literal_eval)

    
    def get_top_elements(self,data,features = ['genres','keywords']):
        for feature in features:
            data[feature] = data[feature].apply(lambda x: x[:3])


    def clean_director(self,data):
        data['director']=data['director'].apply(lambda x: x.lower().replace(' ','_'))
 

    def clean_top3(self, data,features = ['genres','keywords']):
        for feature in features:
            data[feature] = data[feature].apply(lambda x: [a.lower().replace(' ', '_') for a in x])


    def create_soup_column(self,data):
        data['soup'] = data['keywords'].apply(lambda x: ' '.join(x)) + ' ' + data['director'] + ' ' + data['genres'].apply(lambda x: ' '.join(x))

    
    
    ## STEPS 
    def load_data_from_db(self):
        self.movies_df = self.get_df(data_path = self.config.movies_data)
        self.movies_df  = self.movies_df[['genres', 'keywords','director']]
        logger.info( f'Data loaded from DB' )

    def process_data(self):
        self.apply_literal_eval(self.movies_df)
        self.get_top_elements(self.movies_df)
        self.clean_director(self.movies_df)
        self.clean_top3(self.movies_df)
        self.create_soup_column(self.movies_df)
        logger.info( f'Data processed' )

    def create_similarity_matrix(self):
        cv = CountVectorizer(stop_words='english')
        cv_matrix = cv.fit_transform(self.movies_df['soup']).astype('float16')
        cosine_sim = cosine_similarity(cv_matrix,dense_output=False)
        logger.info( f'Similarity matrix calculated' )
        return cosine_sim
    
    def save_similarity_matrix(self):
        cosine_sim = self.create_similarity_matrix()
        # pickle.dump(csr_matrix(cosine_sim), open(self.config.content_matrix,'wb'))
        save_npz(self.config.content_matrix,cosine_sim)
        logger.info( f'Similarity matrix saved in {self.config.content_matrix}' )
        