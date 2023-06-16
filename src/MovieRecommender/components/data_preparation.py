import os
import json
import pandas as pd
import numpy as np
import urllib
import urllib.request as request
from MovieRecommender.constants import *
from MovieRecommender.logging import logger
from MovieRecommender.entity.config_entity import DataPreparationConfig


class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
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

    def drop_nan_vals(self, df):
        """
        Drop rows with missing values (NaN) from a DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame to drop missing values from.

        Returns:
            None (modifies df in-place).
        """
        return df.dropna(inplace=True)

    def drop_cols(self, df, columns):
        """
        Drop specified columns from a DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame to drop columns from.
            columns (list): List of column names to drop.

        Returns:
            None (modifies df in-place).
        """
        return df.drop(columns=columns, inplace=True)

    def merge_df(self, df1, df2, col_name):
        """
        Merge two DataFrames based on a common column.

        Args:
            df1 (pandas.DataFrame): First DataFrame to merge.
            df2 (pandas.DataFrame): Second DataFrame to merge.
            col_name (str): Name of the common column to merge on.

        Returns:
            pandas.DataFrame: Merged DataFrame.
        """
        merged_df = df1.merge(df2, on=col_name)
        return merged_df

    def merge_df_with_index(self, df1, df2):
        """
        Merge two DataFrames based on their index.

        Args:
            df1 (pandas.DataFrame): First DataFrame to merge.
            df2 (pandas.DataFrame): Second DataFrame to merge.

        Returns:
            pandas.DataFrame: Merged DataFrame.
        """
        merged_df = pd.merge(df1, df2, left_index=True, right_index=True)
        return merged_df

    def rename_cols(self, df, columns):
        """rename columns

        Args:
            df (pandas): pandas dataframe
            columns (dict): key val with key -> column to be replaced
                            val -> new name
        Returns:
            df: modified df
        """
        return df.rename(columns=columns, inplace=True)

    def reindex_columns(self, df, column_order):
        """
        Reindex the columns of a DataFrame according to the desired order.

        Args:
            df (pandas.DataFrame): The DataFrame to reindex.
            column_order (list): The desired order of columns.

        Returns:
            pandas.DataFrame: The DataFrame with reindexed columns.
        """
        # Reindex the columns
        df = df.reindex(columns=column_order)

        return df

    def unique_genres(self, df):
        """
        the total number of unique genres in a DataFrame's 'genres' column.

        Args:
            df (pandas.DataFrame): DataFrame containing a 'genres' column.

        Returns:
            int: Total number of unique genres.

        Usage:
            df = pd.DataFrame({'genres': [['Action', 'Adventure'], ['Comedy', 'Romance'], ['Drama']]})
            total_genres = count_unique_genres(df)
            print(total_genres)  # Output: 4
        """
        genre_column = df["genres"]
        unique_genres = set()

        for genres_list in genre_column:
            unique_genres.update(genres_list)

        # total_genres = len(unique_genres)
        return list(unique_genres)

    def save_dataframe_to_csv(self, df, path_to_file):
        """
        Save a DataFrame to a specified folder as a CSV file.

        Args:
            df (pandas.DataFrame): The DataFrame to save.
            file_name (str): The name of the CSV file (including the extension).

        Returns:
            None
        """

        # Save the DataFrame to CSV
        df.to_csv(path_to_file, index=False)

        logger.info(f"DataFrame saved successfully in '{path_to_file}'.")

    def save_dataframe_to_parquet(self, df, path_to_file):
        """
        Save a DataFrame to a specified folder as a parquet file.

        Args:
            df (pandas.DataFrame): The DataFrame to save.
            file_name (str): The name of the CSV file (including the extension).

        Returns:
            None
        """

        # Save the DataFrame to CSV
        df.to_parquet(path_to_file)

    def save_dataframe_to_feather(self, df, path_to_file):
        """
        Save a DataFrame to a specified folder as a parquet file.

        Args:
            df (pandas.DataFrame): The DataFrame to save.
            file_name (str): The name of the CSV file (including the extension).

        Returns:
            None
        """

        # Save the DataFrame to CSV
        df.to_feather(path_to_file)

    def fetch_movie_details_from_api(self, tmdb_id):
        movie_url = f"{self.config.source_URL}{tmdb_id}?api_key={API_KEY}"
        credits_url = f"{self.config.source_URL}{tmdb_id}/credits?api_key={API_KEY}"
        keywords_url = f"{self.config.source_URL}{tmdb_id}/keywords?api_key={API_KEY}"

        movie_data = {}

        try:
            with request.urlopen(movie_url) as response:
                movie_data = json.loads(response.read().decode("utf-8"))

            with request.urlopen(credits_url) as response:
                credits_data = json.loads(response.read().decode("utf-8"))
                crew = credits_data.get("crew", [])
                director = next(
                    (
                        member["name"]
                        for member in crew
                        if member.get("job") == "Director"
                    ),
                    np.nan,
                )
                movie_data["director"] = director

            with request.urlopen(keywords_url) as response:
                keywords_data = json.loads(response.read().decode("utf-8"))
                keywords = [
                    keyword["name"] for keyword in keywords_data.get("keywords", [])
                ]
                movie_data["keywords"] = keywords

        except urllib.error.HTTPError as e:
            logger.info(
                f"Error occurred while requesting movie data {tmdb_id}:", e.code
            )

        return movie_data

    def fetch_batch_movie_details(self, tmdb_ids):
        movie_data_list = []
        for tmdb_id in tmdb_ids:
            cache_file = Path(os.path.join(self.config.cache_dir, f"{tmdb_id}.json"))

            try:
                with open(cache_file, "r") as file:
                    movie_data = json.load(file)
            except FileNotFoundError:
                # If the cache file does not exist, fetch the movie details from the API
                movie_data = self.fetch_movie_details_from_api(tmdb_id)
                # Cache the movie details
                with open(cache_file, "w") as file:
                    json.dump(movie_data, file)

            movie_data_list.append(movie_data)

        return movie_data_list

    def get_movie_details(self, tmdb_id):
        cache_file = Path(os.path.join(self.config.cache_dir, f"{tmdb_id}.json"))
        try:
            with open(cache_file, "r") as file:
                movie_data = json.load(file)
        except FileNotFoundError:
            # If the cache file does not exist, fetch the movie details from the API
            movie_data = self.fetch_movie_details_from_api(tmdb_id)
            # Cache the movie details
            with open(cache_file, "w") as file:
                json.dump(movie_data, file)

        # Extract the desired fields from the movie data
        genres = [genre["name"] for genre in movie_data.get("genres", [])]
        overview = movie_data.get("overview", "")
        popularity = movie_data.get("popularity", 0)
        poster_path = movie_data.get("poster_path", "")
        vote_average = movie_data.get("vote_average", 0)
        vote_count = movie_data.get("vote_count", 0)
        director = movie_data.get("director", np.nan)
        keywords = movie_data.get("keywords", np.nan)

        data = {
            "genres": genres,
            "overview": overview,
            "popularity": popularity,
            "poster_path": poster_path,
            "vote_average": vote_average,
            "vote_count": vote_count,
            "director": director,
            "keywords": keywords,
        }

        return data

    def get_batch_movie_details(self, tmdb_ids):
        movie_data_list = self.fetch_batch_movie_details(tmdb_ids)

        result = []
        for movie_data in movie_data_list:
            try:
                tmdb_id = movie_data["id"]
            except KeyError:
                tmdb_id = None
            data = self.get_movie_details(tmdb_id)
            result.append(data)
        return result

    ## STEPS
    def load_data_from_db(self):
        self.movies_df = self.get_df(data_path=self.config.movies_data)
        self.links_df = self.get_df(data_path=self.config.links_data)
        self.ratings_df = self.get_df(data_path=self.config.ratings_data)
        logger.info(f"Data loaded from DB")

    def preprocess_data(self):
        self.links_movie_df = self.movies_df.merge(self.links_df, on="movieId")
        # drop NAN values
        self.links_movie_df.dropna(inplace=True, axis=0)
        # change tmdb type from float to int
        self.links_movie_df["tmdbId"] = self.links_movie_df["tmdbId"].astype(int)
        # drop timestamp from ratings_df
        self.ratings_df.drop(columns="timestamp", inplace=True)
        logger.info(f"Data preprocessed")

    def get_data_from_tmdb(self):
        self.tmdb_ids = self.links_movie_df.tmdbId.values
        self.movie_data = self.get_batch_movie_details(tmdb_ids=self.tmdb_ids)
        logger.info(f"Data loaded from TMDB")

    def prepare_final_data(self):
        self.tmdb_df = pd.DataFrame(self.movie_data)
        # merge tmdb data and links_movie data
        self.links_movie_df_merged = pd.merge(
            self.links_movie_df, self.tmdb_df, left_index=True, right_index=True
        )
        # drop NAN values
        self.links_movie_df_merged.dropna(axis=0, inplace=True)
        # this will create 2 separate genres, we will drop one of them
        self.links_movie_df_merged.drop(labels="genres_x", axis=1, inplace=True)
        # rename the other genres column
        self.links_movie_df_merged.rename(columns={"genres_y": "genres"}, inplace=True)
        self.final_df = self.links_movie_df_merged.merge(self.ratings_df, on="movieId")
        # Convert float64 columns to float32
        float_columns = ["popularity", "vote_average", "rating"]
        self.final_df[float_columns] = self.final_df[float_columns].astype("float32")

        # Convert int64 columns to int32
        int_columns = ["movieId", "imdbId", "tmdbId", "vote_count", "userId"]
        self.final_df[int_columns] = self.final_df[int_columns].astype("int32")

        # Define the desired column order
        column_order = [
            "movieId",
            "imdbId",
            "tmdbId",
            "title",
            "genres",
            "director",
            "overview",
            "vote_average",
            "vote_count",
            "popularity",
            "keywords",
            "userId",
            "rating",
            "poster_path",
        ]
        self.final_df = self.reindex_columns(self.final_df, column_order)
        logger.info(f"Final data prepared")

    def save_final_data_in_db(self):
        self.save_dataframe_to_feather(self.final_df, self.config.final_data_path)
        logger.info(
            f"Final data saved successfully as '{self.config.final_data_path}'."
        )
        unique_genres_list = self.unique_genres(self.links_movie_df_merged)
        with open(self.config.unique_category_path, "w") as file:
            json.dump(unique_genres_list, file)

        logger.info(
            f"Unique genres saved successfully as '{self.config.unique_category_path}'."
        )
