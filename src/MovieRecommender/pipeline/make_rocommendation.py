import pandas as pd
from scipy.sparse import load_npz
import json
import numpy as np
import pickle
import os
from surprise.dataset import Trainset


class RecommendationPipeline:
    def __init__(self):
        self.movies_df = pd.read_csv(
            os.path.join("artifacts", "data_preparation", "final_data", "movies.csv")
        )[["title", "movieId", "tmdbId", "genres", "poster_path"]]
        self.movies_df["year"] = (
            self.movies_df["title"].str.extract(r"\((\d+)\)").fillna(-1)
        )
        self.ratings_df = pd.read_csv(
            os.path.join("artifacts", "data_preparation", "final_data", "ratings.csv")
        )
        self.unique_genres = json.load(
            open(
                os.path.join(
                    "artifacts",
                    "data_preparation",
                    "final_data",
                    "unique_categories.json",
                ),
                "rb",
            )
        )
        self.cosine_sim = load_npz(
            os.path.join("artifacts", "content_based_model", "content_matrix.npz")
        )
        self.svd_model = pickle.load(
            open(
                os.path.join(
                    "artifacts", "collaborative_filtering_model", "svd_model.pkl"
                ),
                "rb",
            )
        )
        self.svd_user_indices = pickle.load(
            open(
                os.path.join(
                    "artifacts", "collaborative_filtering_model", "svd_user_indices.pkl"
                ),
                "rb",
            )
        )
        self.svd_item_indices = pickle.load(
            open(
                os.path.join(
                    "artifacts", "collaborative_filtering_model", "svd_item_indices.pkl"
                ),
                "rb",
            )
        )
        self.indices = pickle.load(
            open(
                os.path.join(
                    "artifacts", "collaborative_filtering_model", "nn_item_indices.pkl"
                ),
                "rb",
            )
        )
        self.user_movie_matrix = load_npz(
            os.path.join(
                "artifacts", "collaborative_filtering_model", "user_movie_matrix.npz"
            )
        )
        self.nn_model = pickle.load(
            open(
                os.path.join(
                    "artifacts",
                    "collaborative_filtering_model",
                    "nearest_neighbors_movie.pkl",
                ),
                "rb",
            )
        )

    def create_ranked_df(self, movies, reviews, min_rating=10):
        """
        INPUT
        movies - the movies dataframe
        reviews - the reviews dataframe

        OUTPUT
        ranked_movies - a dataframe with movies that are sorted by highest avg rating, more reviews,
                        then time, and must have more than 4 ratings
        """

        # Pull the average ratings and number of ratings for each movie
        C = reviews["rating"].mean()
        movie_ratings = reviews.groupby("movieId")["rating"]
        avg_ratings = movie_ratings.mean()  # R
        num_ratings = movie_ratings.count()  # v
        m = min_rating
        weighted_rating = ((avg_ratings * num_ratings) / (num_ratings + m)) + (
            (C * m) / (num_ratings + m)
        )

        rating_count_df = pd.DataFrame(
            {"num_ratings": num_ratings, "weighted_rating": weighted_rating}
        ).reset_index()

        # merge with the movies dataset
        movie_recs = movies.merge(rating_count_df, on="movieId")

        # filter out the movies that qualify for the chart
        ratings_filtered = movie_recs[movie_recs["num_ratings"] > m]

        # sort by top avg rating and number of ratings
        ranked_movies = ratings_filtered.sort_values(
            ["weighted_rating", "num_ratings"], ascending=False
        )

        return ranked_movies

    def watched_movies_by_user(self, user_id):
        trainset: Trainset = self.svd_model.trainset

        user_ratings = trainset.ur[
            user_id
        ]  # Get the user's ratings from the trainset.ur attribute
        watched_movies = [
            item_id for item_id, _ in user_ratings
        ]  # Extract the item IDs
        watched_movies = [
            key
            for key, value in self.svd_item_indices.items()
            for item_id in watched_movies
            if value == item_id
        ]
        return watched_movies

    def get_user_profile(self, user_id):
        for genre in self.unique_genres:
            self.movies_df[genre] = self.movies_df["genres"].apply(
                lambda x: 1 if genre in x else 0
            )

        # do the above outside the function
        watched_movies = self.watched_movies_by_user(user_id)
        watched_movie_genres = (
            self.movies_df[self.unique_genres][
                self.movies_df["title"].isin(watched_movies)
            ]
            .sum(axis=0)
            .to_dict()
        )
        return watched_movie_genres

    def get_avg_ratings(self, movie_lists):
        # Get the index of the movie that matches the title
        ratings = []
        for movie in movie_lists:
            idx = self.indices[movie]
            ratings.append(
                np.round(np.mean(self.user_movie_matrix.getrow(idx).data), 2)
            )
        return ratings

    def fetch_poster_url(self, movie_lists):
        poster_path_url = "https://image.tmdb.org/t/p/w500"
        urls = []
        for movie in movie_lists:
            urls.append(
                poster_path_url
                + self.movies_df["poster_path"][self.movies_df["title"] == movie].iloc[
                    0
                ]
            )
        return urls

    def popular_recs_filtered(self, n_top=10, years=None, genres=None):
        """
        REDO THIS DOC STRING

        INPUT:
        user_id - the user_id (str) of the individual you are making recommendations for
        n_top - an integer of the number recommendations you want back
        ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time
        years - a list of strings with years of movies
        genres - a list of strings with genres of movies

        OUTPUT:
        top_movies - a list of the n_top recommended movies by movie title in order best to worst
        """

        ranked_movies = self.create_ranked_df(self.movies_df, self.ratings_df)
        ranked_movies["year"] = (
            ranked_movies["title"].str.extract(r"\((\d+)\)").fillna(-1)
        )

        # Create new columns based on the number of unique genres
        genre_columns = []
        for genre in self.unique_genres:
            genre_columns.append(
                ranked_movies["genres"]
                .apply(lambda x: int(genre in x))
                .rename(f"{genre}")
            )

        # Concatenate the genre columns with the movies_df DataFrame
        df_concatenated = pd.concat([ranked_movies] + genre_columns, axis=1)

        # Filter movies based on years and genres if provided
        if years is not None and genres is not None:
            filtered_movies = df_concatenated[
                (df_concatenated["year"].isin(years))
                & (df_concatenated[genres].sum(axis=1) > 0)
            ]
        elif years is not None:
            filtered_movies = df_concatenated[df_concatenated["year"].isin(years)]
        elif genres is not None:
            filtered_movies = df_concatenated[df_concatenated[genres].sum(axis=1) > 0]
        else:
            filtered_movies = df_concatenated.copy()

        # Sort the filtered movies by rank and select the top n_top movies
        top_movies = filtered_movies["title"][:n_top]
        poster_url = self.fetch_poster_url(top_movies)
        avg_rating = self.get_avg_ratings(top_movies)
        return pd.DataFrame(
            {"movie": top_movies, "poster_path": poster_url, "rating": avg_rating}
        )

    # Function that takes in movie title as input and outputs most similar movies
    def content_recommendations(self, movie_name, n_top=10):
        movie_data = self.create_ranked_df(self.movies_df, self.ratings_df)
        # cosine_sim = pd.DataFrame(cosine_sim.todense())

        indices = pd.Series(
            data=list(self.movies_df.index), index=self.movies_df["title"]
        )

        # Get the index of the movie that matches the title
        idx = indices[movie_name]

        # Get the row vector of cosine similarity scores
        similarity_scores = self.cosine_sim[idx, :]

        # Convert the row vector to a dense array
        sim_scores_dense = similarity_scores.toarray()[0]

        # Enumerate the similarity scores with their indices
        sim_scores = list(enumerate(sim_scores_dense))

        # Sort the movies based on the similarity scores
        sim_scores.sort(key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1 : (n_top + 1)]

        # Get the movie indices
        ind = [x[0] for x in sim_scores]

        movies_list = self.movies_df.iloc[ind]["title"].tolist()
        movie_data = movie_data[movie_data["title"].isin(movies_list)]

        poster_url = self.fetch_poster_url(movie_data["title"].tolist())
        movie_data["poster_path"] = poster_url
        movie_data = movie_data.rename(
            columns={"weighted_rating": "rating", "title": "movie"}
        )
        movie_data["rating"] = movie_data["rating"].apply(lambda x: np.round(x, 2))
        return movie_data[["movie", "poster_path", "rating"]]

    def recommend_movie_neighbour(self, movie_name, n_top=10):
        # Get the index of the movie that matches the title
        idx = self.indices[movie_name]

        movie_list = []
        # movie_id = np.where(user_movie_matrix.tocoo().row == movie_name)[0][0]
        distance, suggestion = self.nn_model.kneighbors(
            self.user_movie_matrix[idx], n_neighbors=int(n_top) + 1
        )

        for movies_id in suggestion[0]:
            movie = next(
                key for key, val in self.indices.items() if val == movies_id
            )  # since we know the value is present
            movie_list.append(movie)

        poster_url = self.fetch_poster_url(movie_list)
        avg_rating = self.get_avg_ratings(movie_list)

        return pd.DataFrame(
            {
                "movie": movie_list[1:],
                "poster_path": poster_url[1:],
                "rating": avg_rating[1:],
            }
        )

    def recommend_top_movie_user(self, user_id, n_top=10):
        watched_movies = self.watched_movies_by_user(user_id)
        unwatched_movies = [
            key for key, _ in self.svd_item_indices.items() if key not in watched_movies
        ]
        predicted_ratings = {
            item_id: self.svd_model.predict(user_id, item_id).est
            for item_id in unwatched_movies
        }
        sorted_movies_dict = {
            item_id: predicted_ratings[item_id]
            for item_id in sorted(
                unwatched_movies, key=lambda x: predicted_ratings[x], reverse=True
            )
        }

        top_movies = list(sorted_movies_dict.items())[:n_top]

        # Separate the values into two lists
        movies, pred_rating = zip(*top_movies)
        poster_url = self.fetch_poster_url(movies)
        # avg_rating =  self.get_avg_ratings(movies)
        return pd.DataFrame(
            {"movie": movies, "poster_path": poster_url, "pred_rating": pred_rating}
        )

    def recommend_similar_movie_user(self, user_id, movie, topn=10):
        watched_movies = self.watched_movies_by_user(user_id)
        unwatched_movies = [
            key for key, _ in self.svd_item_indices.items() if key not in watched_movies
        ]
        # Step 3: Determine the number of similar movies to consider
        num_similar_movies = max(len(watched_movies) * 1.5, 20)

        # Step 1: Get recommendations from recommend_movie_neighbour
        recommendations = self.recommend_movie_neighbour(
            movie, n_top=num_similar_movies
        )["movie"].to_list()
        recommended_unwatched_movies = set(unwatched_movies) & set(recommendations)
        # Step 6: Predict ratings for unwatched movies
        predicted_ratings = {
            movie: self.svd_model.predict(user_id, movie).est
            for movie in recommended_unwatched_movies
        }
        sorted_movies_dict = {
            movie: predicted_ratings[movie]
            for movie in sorted(
                recommended_unwatched_movies,
                key=lambda x: predicted_ratings[x],
                reverse=True,
            )
        }

        top_movies = list(sorted_movies_dict.items())[:topn]

        # Separate the values into two lists
        movies, pred_rating = zip(*top_movies)
        poster_url = self.fetch_poster_url(movies)
        # avg_rating =  self.get_avg_ratings(movies)
        return pd.DataFrame(
            {"movie": movies, "poster_path": poster_url, "pred_rating": pred_rating}
        )
