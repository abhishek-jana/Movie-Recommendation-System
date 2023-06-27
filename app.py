import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
from src.MovieRecommender.pipeline.make_rocommendation import RecommendationPipeline
import numpy as np
from textwrap3 import wrap


@st.cache_resource
def make_recommendation():
    return RecommendationPipeline()


make_reco = make_recommendation()


def create_movie_layout(data):
    st.image(data["poster_path"])
    st.markdown(data["movie"])
    with st.expander("See details"):
        st.write(f"Rating: {data['rating']}/5")


def create_movie_layout_v2(data, num_columns=5):
    num_movies = len(data)
    num_rows = (num_movies + num_columns - 1) // num_columns
    image_width = 140

    for i in range(num_rows):
        col1, col2, col3, col4, col5 = st.columns(num_columns)
        cols = [col1, col2, col3, col4, col5]
        for j in range(num_columns):
            index = i * num_columns + j
            if index < num_movies:
                movie = data.iloc[index]
                col = cols[j]
                with col:
                    st.image(movie["poster_path"], width=image_width)
                    st.markdown(wrap(movie["movie"]), 35)
                    st.markdown(f"Rating: {movie['rating']}/5")
                    # with st.expander("See details"):
                    #     st.markdown(
                    #         f"Potential match: {np.round(100*movie['pred_rating']/5,2)} \%"
                    #     )
                    # Add additional details as needed
            else:
                # Add an empty column if there are fewer movies than columns
                st.empty()


def create_movie_layout_by_user(data):
    st.image(data["poster_path"])
    st.markdown(data["movie"])
    st.markdown(f"Potential match: {np.round(100*data['pred_rating']/5,2)} \%")


def create_movie_layout_by_user_v2(data, num_columns=5):
    num_movies = len(data)
    num_rows = (num_movies + num_columns - 1) // num_columns
    image_width = 140
    for i in range(num_rows):
        col1, col2, col3, col4, col5 = st.columns(num_columns)
        cols = [col1, col2, col3, col4, col5]
        for j in range(num_columns):
            index = i * num_columns + j
            if index < num_movies:
                movie = data.iloc[index]
                col = cols[j]
                with col:
                    st.image(movie["poster_path"], width=image_width)
                    st.markdown(wrap(movie["movie"]), 35)
                    st.markdown(
                        f"Potential match: {np.round(100*movie['pred_rating']/5,2)} \%"
                    )
                    # with st.expander("See details"):
                    #     st.markdown(
                    #         f"Potential match: {np.round(100*movie['pred_rating']/5,2)} \%"
                    #     )
                    # Add additional details as needed
            else:
                # Add an empty column if there are fewer movies than columns
                st.empty()


def render_spider_chart(cat, data):
    color_map = pc.qualitative.Plotly + pc.qualitative.Set3

    fig = go.Figure()

    for i, category in enumerate(cat):
        if i < len(data):
            color_index = i % len(color_map)
            color = color_map[color_index]

            next_data_index = (i + 1) % len(data)

            fig.add_trace(
                go.Scatterpolar(
                    r=[0, data[i], data[next_data_index], 0],
                    theta=[
                        category,
                        category,
                        cat[next_data_index],
                        cat[i],
                    ],
                    fill="toself",
                    fillcolor=color,
                    marker=dict(color=color),
                    line=dict(color=color),
                    name=category,
                )
            )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(data)])),
        showlegend=True,
        legend=dict(orientation="v", yanchor="bottom", y=0.0, xanchor="right", x=0.0),
        width=700,
        height=450,
        margin=dict(l=0, r=0, t=20, b=20),
    )

    return fig


# Example data
categories = make_reco.unique_genres
years = make_reco.movies_df["year"].astype(str).unique()
movies_list = list(make_reco.indices.keys())
existing_users = list(make_reco.svd_user_indices.keys())
existing_movies = list(make_reco.svd_item_indices.keys())


st.title("Movie Recommendation System")

st.write(
    "This is a Movie Recommendation System that recommends movies based on your ratings."
)

# Get user input or selection
calculate_most_popular = st.checkbox("Most Popular Movies Based on Genre")
calculate_content_based = st.checkbox("Content Based Movie Recommendation")
calculate_similar_user_based = st.checkbox("Similar User Based Movie Recommendation")
calculate_existing_user_based = st.checkbox(
    "Movie Recommendation Based on an Existing User"
)
calculate_movie_user_based = st.checkbox(
    "Movie Recommendation Based on an Existing User and Movies Watched"
)


st.header("Most Popular Movies Based on Genre")

with st.expander("See details"):
    st.write(
        """
            This is a content-based movie recommendation system that generates personalized movie recommendations based on the content and attributes of movies. The system analyzes the features and characteristics of movies and recommends similar movies to users based on their preferences.

            ## How it Works

            The content-based movie recommendation system follows these steps to generate personalized recommendations:

            1. **Movie Representation**: Each movie is represented by a set of relevant features or attributes, such as genre, director, cast, and plot keywords. These features capture the content and characteristics of the movies.

            2. **User Profile**: The system creates a user profile that reflects the preferences and interests of the user. This profile is built based on the user's historical movie ratings, genre preferences, and other relevant information.

            3. **Feature Extraction**: The system extracts the relevant features from both the movies and the user profile. This can be done using techniques like natural language processing (NLP) for extracting keywords or one-hot encoding for categorical features.

            4. **Similarity Calculation**: The system calculates the similarity between movies based on their feature vectors and the user profile. This can be done using similarity metrics such as cosine similarity or Euclidean distance.

            5. **Recommendation Generation**: Using the calculated similarity scores, the system generates movie recommendations for the user. It identifies movies that have high similarity scores with the movies the user has previously liked or rated.

            6. **Filtering and Ranking**: The system applies additional filtering criteria, such as removing movies the user has already watched or limiting the recommendations to a specific genre. The recommended movies are then ranked based on their similarity scores or other relevance measures.

            7. **Presentation of Recommendations**: The system presents the generated movie recommendations to the user through a user interface or any other suitable method.

            ## Advantages and Benefits

            - **Personalized Recommendations**: The system provides movie recommendations that align with the user's preferences and interests, enhancing their movie-watching experience.

            - **Content Understanding**: The system considers the content and attributes of movies, allowing it to recommend movies with similar themes, genres, or characteristics that the user might enjoy.

            - **Diverse Movie Selection**: The recommendations may introduce users to a diverse range of movies that align with their specific interests and expand their movie choices.

            ## Limitations and Challenges

            1. Limited Serendipity: Content-based recommendation systems heavily rely on item features, which can result in recommendations that are similar to the items the user has already interacted with, potentially lacking diversity and serendipitous discoveries.

            2. Overemphasis on Item Features: Content-based systems primarily consider item characteristics, often neglecting the social or collaborative aspects of recommendations. This can limit the ability to capture emerging trends or recommendations based on collective user behavior.

            3. Limited Exploration of User Preferences: Content-based systems rely on the user's historical interactions with items, which may not effectively capture evolving or changing user interests, resulting in recommendations that align with past behavior but fail to adapt to the user's current preferences.

            4. Cold Start Problem: Content-based systems face challenges when dealing with new users who have limited or no historical data. Without sufficient data on user preferences, it becomes challenging to provide accurate and personalized recommendations.

            5. Dependency on Item Metadata: Content-based systems heavily rely on the availability and quality of item metadata, such as tags, keywords, or descriptions. Incomplete, inaccurate, or biased metadata can negatively impact the quality and relevance of recommendations.

            6. Limited Diversity in Recommendations: Content-based systems may struggle to provide diverse recommendations, as they primarily focus on item features and similarities. This can lead to a narrower range of recommendations and potentially overlook less popular or niche items.

            7. Difficulty Handling Subjective Preferences: Content-based systems may struggle to capture subjective preferences or nuanced user tastes that cannot be easily represented by item features or metadata. This can limit the ability to provide personalized recommendations that align with individual user preferences.

            It's important to consider these limitations when implementing and evaluating content-based recommendation systems and to explore hybrid approaches or incorporate other recommendation techniques to overcome these challenges.
        """
    )

if calculate_most_popular:
    col1, col2 = st.columns(2)
    with col1:
        genre = st.multiselect(
            "Select genre(s)",
            [None] + categories,
            help="Select genre(s) or leave it blank",
        )

        if None in genre:
            # If None is selected, disable or clear other options
            genre = categories  # Clear the selection

        else:
            # Other options are allowed
            # Process the selected genres
            pass

    with col2:
        year = st.multiselect(
            "Select year(s)",
            [None] + list(years),
            help="Select year(s) or leave it blank",
        )
        if None in year:
            # If None is selected, disable or clear other options
            year = list(years)  # Clear the selection
            # Or disable other options

        else:
            # Other options are allowed
            # Process the selected genres
            pass

    if st.button("Show Recommendation", key="1"):
        st.write(f"Here are the top 10 most popular movies")
        popular_df = make_reco.popular_recs_filtered(n_top=10, years=year, genres=genre)
        popular_df = popular_df.sort_values("rating", ascending=False)
        create_movie_layout_v2(data=popular_df, num_columns=5)

st.header("Content Based Movie Recommendation")
with st.expander("See details"):
    st.write(
        """
            This is a movie recommendation system that generates similar movie recommendations based on collaborative filtering techniques, specifically item-based and user-based collaborative filtering. The system analyzes the similarities between movies or users and recommends movies that are similar to the ones the user has previously liked or rated.

            ## Item-based Collaborative Filtering

            Item-based collaborative filtering focuses on the similarity between items/movies. The system follows these steps to generate similar movie recommendations:

            1. **Item Similarity Calculation:** The system calculates the similarity between movies/items based on their attributes or features. This can be done using similarity metrics such as cosine similarity or Pearson correlation coefficient.

            2. **Neighborhood Selection:** The system selects a subset of movies/items that are most similar to a target movie/item. This subset represents the "neighborhood" of the target movie/item.

            3. **Recommendation Generation:** Using the ratings of the selected neighborhood, the system generates similar movie recommendations. It identifies movies/items that are highly rated by users who have also rated the target movie/item.

            4. **Filtering and Ranking:** The system applies additional filtering criteria, such as removing movies/items the user has already rated or limiting the recommendations to a specific genre. The recommended movies/items are then ranked based on their similarity scores or other relevance measures.

            5. **Presentation of Recommendations:** The system presents the generated similar movie recommendations to the user through a user interface or any other suitable method.

            ## User-based Collaborative Filtering

            User-based collaborative filtering focuses on the similarity between users. The system follows these steps to generate similar movie recommendations:

            1. **User Identification**: The system identifies the existing user based on their unique identifier or login credentials.
            2. **User Similarity Calculation**: The system calculates the similarity between the existing user and other users in the database. This is done by comparing their movie ratings using similarity metrics such as cosine similarity or Pearson correlation coefficient.
            3. **Neighborhood Selection**: The system selects a subset of users who are most similar to the existing user based on the calculated similarity scores. This subset represents the "neighborhood" of the existing user.
            4. **Recommendation Generation**: Using the movie ratings of the selected neighborhood, the system generates similar movie recommendations for the existing user. It identifies movies that are highly rated by users who are similar to the existing user.
            5. **Filtering and Ranking**: The system applies additional filtering criteria, such as removing movies the user has already rated or limiting the recommendations to a specific genre. The recommended movies are then ranked based on their average ratings or other relevance measures.
            6. **Presentation of Recommendations**: The system presents the generated similar movie recommendations to the existing user through a user interface or any other suitable method.

            ## Advantages and Benefits

            - **Personalized Recommendations**: The system provides similar movie recommendations based on the user's preferences, helping users discover movies that are similar to the ones they already enjoy.
            - **Diverse Movie Selection**: The recommendations may introduce users to movies from different genres or with similar themes, expanding their movie-watching experience.
            - **Improved User Engagement**: By offering relevant and interesting movie suggestions, the system increases user engagement and satisfaction, leading to increased user retention.

            ## Limitations and Challenges

            - **Cold Start Problem**: If the user has limited or no historical ratings, it becomes challenging to generate accurate recommendations. Additional techniques like content-based filtering or hybrid approaches may be employed to address this issue.
            - **Data Sparsity**: In scenarios where users have sparse rating histories or when the available data is limited, the system may struggle to find a sufficient number of similar movies or users for
            - **Scalability**: As the movie or user database grows, the computational complexity of calculating similarities and generating recommendations for a large number of movies or users increases. Efficient algorithms and distributed computing techniques may be required to handle scalability.
            - **Popularity Bias**: Item-based Collaborative Filtering tends to recommend popular items more frequently, as they have a higher chance of being rated by multiple users. This can lead to a lack of diversity in recommendations and overlook niche or lesser-known items.
            - **Lack of Serendipity**: The method relies heavily on user ratings and similarities, which may not capture serendipitous recommendations that could introduce users to new and unexpected items outside their usual preferences.
            - **Limited Contextual Information**: Collaborative Filtering primarily focuses on item similarities based on user ratings, disregarding contextual information such as temporal or situational factors that may influence user preferences.
        """
    )

if calculate_content_based:
    st.write("Select a movie from the list below to see its similar movies.")
    movie = st.selectbox("Select a movie", movies_list, key="10")

    if st.button("Show Recommendation", key="2"):
        st.write(f"Here are the top 10 movies similar to {movie}")
        content_reco_df = make_reco.content_recommendations(movie)
        create_movie_layout_v2(data=content_reco_df, num_columns=5)

st.header("Similar User Based Movie Recommendation")

if calculate_similar_user_based:
    st.write("Select a movie from the list below to see its similar movies.")
    movie_nn = st.selectbox("Select a movie", movies_list, key="20")

    if st.button("Show Recommendation", key="3"):
        st.write(f"Here are the top 10 movies other users are watching")
        neighbour_reco_df = make_reco.recommend_movie_neighbour(movie_nn)
        create_movie_layout_v2(data=neighbour_reco_df, num_columns=5)

st.header("Movie Recommendation Based on an Existing User")
with st.expander("See details"):
    st.write(
        """
            This is a movie recommendation system that generates personalized movie recommendations based on the preferences of an existing user. The system utilizes collaborative filtering techniques to analyze the user's historical movie ratings and recommend movies that are likely to be of interest to them.

            ## How it Works

            The movie recommendation system follows these steps to generate personalized recommendations for an existing user:

            1. **User Identification**: The system identifies the existing user based on their unique identifier or login credentials.

            2. **User Ratings**: The system retrieves the user's historical movie ratings from the database or user profile. These ratings serve as a basis for understanding the user's preferences.

            3. **Similarity Calculation**: The system calculates the similarity between the existing user and other users in the database. This is done by comparing their movie ratings using similarity metrics such as cosine similarity or Pearson correlation coefficient.

            4. **Neighborhood Selection**: The system selects a subset of users who are most similar to the existing user based on the calculated similarity scores. This subset is often referred to as the "neighborhood."

            5. **Recommendation Generation**: Using the ratings of the selected neighborhood, the system generates movie recommendations for the existing user. This can be done using various collaborative filtering techniques, such as user-based or item-based collaborative filtering.

            6. **Filtering and Ranking**: The system applies additional filtering criteria, such as removing movies the user has already rated or limiting the recommendations to a certain genre. The recommended movies are then ranked based on predicted ratings or other relevance measures.

            7. **Presentation of Recommendations**: The system presents the generated movie recommendations to the existing user. This can be done through a user interface, email notification, or any other suitable method.

            ## Advantages and Benefits

            - **Personalized Recommendations**: The system provides personalized movie recommendations tailored to the preferences of the existing user, enhancing their movie-watching experience.

            - **Discovery of New Movies**: The recommendations may introduce the user to movies they might not have discovered otherwise, broadening their movie choices.

            - **Improved User Engagement**: By offering relevant movie suggestions, the system increases user engagement and satisfaction, leading to increased user retention.

            ## Limitations and Challenges

            - **Cold Start Problem**: If the existing user has limited or no historical ratings, it becomes challenging to generate accurate recommendations. Additional techniques like content-based filtering or hybrid approaches may be employed to address this issue.

            - **Data Sparsity**: In scenarios where users have sparse rating histories or when the available data is limited, the system may struggle to find a sufficient number of similar users for accurate recommendations.

            - **Scalability**: As the user base grows, the computational complexity of calculating similarity and generating recommendations for a large number of users increases. Efficient algorithms and distributed computing techniques may be required to handle scalability.

            ## Examples and Applications

            Movie recommendation systems based on an existing user have been successfully deployed in various applications, including:

            - **Streaming Platforms**: Online streaming platforms like Netflix and Amazon Prime Video utilize these systems to suggest movies and TV shows to their subscribers based on their viewing history.

            - **E-commerce**: E-commerce platforms often employ recommendation systems to suggest related or complementary products to users based on their previous purchases or browsing behavior.

            - **Social Media**: Social media platforms may recommend movies or videos to users based on their interests and interactions with similar users.

        """
    )


if calculate_existing_user_based:
    user = st.selectbox("Select an existing user", existing_users, key="30")

    user_profile = make_reco.get_user_profile(user)
    with st.expander("See details"):
        # Render the Spider Chart
        spider_chart = render_spider_chart(
            list(user_profile.keys()), list(user_profile.values())
        )

        # Render the chart in Streamlit
        st.plotly_chart(spider_chart)

    # if st.button("Show Recommendation", key="4"):
    #     st.write(f"Based on user: {user}'s profile, we recommend the following movies.")
    #     top_user_df = make_reco.recommend_top_movie_user(user, n_top=10)
    #     col1, col2, col3, col4, col5 = st.columns(5)

    #     with col1:
    #         create_movie_layout_by_user(top_user_df.iloc[0])
    #         create_movie_layout_by_user(top_user_df.iloc[5])

    #     with col2:
    #         create_movie_layout_by_user(top_user_df.iloc[1])
    #         create_movie_layout_by_user(top_user_df.iloc[6])

    #     with col3:
    #         create_movie_layout_by_user(top_user_df.iloc[2])
    #         create_movie_layout_by_user(top_user_df.iloc[7])

    #     with col4:
    #         create_movie_layout_by_user(top_user_df.iloc[3])
    #         create_movie_layout_by_user(top_user_df.iloc[8])

    #     with col5:
    #         create_movie_layout_by_user(top_user_df.iloc[4])
    #         create_movie_layout_by_user(top_user_df.iloc[9])

    if st.button("Show Recommendation", key="4"):
        st.write(f"Based on user: {user}'s profile, we recommend the following movies.")

        # Organize the recommendations in a structured manner
        top_user_df = make_reco.recommend_top_movie_user(user, n_top=10)
        create_movie_layout_by_user_v2(data=top_user_df, num_columns=5)

st.header("Movie Recommendation Based on an Existing User and Movies Watched")

if calculate_movie_user_based:
    col1, col2 = st.columns(2)
    with col1:
        user_svd = st.selectbox("Select an existing user", existing_users, key="40")

    with col2:
        movie_svd = st.selectbox("Select a movie", existing_movies, key="50")

    if st.button("Show Recommendation", key="5"):
        st.write(
            f"Here are the top 10 recommedations for: {user_svd} based on the movie: {movie_svd}"
        )
        top_movie_user_df = make_reco.recommend_similar_movie_user(user_svd, movie_svd)
        create_movie_layout_by_user_v2(data=top_movie_user_df, num_columns=5)
        # col1, col2, col3, col4, col5 = st.columns(5)

        # with col1:
        #     create_movie_layout_by_user(top_movie_user_df.iloc[0])
        #     create_movie_layout_by_user(top_movie_user_df.iloc[5])

        # with col2:
        #     create_movie_layout_by_user(top_movie_user_df.iloc[1])
        #     create_movie_layout_by_user(top_movie_user_df.iloc[6])

        # with col3:
        #     create_movie_layout_by_user(top_movie_user_df.iloc[2])
        #     create_movie_layout_by_user(top_movie_user_df.iloc[7])

        # with col4:
        #     create_movie_layout_by_user(top_movie_user_df.iloc[3])
        #     create_movie_layout_by_user(top_movie_user_df.iloc[8])

        # with col5:
        #     create_movie_layout_by_user(top_movie_user_df.iloc[4])
        #     create_movie_layout_by_user(top_movie_user_df.iloc[9])
