import streamlit as st



st.title("Movie Recommendation System")

st.write("This is a Movie Recommendation System that recommends movies based on your ratings.")


st.header("Most Popular Movies Based on Genre")

genre = st.selectbox("Select a genre", ("Action", "Adventure", "Animation", "Comedy", "Crime",
                                "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                                "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
                                "War", "Western"))

if st.button('Show Recommendation', key="1"):
    st.write(f"Here are the top 10 {genre} based movies")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg')
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg')  
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")

    with col2:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")

    with col3:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")

    with col4:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")

    with col5:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg')
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg')
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]") 

st.header("Similar Movie Recommendation")
st.write("Select a movie from the list below to see its similar movies.")
movie = st.selectbox("Select a movie", ("The Shawshank Redemption", "The Godfather", "The Godfather: Part II",
                                "The Dark Knight", "The Dark Knight Rises", "The Good, the Bad and the Ugly",
                                "The Good, the Bad and the Ugly 2"))

if st.button('Show Recommendation', key="2"):
    st.write(f"Here are the top 10 movies similar to {movie}")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg')
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg')  
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")

    with col2:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")

    with col3:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")

    with col4:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")

    with col5:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg')
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg')
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]") 

st.header("Movie Recommendation Based on an Existing User")

col1, col2 = st.columns(2)
with col1:
    user = st.selectbox("Select an existing user", ("John Doe", "Jane Doe"))
with col2:
    movie = st.selectbox("Select a movie", ("The Shawshank Redemption", "The Godfather", "The Godfather: Part II",))

if st.button('Show Recommendation', key="3"):
    # st.text(f"Here are the top 10 movies similar to {movie}")
    st.write(f"Here are the top 10 recommedations for: {user} based on the movie: {movie}")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg')
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg')  
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")

    with col2:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")

    with col3:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")

    with col4:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg') 
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")

    with col5:
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg')
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]")
        st.image('http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg')
        st.markdown(f"The Shawshank Redemption [Rating : 5/10]") 


