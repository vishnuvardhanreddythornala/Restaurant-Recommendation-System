import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Load the dataset
df = pd.read_csv("D:\RestaurantRec\zomato.csv")

# Combine relevant features into a single text column for TF-IDF vectorization
df['combined_features'] = df['restaurant_type'] + ' ' + df['cuisine_type'] + ' ' + df['area']

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the vectorizer on your combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Calculate the cosine similarity between restaurant descriptions
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(user_type, user_cuisine, user_location):
    # Create a user profile based on preferences
    user_profile = f"{user_type} {user_cuisine} {user_location}"

    # Filter the dataset to include only restaurants in the specified location
    filtered_df = df[df['area'] == user_location]

    if filtered_df.empty:
        return pd.Series([])  # Return an empty Series if no matching restaurants are found

    # Transform the user profile using the TF-IDF vectorizer
    user_profile_vector = tfidf_vectorizer.transform([user_profile])

    # Calculate the cosine similarity between the user profile and restaurant descriptions in the filtered dataset
    cosine_scores = linear_kernel(user_profile_vector, tfidf_vectorizer.transform(filtered_df['combined_features']))

    # Get the top 10 recommended restaurants based on cosine similarity
    restaurant_indices = cosine_scores.argsort()[0][-10:][::-1]
    recommended_restaurants = filtered_df.iloc[restaurant_indices]['restaurant_name']

    return recommended_restaurants

st.title("Restaurant Recommendation App")

user_type_input = st.text_input("Enter your preferred restaurant type:")
user_cuisine_input = st.text_input("Enter your preferred cuisine type:")
user_location_input = st.text_input("Enter your preferred location:")

if st.button("Get Recommendations"):
    recommended_restaurants = get_recommendations(user_type_input, user_cuisine_input, user_location_input)

    if not recommended_restaurants.empty:
        st.subheader(f"Recommended restaurants in {user_location_input} with {user_cuisine_input} cuisine and {user_type_input} type:")
        for restaurant in recommended_restaurants:
            st.write(restaurant)
    else:
        st.write("No matching restaurants found.")
