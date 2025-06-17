import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
import random

# Loading the dataset
df = pd.read_csv("zomato.csv")
# Mapping cuisines to sample popular dishes
sample_dishes = {
    "North Indian": ["Butter Chicken", "Paneer Tikka", "Butter Naan"],
    "South Indian": ["Masala Dosa", "Idli Sambar", "Filter Coffee"],
    "Chinese": ["Hakka Noodles", "Manchurian", "Spring Roll"],
    "Italian": ["Margherita Pizza", "Pasta Alfredo", "Tiramisu"],
    "Mexican": ["Tacos", "Burrito", "Nachos"],
    "Bakery": ["Chocolate Cake", "Croissant", "Cheesecake"],
    "Cafe": ["Espresso", "Cappuccino", "Brownie"]

}
def get_sample_dishes(cuisine_type):
    dishes = []
    for cuisine in sample_dishes:
        if cuisine.lower() in cuisine_type.lower():
            dishes = sample_dishes[cuisine]
            break
    if not dishes:
        dishes = ["Chef's Special", "Seasonal Dish"]  # default if cuisine not matched
    return ", ".join(dishes)





# Define some standard times
opening_times = ["8:00 AM", "9:00 AM", "10:00 AM", "11:00 AM"]
closing_times = ["10:00 PM", "11:00 PM", "11:30 PM", "12:00 AM"]

# Add columns randomly
df['opening_time'] = [random.choice(opening_times) for _ in range(len(df))]
df['closing_time'] = [random.choice(closing_times) for _ in range(len(df))]



# Combining relevant features into a single text column for TF-IDF vectorization
df['combined_features'] = df['restaurant_type'] + ' ' + df['cuisine_type'] + ' ' + df['area']

# Creating a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()  #It means that  creating an instance of the TfidfVectorizer class from scikit-learn.

# Fit and transform the vectorizer on  combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Calculating the cosine similarity between restaurant descriptions
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(user_type, user_cuisine, user_location, top_n, price_filter):

    # we need to create a user profile based on preferences
    user_profile = f"{user_type} {user_cuisine} {user_location}"

    # we need to Filter the dataset to include only restaurants in the specified location
    filtered_df = df[df['area'] == user_location]

    # Filter by price
    if price_filter == "Low (₹0-400)":
        filtered_df = filtered_df[filtered_df['avg_cost'] <= 400]
    elif price_filter == "Moderate (₹400-800)":
        filtered_df = filtered_df[(filtered_df['avg_cost'] > 400) & (filtered_df['avg_cost'] <= 800)]
    elif price_filter == "High (₹800+)":
        filtered_df = filtered_df[filtered_df['avg_cost'] > 800]
    # else: if "Select price range" selected, no filtering by price

    if filtered_df.empty:
        return pd.Series([])  # Returns an empty Series if no matching restaurants are found

    # Transforming the user profile using the TF-IDF vectorizer
    user_profile_vector = tfidf_vectorizer.transform([user_profile])

    # Calculating the cosine similarity between the user profile and restaurant descriptions in the filtered dataset
    cosine_scores = linear_kernel(user_profile_vector, tfidf_vectorizer.transform(filtered_df['combined_features']))

    # Get the top 10 recommended restaurants based on cosine similarity
    restaurant_indices = cosine_scores.argsort()[0][-top_n:][::-1]

    recommended_restaurants = filtered_df.iloc[restaurant_indices]

    return recommended_restaurants

st.title("Restaurant Recommendation App")

# Custom CSS for better matching dropdown background
st.markdown("""
<style>
div[data-baseweb="select"] > div {
    background-color: rgba(30, 30, 30, 0.8) !important;
    border-radius: 8px !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)



# Fetch dropdown options
restaurant_types = sorted(df['restaurant_type'].dropna().unique().tolist())
cuisine_types = sorted(df['cuisine_type'].dropna().unique().tolist())
locations = sorted(df['area'].dropna().unique().tolist())


# Add "Select..." at the start
restaurant_types.insert(0, "Select a restaurant type")
cuisine_types.insert(0, "Select a cuisine type")
locations.insert(0, "Select a location")

# Dropdowns
user_type_input = st.selectbox("Select your preferred restaurant type:", restaurant_types)
user_cuisine_input = st.selectbox("Select your preferred cuisine type:", cuisine_types)
user_location_input = st.selectbox("Select your preferred location:", locations)

# ✅ Add Price Range Dropdown here
price_filter = st.selectbox(
    "Select your preferred price range:",
    ["Select price range", "Low (₹0-400)", "Moderate (₹400-800)", "High (₹800+)"]
)

# Top-N slider (Add this line)
top_n = st.slider("Select number of recommendations you want:", min_value=5, max_value=20, value=10)


# Change placeholder text color if not selected
if user_type_input.startswith("Select"):
    st.markdown("<style>div[data-baseweb='select'] span{color: grey !important;}</style>", unsafe_allow_html=True)
if user_cuisine_input.startswith("Select"):
    st.markdown("<style>div[data-baseweb='select'] span{color: grey !important;}</style>", unsafe_allow_html=True)
if user_location_input.startswith("Select"):
    st.markdown("<style>div[data-baseweb='select'] span{color: grey !important;}</style>", unsafe_allow_html=True)


# Button: Get Recommendations
if st.button("Get Recommendations"):
    if ("Select" in user_type_input) or ("Select" in user_cuisine_input) or ("Select" in user_location_input) or (
            price_filter == "Select price range"):
        st.warning("⚠️ Please select valid options from all dropdowns before getting recommendations.")
    else:
        recommended_restaurants = get_recommendations(user_type_input, user_cuisine_input, user_location_input, top_n,
                                                      price_filter)

    if ("Select" in user_type_input) or ("Select" in user_cuisine_input) or ("Select" in user_location_input):
        st.warning("⚠️ Please select valid options from all dropdowns before getting recommendations.")
    else:
        recommended_restaurants = get_recommendations(user_type_input, user_cuisine_input, user_location_input, top_n, price_filter)

        if not recommended_restaurants.empty:
            st.subheader(f"Recommended restaurants in {user_location_input} with {user_cuisine_input} cuisine and {user_type_input} type:")
            top_pick = recommended_restaurants.iloc[0]['restaurant_name']
            import time  # <-- Add this at the top (already you have it? good!)

            # Inside your "Get Recommendations" button logic
            for idx, row in recommended_restaurants.iterrows():
                restaurant = row['restaurant_name']
                opening_time = row['opening_time']
                closing_time = row['closing_time']
                cuisine_type = row['cuisine_type']

                dishes_preview = get_sample_dishes(cuisine_type)
                display_name = restaurant
                if restaurant == top_pick:
                    display_name += " → 🏆 Top Pick for You"

                query = f"{restaurant} {user_location_input}"
                maps_url = f"https://www.google.com/maps/dir/?api=1&destination={query.replace(' ', '+')}&travelmode=driving"

                st.markdown(f"""
                    <div style="margin-bottom:15px;">
                        <b>🍽️ {display_name}</b><br>
                        🕒 Open from {opening_time} to {closing_time}<br>
                        🍽️ Must-try: <i>{dishes_preview}</i><br>
                        <a href="{maps_url}" target="_blank" style="
                            display: inline-block;
                            margin-top: 5px;
                            padding: 8px 16px;
                            background-color: #32CD32;
                            color: white;
                            border-radius: 5px;
                            text-decoration: none;
                            font-weight: bold;
                        ">🧭 Navigate Now</a>
                    </div>
                """, unsafe_allow_html=True)

                time.sleep(0.2)  # 🌟 Add this line to create smooth appearance



        else:
            st.info("No matching restaurants found. Try selecting different options.")

# -------------- 🛑 End of Get Recommendations Button ----------------

# 🎲 Surprise Me! button should be OUTSIDE!
if st.button("🎲 Surprise Me!"):
    if ("Select" in user_type_input) or ("Select" in user_cuisine_input) or ("Select" in user_location_input):
        st.warning("⚠️ Please select valid options from all dropdowns before using Surprise Me.")
    else:
        # First, filter by location only
        filtered_df = df[df['area'].str.contains(user_location_input, case=False, na=False)]

        if not filtered_df.empty:
            random_restaurant = filtered_df.sample(1).iloc[0]
            random_name = random_restaurant['restaurant_name']
            random_area = random_restaurant['area']
            random_cuisine = random_restaurant['cuisine_type']
            random_type = random_restaurant['restaurant_type']

            st.success(f"🍽️ **{random_name}** in *{random_area}*")
            st.write(f"Type: {random_type} | Cuisine: {random_cuisine}")

            query = f"{random_name} {random_area}"
            maps_url = f"https://www.google.com/maps/dir/?api=1&destination={query.replace(' ', '+')}&travelmode=driving"
            st.markdown(f"[🧭 Navigate Now]({maps_url})", unsafe_allow_html=True)
        else:
            st.info("No matching restaurants found for your location. Try changing your selections.")
