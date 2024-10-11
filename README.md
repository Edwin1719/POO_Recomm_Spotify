# POO_Recomm_Spotify

![Logo](https://www.pngkit.com/png/detail/22-225664_listen-on-spotify-png.png)

POO_Recomm_Spotify is a recommendation system built with object-oriented programming to analyze Spotify data. It efficiently suggests similar songs based on audio features and text attributes like artist and album names, using TF-IDF and cosine similarity. Optimized for precision, this tool provides top song recommendations at your fingertips

# FEATURES

-**Data Loading:** The project begins by loading the dataset Most Streamed Spotify Songs 2024.csv using the Pandas library. This CSV file contains essential information about various songs, including attributes such as Artist, Album Name, and various metrics related to streaming performance. Proper loading of this data is crucial as it serves as the foundation for the recommendation system.

-**Data Cleaning:** Once the data is loaded, the next step involves cleaning specific columns, such as Spotify Streams, Spotify Popularity, Track Score, YouTube Views, TikTok Posts, and TikTok Likes. The cleaning process includes removing commas and spaces, converting these columns to numeric types, and handling any missing values by replacing them with their respective median values. This ensures the dataset is consistent and ready for analysis.

-**Feature Selection:** After the data cleaning process, feature selection is conducted to identify the key attributes that will be used for recommendations. The selected features include Track Score, Spotify Streams, Spotify Popularity, YouTube Views, TikTok Posts, and TikTok Likes. This selection is essential as it determines which aspects of the data will influence the recommendations provided to the user.

-**Text Vectorization:** To effectively analyze text data, the project employs TfidfVectorizer on the Artist and Album Name columns. This step transforms these textual features into TF-IDF matrices, which capture the importance of each word relative to the entire dataset. Vectorization is crucial for enabling the model to understand and process the text data quantitatively.

-**Normalization:** Following vectorization, the numerical features undergo normalization using MinMaxScaler. This process scales the values of the selected features to a range between 0 and 1. Normalization is important because it ensures that no single feature disproportionately influences the recommendation algorithm, allowing for a more balanced analysis of similarities between songs.

-**Similarity Calculation:** The project then combines the TF-IDF matrices with the normalized numerical features to create a single DataFrame. Using this combined data, a cosine similarity matrix is calculated to measure the similarity between different tracks based on their features. This matrix is central to the recommendation engine, as it quantifies how closely related songs are to each other.

-**Recommendation Logic:** With the similarity matrix established, the project implements the recommendation logic that suggests similar tracks based on the user's input track name. The algorithm identifies the index of the input track, retrieves the indices of the most similar tracks, and returns their details. Additionally, a method is included to fetch the top tracks based on their Spotify Popularity, providing users with a curated list of popular songs.

-**Streamlit Application Development:** Finally, the application is built using Streamlit, creating a user-friendly interface that allows users to enter song names and select the number of recommendations. The app displays the recommended tracks alongside the top songs, ensuring a seamless user experience. To enhance engagement, social media links are incorporated into the footer, allowing users to connect with the developer on various platforms.

# TECHNOLOGIES USED

**Descripcion:** Integrating object-oriented programming (OOP) with tools like Pandas, Scikit-learn, and Streamlit improves the development of applications like recommendation algorithms by promoting modularity, reusability, and easier maintenance. OOP enables the creation of scalable and adaptable components that streamline data processing and machine learning workflows. This synergy enables businesses to develop interactive applications that effectively present information, facilitating faster decision-making and greater responsiveness to market changes. Ultimately, leveraging OOP fosters innovation and efficiency, giving businesses a competitive advantage in a rapidly evolving landscape.

-**Python:** The primary programming language used for developing the application, known for its readability and extensive libraries that facilitate data analysis and machine learning.

-**Pandas:** A data manipulation library in Python that simplifies loading, cleaning, and transforming datasets, enabling efficient handling of song data in the project.

-**Scikit-Learn:** A machine learning library that provides tools for data scaling, text vectorization, and calculating similarity, crucial for building the recommendation algorithm.

-**Streamlit:** An open-source framework for creating web applications in Python, used to develop the user interface of the recommender system with interactive features.

-**Jupyter Notebook:** A web application for creating documents with live code and visualizations, commonly used during development for prototyping and data exploration.

-**TfidfVectorizer:** A component from Scikit-Learn that converts text data into TF-IDF feature vectors, enabling analysis of textual similarities in the recommendation process.

-**Cosine Similarity:** A metric used to measure the similarity between songs based on their feature vectors, allowing the application to recommend closely related tracks.

-**DOCUMENTATION**

! https://layla-scheli.medium.com/iebs-proyecto-final-sistemas-de-recomendacion-2024-915d35a62385

! https://www.youtube.com/watch?v=TZixpP_AmMY
