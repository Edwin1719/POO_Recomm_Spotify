import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Asegúrate de que el archivo CSV esté en la misma carpeta o proporciona la ruta correcta
spotify = pd.read_csv("Most Streamed Spotify Songs 2024.csv", encoding='latin-1')

# Limpiar los datos (puedes usar la función que ya tienes) 
columns_to_clean = ['Spotify Streams', 'Spotify Popularity', 'Track Score', 
                    'YouTube Views', 'TikTok Posts', 'TikTok Likes']

for column in columns_to_clean:
    spotify[column] = spotify[column].astype(str).str.replace(',', '', regex=True)
    spotify[column] = spotify[column].str.replace(' ', '', regex=True)
    spotify[column] = pd.to_numeric(spotify[column], errors='coerce')
    median_value = spotify[column].median()
    spotify[column] = spotify[column].fillna(median_value)

class Recommender:
    def __init__(self, data):
        """Inicializa la clase con los datos del DataFrame"""
        self.data = data
        self.features = [
            'Track Score', 
            'Spotify Streams', 
            'Spotify Popularity',
            'YouTube Views', 
            'TikTok Posts', 
            'TikTok Likes'
        ]

        # TF-IDF para las columnas de texto
        self.vectorizer_artist = TfidfVectorizer()
        self.vectorizer_album = TfidfVectorizer()

        # Crear matrices TF-IDF y normalizar características numéricas
        self._prepare_data()

    def _prepare_data(self):
        """Prepara los datos para la recomendación"""
        artist_matrix = self.vectorizer_artist.fit_transform(self.data['Artist'].fillna('desconocido'))
        album_matrix = self.vectorizer_album.fit_transform(self.data['Album Name'].fillna('desconocido'))
        numeric_data_scaled = MinMaxScaler().fit_transform(self.data[self.features])

        # Combinar las matrices TF-IDF y numéricas
        self.data_tfidf = pd.concat([
            pd.DataFrame(artist_matrix.toarray()), 
            pd.DataFrame(album_matrix.toarray()), 
            pd.DataFrame(numeric_data_scaled)
        ], axis=1)

        # Crear una matriz de similitud de coseno
        self.similarity_matrix = cosine_similarity(self.data_tfidf)

    def recommend_similar_tracks(self, track_name, n=5):
        """Recomienda canciones similares basadas en la similitud del coseno"""
        idx = self.data.index[self.data['Track'] == track_name].tolist()[0]
        similar_indices = self.similarity_matrix[idx].argsort()[-(n+1):-1][::-1]
        return self.data.iloc[similar_indices][['Track', 'Artist', 'Spotify Popularity']]

    def get_top_tracks(self, n=10):
        """Devuelve las top n canciones más populares según Spotify Popularity"""
        return self.data.nlargest(n, 'Spotify Popularity')[['Track', 'Artist', 'Spotify Popularity']]

# Instanciar el recomendador
recommender = Recommender(spotify)
