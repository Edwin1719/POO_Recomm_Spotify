import streamlit as st
from recommender import recommender  # Importa el recomendador
from st_social_media_links import SocialMediaIcons

# Función para la aplicación de Streamlit
def main():
    # Mostrar el logo en la parte superior izquierda
    logo_url = "https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg"  # URL del logo de Spotify
    st.image(logo_url, width=150, use_column_width=False)

    st.title("Songs Recommender")

    # Entrada del usuario en la barra lateral
    track_name = st.sidebar.text_input("Enter the name of a song:")

    # Número de recomendaciones en la barra lateral
    num_recommendations = st.sidebar.slider("Number of recommendations:", 1, 10, 5)

    # Botón para obtener recomendaciones
    if st.sidebar.button("Get Recommendations"):
        if track_name:
            try:
                recommendations = recommender.recommend_similar_tracks(track_name, n=num_recommendations)
                st.write("Recommendaciones:")
                st.dataframe(recommendations)
            except IndexError:
                st.write("Canción no encontrada en la base de datos.")
        else:
            st.write("Ingrese el nombre de una canción para obtener recomendaciones.")

    # Mostrar las top n canciones
    st.subheader("Top Songs")
    num_top_tracks = st.sidebar.slider("Number of top songs:", 1, 20, 10)
    top_tracks = recommender.get_top_tracks(n=num_top_tracks)
    st.dataframe(top_tracks)

if __name__ == "__main__":
    main()

# Pie de página con información del desarrollador y logos de redes sociales
st.markdown("""
---
**Desarrollador:** Edwin Quintero Alzate<br>
**Email:** egqa1975@gmail.com<br>
""")

social_media_links = [
    "https://www.facebook.com/edwin.quinteroalzate",
    "https://www.linkedin.com/in/edwinquintero0329/",
    "https://github.com/Edwin1719"]

social_media_icons = SocialMediaIcons(social_media_links)
social_media_icons.render()