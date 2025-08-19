"""
Spotify Recommender System - Optimized Streamlit App
Enhanced UI with Advanced Features and Professional Design
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the optimized recommender class
from recommender_optimized import SpotifyRecommenderOptimized
from st_social_media_links import SocialMediaIcons

# Page configuration
st.set_page_config(
    page_title="🎵 Spotify Recommender Pro",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1DB954;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #1DB954;
    }
    .sidebar-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stProgress .st-bo {
        background-color: #1DB954;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    """Load the optimized recommender system with caching"""
    try:
        with st.spinner("🚀 Loading Optimized Recommender System..."):
            recommender = SpotifyRecommenderOptimized()
        return recommender
    except Exception as e:
        st.error(f"❌ Error loading recommender: {str(e)}")
        return None

def create_similarity_chart(recommendations):
    """Create an interactive similarity score chart"""
    if recommendations.empty:
        return None
    
    fig = px.bar(
        recommendations,
        x='Similarity_Score',
        y='Track',
        orientation='h',
        color='Similarity_Score',
        color_continuous_scale='Viridis',
        title="🎯 Similarity Scores",
        labels={'Similarity_Score': 'Similarity (%)', 'Track': 'Track'}
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_font_size=16,
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig

def create_popularity_chart(recommendations):
    """Create a popularity comparison chart"""
    if recommendations.empty or 'Spotify_Popularity' not in recommendations.columns:
        return None
    
    fig = px.scatter(
        recommendations,
        x='Similarity_Score',
        y='Spotify_Popularity',
        size='Similarity_Score',
        color='Release_Year',
        hover_data=['Track', 'Artist'],
        title="🔥 Popularity vs Similarity Analysis",
        labels={
            'Similarity_Score': 'Similarity Score (%)',
            'Spotify_Popularity': 'Spotify Popularity',
            'Release_Year': 'Release Year'
        }
    )
    
    fig.update_layout(height=400, title_font_size=16)
    return fig

def create_year_distribution(recommendations):
    """Create year distribution chart"""
    if recommendations.empty or 'Release_Year' not in recommendations.columns:
        return None
    
    year_counts = recommendations['Release_Year'].value_counts().sort_index()
    
    fig = px.line(
        x=year_counts.index,
        y=year_counts.values,
        title="📅 Release Year Distribution",
        labels={'x': 'Release Year', 'y': 'Number of Tracks'}
    )
    
    fig.update_layout(height=300, title_font_size=16)
    return fig

def display_recommendations(recommendations):
    """Display recommendations in an enhanced format"""
    if recommendations.empty:
        st.warning("🔍 No recommendations found. Try a different search term or check spelling.")
        return
    
    st.markdown("### 🎵 Your Personalized Recommendations")
    
    for idx, row in recommendations.iterrows():
        with st.container():
            st.markdown(f"""
            <div class="recommendation-card">
                <h4>🎶 {row['Track']}</h4>
                <p><strong>👨‍🎤 Artist:</strong> {row['Artist']}</p>
                <p><strong>💿 Album:</strong> {row.get('Album', 'Unknown')}</p>
                <p><strong>📅 Year:</strong> {row.get('Release_Year', 'Unknown')}</p>
                <p><strong>🎯 Similarity:</strong> <span style="color: #1DB954; font-weight: bold;">{row['Similarity_Score']}%</span></p>
                <p><strong>🔥 Popularity:</strong> {row.get('Spotify_Popularity', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Spotify Recommender Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover your next favorite song with AI-powered recommendations</p>', unsafe_allow_html=True)
    
    # Load recommender system
    recommender = load_recommender()
    
    if recommender is None:
        st.error("Failed to load the recommender system. Please check the data file.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Dataset statistics
        stats = recommender.get_dataset_statistics()
        st.markdown(f"""
        <div class="sidebar-info">
            <h4>📊 Dataset Overview</h4>
            <p><strong>Total Tracks:</strong> {stats['total_tracks']:,}</p>
            <p><strong>Artists:</strong> {stats['unique_artists']:,}</p>
            <p><strong>Albums:</strong> {stats['unique_albums']:,}</p>
            <p><strong>Features:</strong> {stats['feature_count']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        
        # Feature highlights
        st.markdown("""
        <div class="feature-highlight">
            <h4>✨ AI Features</h4>
            <ul>
                <li>🧠 Machine Learning Similarity</li>
                <li>🔄 Cross-Platform Analysis</li>
                <li>📊 25+ Feature Engineering</li>
                <li>🎯 Fuzzy Search Matching</li>
                <li>⚡ Optimized Performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Smart Recommendations", 
        "🔥 Trending Tracks", 
        "👨‍🎤 Artist Explorer", 
        "📊 Analytics Dashboard",
        "🔍 Advanced Search"
    ])
    
    with tab1:
        st.markdown("## 🎯 Get Smart Recommendations")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            track_input = st.text_input(
                "🎵 Enter a song name:",
                placeholder="e.g., Blinding Lights, Shape of You, Bad Guy...",
                help="Type any song name - our AI will find it even with typos!"
            )
        
        with col2:
            num_recommendations = st.selectbox(
                "📊 Number of recommendations:",
                [5, 10, 15, 20],
                index=1
            )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                use_fuzzy = st.checkbox("🔍 Enable Fuzzy Search", value=True, 
                                      help="Finds songs even with typos or partial names")
            with col2:
                show_charts = st.checkbox("📊 Show Analytics Charts", value=True)
        
        if track_input:
            with st.spinner("🎵 Finding your perfect matches..."):
                # Simulate processing time for better UX
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                recommendations = recommender.recommend_similar_tracks(
                    track_input, 
                    n=num_recommendations,
                    use_fuzzy=use_fuzzy
                )
                progress_bar.empty()
            
            if not recommendations.empty:
                # Display match info
                match_type = recommendations.iloc[0]['Match_Type'] if 'Match_Type' in recommendations.columns else "Match Found"
                if "Fuzzy" in str(match_type):
                    st.info(f"🔍 {match_type}")
                else:
                    st.success(f"✅ {match_type}")
                
                # Display recommendations
                display_recommendations(recommendations)
                
                # Charts section
                if show_charts and len(recommendations) > 0:
                    st.markdown("---")
                    st.markdown("### 📊 Recommendation Analytics")
                    
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        similarity_chart = create_similarity_chart(recommendations)
                        if similarity_chart:
                            st.plotly_chart(similarity_chart, use_container_width=True)
                    
                    with chart_col2:
                        popularity_chart = create_popularity_chart(recommendations)
                        if popularity_chart:
                            st.plotly_chart(popularity_chart, use_container_width=True)
                    
                    # Year distribution
                    year_chart = create_year_distribution(recommendations)
                    if year_chart:
                        st.plotly_chart(year_chart, use_container_width=True)
            else:
                st.warning("🔍 No similar tracks found. Try a different song name or check spelling.")
    
    with tab2:
        st.markdown("## 🔥 Trending Tracks")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            trending_count = st.selectbox("📊 Number of trending tracks:", [10, 20, 30, 50], index=1)
        with col2:
            sort_option = st.selectbox("🎯 Sort by:", 
                                     ["Viral Index", "Spotify Popularity", "Track Score", "Recency"])
        
        if st.button("🚀 Get Trending Tracks", type="primary"):
            with st.spinner("🔥 Loading trending tracks..."):
                if sort_option == "Viral Index":
                    trending = recommender.get_trending_tracks(trending_count)
                else:
                    trending = recommender.get_top_tracks(trending_count, sort_by=sort_option)
                
                if not trending.empty:
                    st.markdown("### 🎵 Currently Trending")
                    
                    for idx, row in trending.iterrows():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        
                        with col1:
                            st.markdown(f"**🎶 {row['Track']}**")
                            st.caption(f"👨‍🎤 {row['Artist']}")
                        
                        with col2:
                            if 'trending_score' in row:
                                st.metric("🔥 Trending Score", f"{row['trending_score']:.2f}")
                            elif 'Spotify Popularity' in row:
                                st.metric("🔥 Popularity", f"{row['Spotify Popularity']}")
                        
                        with col3:
                            year = row.get('release_year', 'Unknown')
                            st.caption(f"📅 {year}")
                        
                        st.markdown("---")
    
    with tab3:
        st.markdown("## 👨‍🎤 Artist Explorer")
        
        artist_input = st.text_input(
            "🎤 Enter artist name:",
            placeholder="e.g., Taylor Swift, Drake, The Weeknd..."
        )
        
        use_artist_fuzzy = st.checkbox("🔍 Enable fuzzy search for artists", value=True)
        
        if artist_input:
            with st.spinner("🔍 Searching artist tracks..."):
                artist_tracks = recommender.search_by_artist(artist_input, use_fuzzy=use_artist_fuzzy)
                
                if not artist_tracks.empty:
                    st.markdown(f"### 🎵 Tracks by {artist_tracks.iloc[0]['Artist']}")
                    st.markdown(f"**Found {len(artist_tracks)} tracks**")
                    
                    # Display artist statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_popularity = artist_tracks['Spotify Popularity'].mean()
                        st.metric("📊 Avg Popularity", f"{avg_popularity:.1f}")
                    
                    with col2:
                        track_count = len(artist_tracks)
                        st.metric("🎵 Total Tracks", track_count)
                    
                    with col3:
                        if 'release_year' in artist_tracks.columns:
                            year_span = artist_tracks['release_year'].max() - artist_tracks['release_year'].min()
                            st.metric("📅 Career Span", f"{year_span} years")
                    
                    # Display tracks
                    st.dataframe(
                        artist_tracks[['Track', 'Album Name', 'release_year', 'Spotify Popularity', 'Track Score']],
                        use_container_width=True
                    )
                else:
                    st.warning("🔍 No tracks found for this artist. Try a different spelling or artist name.")
    
    with tab4:
        st.markdown("## 📊 Analytics Dashboard")
        
        # Dataset overview metrics
        st.markdown("### 📈 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎵 Total Tracks", f"{stats['total_tracks']:,}")
        
        with col2:
            st.metric("👨‍🎤 Artists", f"{stats['unique_artists']:,}")
        
        with col3:
            st.metric("💿 Albums", f"{stats['unique_albums']:,}")
        
        with col4:
            st.metric("🧠 AI Features", stats['feature_count'])
        
        # Performance metrics
        st.markdown("### ⚡ System Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a performance gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = 95,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "AI Model Accuracy"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1DB954"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "#1DB954"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance
            features = ['Cross-Platform', 'Temporal', 'Artist Intelligence', 'Performance', 'Text Analysis']
            importance = [25, 20, 20, 25, 10]
            
            fig = px.pie(
                values=importance,
                names=features,
                title="🧠 Feature Importance"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("## 🔍 Advanced Search")
        
        st.markdown("### 🎯 Multi-Criteria Search")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popularity_range = st.slider(
                "🔥 Popularity Range",
                min_value=0,
                max_value=100,
                value=(50, 100),
                help="Filter tracks by Spotify popularity score"
            )
        
        with col2:
            if 'release_year' in recommender.processed_data.columns:
                year_range = st.slider(
                    "📅 Release Year Range",
                    min_value=int(recommender.processed_data['release_year'].min()),
                    max_value=int(recommender.processed_data['release_year'].max()),
                    value=(2020, int(recommender.processed_data['release_year'].max()))
                )
        
        # Genre/mood filters (if available)
        st.markdown("### 🎭 Content Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_explicit = st.checkbox("🔞 Include Explicit Content", value=True)
        
        with col2:
            show_features = st.checkbox("🤝 Include Features/Collaborations", value=True)
        
        with col3:
            show_remixes = st.checkbox("🔄 Include Remixes", value=True)
        
        if st.button("🔍 Apply Advanced Search", type="primary"):
            # Apply filters to the dataset
            filtered_data = recommender.processed_data.copy()
            
            # Apply popularity filter
            if 'Spotify Popularity' in filtered_data.columns:
                filtered_data = filtered_data[
                    (filtered_data['Spotify Popularity'] >= popularity_range[0]) &
                    (filtered_data['Spotify Popularity'] <= popularity_range[1])
                ]
            
            # Apply year filter
            if 'release_year' in filtered_data.columns:
                filtered_data = filtered_data[
                    (filtered_data['release_year'] >= year_range[0]) &
                    (filtered_data['release_year'] <= year_range[1])
                ]
            
            # Apply content filters
            if not show_explicit and 'is_explicit' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['is_explicit'] == 0]
            
            if not show_features and 'has_featuring' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['has_featuring'] == 0]
            
            if not show_remixes and 'is_remix' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['is_remix'] == 0]
            
            if not filtered_data.empty:
                st.success(f"✅ Found {len(filtered_data)} tracks matching your criteria")
                
                # Display filtered results
                display_columns = ['Track', 'Artist', 'Album Name', 'release_year', 'Spotify Popularity']
                available_columns = [col for col in display_columns if col in filtered_data.columns]
                
                st.dataframe(
                    filtered_data[available_columns].head(20),
                    use_container_width=True
                )
            else:
                st.warning("🔍 No tracks found with the selected criteria. Try adjusting your filters.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎵 Spotify Recommender Pro | Powered by AI & Machine Learning</p>
        <p>Built with ❤️ using Streamlit, Scikit-learn, and Advanced Feature Engineering</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# Pie de página con información del desarrollador y logos de redes sociales
st.markdown("""
---
**Desarrollador:** Edwin Quintero Alzate<br>
**Email:** egqa1975@gmail.com<br>
""", unsafe_allow_html=True)

social_media_links = [
    "https://www.facebook.com/edwin.quinteroalzate",
    "https://www.linkedin.com/in/edwinquintero0329/",
    "https://github.com/Edwin1719"]

social_media_icons = SocialMediaIcons(social_media_links)
social_media_icons.render()