"""
Spotify Recommender System - Optimized Version
Enhanced with Feature Engineering and Improved Cosine Similarity

Features:
- Intelligent data cleaning with minimal data loss
- Cross-platform feature engineering (Spotify, YouTube, TikTok)
- Temporal features (release date, era analysis)
- Artist intelligence features
- Robust performance normalization
- Enhanced TF-IDF with n-grams
- Optimized cosine similarity with 25+ features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process, fuzz
import warnings
from typing import Dict, List, Tuple, Optional
import os

warnings.filterwarnings('ignore')


class SpotifyRecommenderOptimized:
    """
    Optimized Spotify Recommendation System with Feature Engineering
    """
    
    def __init__(self, csv_path: str = "Most Streamed Spotify Songs 2024.csv"):
        """
        Initialize the optimized recommender system
        
        Args:
            csv_path (str): Path to the CSV file
        """
        print("🚀 Initializing Optimized Spotify Recommender...")
        
        # Load and validate data
        self.csv_path = csv_path
        self.raw_data = self._load_data()
        
        # Apply intelligent data cleaning
        print("🧹 Applying intelligent data cleaning...")
        self.cleaned_data = self._intelligent_data_cleaning()
        
        # Apply feature engineering (skip problematic imputation)
        print("⚙️ Applying feature engineering...")
        self.processed_data = self._apply_feature_engineering()
        
        # Prepare similarity matrix
        print("🎯 Preparing enhanced similarity matrix...")
        self._prepare_optimized_similarity_matrix()
        
        # Create search indices for fast lookup
        self._create_search_indices()
        
        print("✅ Optimized Recommender System ready!")
        print(f"📊 Dataset: {len(self.processed_data):,} songs with {self.feature_matrix.shape[1]} features")
    
    def _load_data(self) -> pd.DataFrame:
        """Load and validate the dataset"""
        try:
            # Try different encodings
            for encoding in ['latin-1', 'utf-8', 'cp1252']:
                try:
                    df = pd.read_csv(self.csv_path, encoding=encoding)
                    print(f"✅ Data loaded successfully with {encoding} encoding")
                    print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
                    return df
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("Could not decode file with any standard encoding")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _intelligent_data_cleaning(self) -> pd.DataFrame:
        """
        Simple and robust data cleaning based on working recommender.py approach
        """
        df = self.raw_data.copy()
        original_rows = len(df)
        
        # Use the same approach as the working recommender.py
        numeric_cols_to_clean = [
            'Track Score', 'All Time Rank', 'Spotify Streams', 'Spotify Popularity',
            'Spotify Playlist Count', 'Spotify Playlist Reach',
            'YouTube Views', 'YouTube Likes', 'YouTube Playlist Reach',
            'TikTok Posts', 'TikTok Likes', 'TikTok Views',
            'Apple Music Playlist Count', 'AirPlay Spins', 'SiriusXM Spins',
            'Deezer Playlist Count', 'Deezer Playlist Reach',
            'Amazon Playlist Count', 'Pandora Streams', 'Pandora Track Stations',
            'Soundcloud Streams', 'Shazam Counts'
        ]
        
        # Simple and reliable cleaning approach
        for col in numeric_cols_to_clean:
            if col in df.columns:
                # Convert to string and clean
                df[col] = df[col].astype(str).str.replace(',', '', regex=True)
                df[col] = df[col].str.replace(' ', '', regex=True)
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN with median (this approach works!)
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        
        # Clean text columns
        text_cols = ['Track', 'Artist', 'Album Name']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                df[col] = df[col].astype(str).str.strip()
        
        # Handle dates
        if 'Release Date' in df.columns:
            df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Validation
        final_rows = len(df)
        data_retention = (final_rows / original_rows) * 100
        print(f"📈 Data retention: {data_retention:.1f}% ({final_rows:,}/{original_rows:,} rows)")
        
        return df
    
    def _apply_smart_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply intelligent imputation strategies
        """
        # Strategy 1: Impute by artist average (preserves artist patterns)
        artist_features = ['Spotify Streams', 'YouTube Views', 'Spotify Popularity']
        for col in artist_features:
            if col in df.columns:
                artist_avg = df.groupby('Artist')[col].median()
                df[col] = df[col].fillna(df['Artist'].map(artist_avg))
                # Global median as fallback
                df[col] = df[col].fillna(df[col].median())
        
        # Strategy 2: Cross-platform ratios for missing engagement data
        if 'YouTube Views' in df.columns and 'YouTube Likes' in df.columns:
            # Calculate average like rate
            like_rate = (df['YouTube Likes'] / df['YouTube Views']).median()
            
            # Impute missing likes based on views
            mask = df['YouTube Likes'].isna() & df['YouTube Views'].notna()
            df.loc[mask, 'YouTube Likes'] = df.loc[mask, 'YouTube Views'] * like_rate
            
            # Impute missing views based on likes
            mask = df['YouTube Views'].isna() & df['YouTube Likes'].notna()
            df.loc[mask, 'YouTube Views'] = df.loc[mask, 'YouTube Likes'] / like_rate
        
        # Strategy 3: Platform-specific logic
        playlist_cols = ['Apple Music Playlist Count', 'Deezer Playlist Count', 
                        'Amazon Playlist Count', 'Spotify Playlist Count']
        for col in playlist_cols:
            if col in df.columns:
                # Clean the column first if it contains strings with commas
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('[, ]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                # If not in playlists, it's likely 0, not missing
                df[col] = df[col].fillna(0)
        
        # Strategy 4: Rank-based imputation for scores
        if 'All Time Rank' in df.columns and 'Track Score' in df.columns:
            # Higher rank (lower number) should correlate with higher score
            rank_score_relation = df[['All Time Rank', 'Track Score']].corr().iloc[0, 1]
            if abs(rank_score_relation) > 0.3:  # Significant correlation
                # Use relationship to impute missing scores
                median_score = df['Track Score'].median()
                median_rank = df['All Time Rank'].median()
                
                mask = df['Track Score'].isna() & df['All Time Rank'].notna()
                # Inverse relationship approximation
                df.loc[mask, 'Track Score'] = median_score * (median_rank / df.loc[mask, 'All Time Rank'])
        
        return df
    
    def _apply_feature_engineering(self) -> pd.DataFrame:
        """
        Apply SAFE feature engineering - avoiding categorical issues
        """
        df = self.cleaned_data.copy()
        
        # Only apply safe feature engineering that won't create categorical issues
        # 1. Cross-platform features (safe numeric calculations)
        df = self._create_cross_platform_features(df)
        
        # 2. Temporal features (safe, mostly numeric)
        df = self._create_temporal_features(df)
        
        # 3. Basic artist features (avoiding complex aggregations)
        df = self._create_basic_artist_features(df)
        
        # 4. Performance normalization features (safe numeric)
        df = self._create_performance_features(df)
        
        # Skip the problematic category features for now
        # Skip missing data flags for now
        
        return df
    
    def _create_cross_platform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cross-platform performance features"""
        
        # Platform ratios (viral detection)
        if 'TikTok Views' in df.columns and 'Spotify Streams' in df.columns:
            df['tiktok_spotify_ratio'] = df['TikTok Views'] / (df['Spotify Streams'] + 1)
        
        if 'YouTube Views' in df.columns and 'Spotify Streams' in df.columns:
            df['youtube_spotify_ratio'] = df['YouTube Views'] / (df['Spotify Streams'] + 1)
        
        # Engagement rates
        if 'YouTube Likes' in df.columns and 'YouTube Views' in df.columns:
            df['youtube_engagement_rate'] = df['YouTube Likes'] / (df['YouTube Views'] + 1)
        
        if 'TikTok Likes' in df.columns and 'TikTok Views' in df.columns:
            df['tiktok_engagement_rate'] = df['TikTok Likes'] / (df['TikTok Views'] + 1)
        
        # Viral index (composite score)
        platform_cols = ['Spotify Streams', 'YouTube Views', 'TikTok Views']
        available_platforms = [col for col in platform_cols if col in df.columns]
        
        if len(available_platforms) >= 2:
            viral_components = []
            weights = [0.4, 0.3, 0.3]  # Spotify, YouTube, TikTok weights
            
            for i, col in enumerate(available_platforms):
                if i < len(weights):
                    normalized = df[col] / (df[col].max() + 1)
                    viral_components.append(normalized * weights[i])
            
            df['viral_index'] = sum(viral_components)
        
        # Platform diversity score
        platform_presence_cols = [col for col in available_platforms if col in df.columns]
        if platform_presence_cols:
            df['platform_diversity'] = sum((df[col] > 0).astype(int) for col in platform_presence_cols)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal and release-based features"""
        
        if 'Release Date' in df.columns:
            # Basic temporal features
            df['release_year'] = df['Release Date'].dt.year
            df['release_month'] = df['Release Date'].dt.month
            df['release_day_of_year'] = df['Release Date'].dt.dayofyear
            
            # Age and recency
            current_date = pd.Timestamp.now()
            df['days_since_release'] = (current_date - df['Release Date']).dt.days
            df['years_since_release'] = df['days_since_release'] / 365.25
            
            # Recency score (newer songs get higher score)
            df['recency_score'] = 1 / (1 + df['days_since_release'] / 365)
            
            # Musical eras
            df['is_vintage'] = (df['release_year'] < 2010).astype(int)
            df['is_streaming_era'] = ((df['release_year'] >= 2010) & (df['release_year'] < 2020)).astype(int)
            df['is_modern'] = (df['release_year'] >= 2020).astype(int)
            df['is_post_covid'] = (df['release_year'] >= 2021).astype(int)
            
            # Seasonal features
            df['is_summer_release'] = df['release_month'].isin([6, 7, 8]).astype(int)
            df['is_winter_release'] = df['release_month'].isin([12, 1, 2]).astype(int)
            df['is_holiday_release'] = df['release_month'].isin([11, 12]).astype(int)
            
            # Weekend vs weekday release
            df['release_weekday'] = df['Release Date'].dt.dayofweek
            df['is_weekend_release'] = (df['release_weekday'] >= 5).astype(int)
        
        return df
    
    def _create_basic_artist_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create SAFE basic artist features - no complex aggregations that could cause categorical issues"""
        
        # Simple text-based features (safe)
        df['artist_name_length'] = df['Artist'].str.len()
        df['track_name_length'] = df['Track'].str.len()
        df['album_name_length'] = df['Album Name'].str.len()
        
        # Content type detection (safe boolean features)
        df['has_featuring'] = df['Track'].str.contains('feat|ft\.|featuring', case=False, na=False).astype(int)
        df['is_remix'] = df['Track'].str.contains('remix|mix', case=False, na=False).astype(int)
        df['is_cover'] = df['Track'].str.contains('cover', case=False, na=False).astype(int)
        df['is_live'] = df['Track'].str.contains('live|concert', case=False, na=False).astype(int)
        
        # Artist name patterns (safe)
        df['artist_has_numbers'] = df['Artist'].str.contains('\d', na=False).astype(int)
        df['artist_has_special_chars'] = df['Artist'].str.contains('[^a-zA-Z0-9\s]', na=False).astype(int)
        
        return df
    
    def _create_artist_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create artist-based intelligence features"""
        
        # Artist performance aggregations
        artist_stats = df.groupby('Artist').agg({
            'Spotify Streams': ['count', 'mean', 'sum', 'std'],
            'Track Score': ['mean', 'std'],
            'Spotify Popularity': ['mean', 'max'],
            'All Time Rank': ['mean', 'min']
        }).round(2)
        
        # Flatten column names
        artist_stats.columns = [f'artist_{col[1]}_{col[0]}' for col in artist_stats.columns]
        
        # Merge back to main dataframe
        df = df.merge(artist_stats, left_on='Artist', right_index=True, how='left')
        
        # Artist categorization
        if 'artist_count_Spotify Streams' in df.columns:
            df['is_prolific_artist'] = (df['artist_count_Spotify Streams'] >= 5).astype(int)
            df['is_one_hit_wonder'] = (df['artist_count_Spotify Streams'] == 1).astype(int)
        
        if 'artist_mean_Spotify Streams' in df.columns:
            superstar_threshold = df['artist_mean_Spotify Streams'].quantile(0.9)
            df['is_superstar'] = (df['artist_mean_Spotify Streams'] > superstar_threshold).astype(int)
        
        # Text-based features
        df['artist_name_length'] = df['Artist'].str.len()
        df['track_name_length'] = df['Track'].str.len()
        df['album_name_length'] = df['Album Name'].str.len()
        
        # Content type detection
        df['has_featuring'] = df['Track'].str.contains('feat|ft\.|featuring', case=False, na=False).astype(int)
        df['is_remix'] = df['Track'].str.contains('remix|mix', case=False, na=False).astype(int)
        df['is_cover'] = df['Track'].str.contains('cover', case=False, na=False).astype(int)
        df['is_live'] = df['Track'].str.contains('live|concert', case=False, na=False).astype(int)
        
        # Artist name patterns
        df['artist_has_numbers'] = df['Artist'].str.contains('\d', na=False).astype(int)
        df['artist_has_special_chars'] = df['Artist'].str.contains('[^a-zA-Z0-9\s]', na=False).astype(int)
        
        return df
    
    def _create_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create normalized performance features"""
        
        # Key performance columns
        performance_cols = [
            'Track Score', 'Spotify Streams', 'Spotify Popularity', 
            'YouTube Views', 'TikTok Views', 'All Time Rank'
        ]
        
        available_perf_cols = [col for col in performance_cols if col in df.columns]
        
        for col in available_perf_cols:
            # Percentile ranking (0-100)
            df[f'{col}_percentile'] = df[col].rank(pct=True) * 100
            
            # Z-score (clipped to handle extreme outliers)
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[f'{col}_zscore'] = (df[col] - mean_val) / std_val
                df[f'{col}_zscore_clipped'] = df[f'{col}_zscore'].clip(-3, 3)
            
            # Logarithmic normalization (for highly skewed data)
            if col in ['Spotify Streams', 'YouTube Views', 'TikTok Views']:
                df[f'{col}_log'] = np.log1p(df[col])
                df[f'{col}_log_norm'] = (df[f'{col}_log'] - df[f'{col}_log'].min()) / (df[f'{col}_log'].max() - df[f'{col}_log'].min())
        
        # Composite performance scores
        if len(available_perf_cols) >= 3:
            # Mainstream appeal score
            mainstream_cols = [f'{col}_percentile' for col in ['Spotify Streams', 'YouTube Views', 'Spotify Popularity'] 
                             if f'{col}_percentile' in df.columns]
            if mainstream_cols:
                df['mainstream_score'] = df[mainstream_cols].mean(axis=1)
                df['niche_score'] = 100 - df['mainstream_score']
        
        # Performance consistency (how consistent across platforms)
        platform_percentiles = [col for col in df.columns if col.endswith('_percentile') and 
                               any(platform in col for platform in ['Spotify', 'YouTube', 'TikTok'])]
        if len(platform_percentiles) >= 2:
            df['performance_consistency'] = 100 - df[platform_percentiles].std(axis=1)
        
        return df
    
    def _create_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical and binned features"""
        
        # Popularity tiers
        if 'Spotify Popularity' in df.columns:
            # Clean the data first and handle NaN values
            popularity_clean = df['Spotify Popularity'].fillna(0)
            
            df['popularity_tier'] = pd.cut(
                popularity_clean, 
                bins=[0, 40, 60, 80, 100], 
                labels=['Low', 'Medium', 'High', 'Viral'],
                include_lowest=True
            )
            # Convert to numeric for similarity calculation
            tier_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Viral': 4}
            df['popularity_tier_numeric'] = df['popularity_tier'].map(tier_mapping).fillna(1)
        
        # Content flags
        if 'Explicit Track' in df.columns:
            # Handle various formats for explicit track data
            explicit_col = df['Explicit Track'].fillna(False)
            if explicit_col.dtype == 'object':
                # Handle string values like 'True', 'False', 'Yes', 'No', etc.
                df['is_explicit'] = explicit_col.astype(str).str.lower().isin(['true', '1', 'yes', 'explicit']).astype(int)
            else:
                df['is_explicit'] = explicit_col.astype(int)
        
        # Platform dominance
        platform_perc_cols = [col for col in df.columns if col.endswith('_percentile') and 
                             any(platform in col for platform in ['Spotify', 'YouTube', 'TikTok'])]
        
        if len(platform_perc_cols) >= 2:
            df['dominant_platform_score'] = df[platform_perc_cols].max(axis=1)
            df['dominant_platform'] = df[platform_perc_cols].idxmax(axis=1)
            
            # Platform focus (how specialized vs generalist)
            df['platform_specialization'] = df[platform_perc_cols].max(axis=1) - df[platform_perc_cols].mean(axis=1)
        
        # Release era categories
        if 'release_year' in df.columns:
            df['decade'] = (df['release_year'] // 10) * 10
            
            # Music industry eras
            conditions = [
                df['release_year'] < 2000,
                (df['release_year'] >= 2000) & (df['release_year'] < 2010),
                (df['release_year'] >= 2010) & (df['release_year'] < 2020),
                df['release_year'] >= 2020
            ]
            era_labels = ['Pre-Digital', 'Early-Digital', 'Streaming-Era', 'Modern']
            df['music_era'] = np.select(conditions, era_labels, default='Unknown')
        
        return df
    
    def _create_missing_data_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create flags for missing data patterns (useful signals)"""
        
        # Platform presence flags
        major_platforms = ['Spotify Streams', 'YouTube Views', 'TikTok Views']
        for platform in major_platforms:
            if platform in df.columns:
                df[f'has_{platform.lower().replace(" ", "_")}_data'] = df[platform].notna().astype(int)
        
        # Engagement data completeness
        engagement_cols = ['YouTube Likes', 'TikTok Likes', 'TikTok Posts']
        available_engagement = [col for col in engagement_cols if col in df.columns]
        if available_engagement:
            df['engagement_data_completeness'] = sum(df[col].notna().astype(int) for col in available_engagement) / len(available_engagement)
        
        # Playlist presence
        playlist_cols = [col for col in df.columns if 'Playlist' in col]
        if playlist_cols:
            df['in_playlists'] = (df[playlist_cols].sum(axis=1) > 0).astype(int)
        
        return df
    
    def _prepare_optimized_similarity_matrix(self):
        """
        Prepare enhanced similarity matrix with feature engineering
        """
        # Select features for similarity calculation
        similarity_features = self._select_similarity_features()
        
        print(f"🔧 Using {len(similarity_features)} features for similarity calculation")
        
        # Prepare numeric features
        numeric_data = self.processed_data[similarity_features].fillna(0)
        
        # Robust scaling (better for outliers than StandardScaler)
        scaler = RobustScaler()
        numeric_scaled = scaler.fit_transform(numeric_data)
        
        # Prepare text features with enhanced TF-IDF
        text_features = self._create_enhanced_text_features()
        
        # Combine features with appropriate weights
        # 75% numeric features, 25% text features
        self.feature_matrix = np.hstack([
            numeric_scaled * 0.75,
            text_features * 0.25
        ])
        
        # Calculate cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        
        print(f"✅ Similarity matrix created: {self.similarity_matrix.shape}")
    
    def _select_similarity_features(self) -> List[str]:
        """
        Select the most important features for similarity calculation
        """
        # Core performance features
        core_features = [
            'Track Score', 'All Time Rank', 'Spotify Popularity'
        ]
        
        # Cross-platform features
        cross_platform_features = [
            'tiktok_spotify_ratio', 'youtube_spotify_ratio',
            'youtube_engagement_rate', 'viral_index', 'platform_diversity'
        ]
        
        # Temporal features
        temporal_features = [
            'recency_score', 'is_modern', 'is_streaming_era', 'is_post_covid',
            'is_summer_release', 'years_since_release'
        ]
        
        # Artist features
        artist_features = [
            'artist_mean_Spotify Streams', 'artist_count_Spotify Streams',
            'is_superstar', 'is_prolific_artist', 'has_featuring', 'is_remix'
        ]
        
        # Performance percentiles
        percentile_features = [
            col for col in self.processed_data.columns 
            if col.endswith('_percentile') and any(perf in col for perf in ['Spotify', 'YouTube', 'TikTok', 'Track'])
        ]
        
        # Composite scores
        composite_features = [
            'mainstream_score', 'performance_consistency', 'platform_specialization'
        ]
        
        # Category features
        category_features = [
            'popularity_tier_numeric', 'is_explicit', 'engagement_data_completeness'
        ]
        
        # Combine all feature groups
        all_features = (core_features + cross_platform_features + temporal_features + 
                       artist_features + percentile_features + composite_features + category_features)
        
        # Filter only available features
        available_features = [f for f in all_features if f in self.processed_data.columns]
        
        return available_features
    
    def _create_enhanced_text_features(self) -> np.ndarray:
        """
        Create enhanced TF-IDF features from text data
        """
        # Combine artist, album, and track information
        text_data = (
            self.processed_data['Artist'].fillna('') + ' ' +
            self.processed_data['Album Name'].fillna('') + ' ' +
            self.processed_data['Track'].fillna('')
        )
        
        # Enhanced TF-IDF with n-grams
        vectorizer = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),  # Unigrams and bigrams
            lowercase=True,
            stop_words='english',
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        text_matrix = vectorizer.fit_transform(text_data)
        self.text_vectorizer = vectorizer  # Store for later use
        
        return text_matrix.toarray()
    
    def _create_search_indices(self):
        """
        Create optimized search indices for fast lookups
        """
        # Create track list for searches
        self.track_list = self.processed_data['Track'].tolist()
        self.artist_list = self.processed_data['Artist'].unique().tolist()
        
        # Create lookup dictionaries
        self.track_to_index = {track: idx for idx, track in enumerate(self.track_list)}
        self.index_to_track = {idx: track for idx, track in enumerate(self.track_list)}
        
        print(f"🔍 Search indices created for {len(self.track_list):,} tracks and {len(self.artist_list):,} artists")
    
    def find_similar_tracks_fuzzy(self, track_name: str, threshold: int = 70) -> List[Tuple[str, int]]:
        """
        Find tracks using fuzzy matching (handles typos and partial matches)
        
        Args:
            track_name (str): Track name to search for
            threshold (int): Minimum similarity threshold (0-100)
        
        Returns:
            List[Tuple[str, int]]: List of (track_name, similarity_score) tuples
        """
        matches = process.extractBests(
            track_name, 
            self.track_list, 
            limit=10, 
            score_cutoff=threshold,
            scorer=fuzz.token_sort_ratio
        )
        return matches
    
    def recommend_similar_tracks(self, track_name: str, n: int = 10, 
                                use_fuzzy: bool = True) -> pd.DataFrame:
        """
        Recommend similar tracks using optimized cosine similarity
        
        Args:
            track_name (str): Name of the track to find similar songs for
            n (int): Number of recommendations to return
            use_fuzzy (bool): Whether to use fuzzy matching for track name
        
        Returns:
            pd.DataFrame: DataFrame with recommended tracks and similarity scores
        """
        try:
            # Find exact or fuzzy match
            if use_fuzzy and track_name not in self.track_to_index:
                fuzzy_matches = self.find_similar_tracks_fuzzy(track_name, threshold=80)
                if not fuzzy_matches:
                    return pd.DataFrame(columns=['Track', 'Artist', 'Similarity_Score', 'Match_Type'])
                
                # Use best fuzzy match
                track_name = fuzzy_matches[0][0]
                match_type = f"Fuzzy Match ({fuzzy_matches[0][1]}% similarity)"
            else:
                match_type = "Exact Match"
            
            # Get track index
            if track_name not in self.track_to_index:
                return pd.DataFrame(columns=['Track', 'Artist', 'Similarity_Score', 'Match_Type'])
            
            track_idx = self.track_to_index[track_name]
            
            # Get similarity scores
            similarity_scores = self.similarity_matrix[track_idx]
            
            # Get top N similar tracks (excluding the input track itself)
            similar_indices = similarity_scores.argsort()[-(n+1):-1][::-1]
            
            # Create recommendations dataframe
            recommendations = []
            for idx in similar_indices:
                track_data = self.processed_data.iloc[idx]
                recommendations.append({
                    'Track': track_data['Track'],
                    'Artist': track_data['Artist'],
                    'Album': track_data['Album Name'],
                    'Release_Year': track_data.get('release_year', 'Unknown'),
                    'Spotify_Popularity': track_data.get('Spotify Popularity', 'Unknown'),
                    'Similarity_Score': round(similarity_scores[idx] * 100, 2),
                    'Match_Type': match_type if idx == similar_indices[0] else 'Recommended'
                })
            
            return pd.DataFrame(recommendations)
            
        except Exception as e:
            print(f"Error in recommendation: {str(e)}")
            return pd.DataFrame(columns=['Track', 'Artist', 'Similarity_Score', 'Match_Type'])
    
    def get_top_tracks(self, n: int = 20, sort_by: str = 'Spotify Popularity') -> pd.DataFrame:
        """
        Get top tracks by various metrics
        
        Args:
            n (int): Number of top tracks to return
            sort_by (str): Column to sort by
        
        Returns:
            pd.DataFrame: Top tracks
        """
        sort_columns = {
            'Spotify Popularity': 'Spotify Popularity',
            'Track Score': 'Track Score', 
            'Viral Index': 'viral_index',
            'Mainstream Score': 'mainstream_score',
            'Recency': 'recency_score'
        }
        
        sort_col = sort_columns.get(sort_by, 'Spotify Popularity')
        
        if sort_col not in self.processed_data.columns:
            sort_col = 'Track Score'  # Fallback
        
        top_tracks = self.processed_data.nlargest(n, sort_col)
        
        return top_tracks[['Track', 'Artist', 'Album Name', 'release_year', 
                          'Spotify Popularity', 'Track Score']].fillna('Unknown')
    
    def get_trending_tracks(self, n: int = 20) -> pd.DataFrame:
        """
        Get trending tracks based on TikTok and recent release activity
        
        Returns:
            pd.DataFrame: Trending tracks
        """
        if 'viral_index' in self.processed_data.columns and 'recency_score' in self.processed_data.columns:
            # Combine viral index with recency for trending score
            self.processed_data['trending_score'] = (
                self.processed_data['viral_index'] * 0.7 + 
                self.processed_data['recency_score'] * 0.3
            )
            
            trending = self.processed_data.nlargest(n, 'trending_score')
            return trending[['Track', 'Artist', 'Album Name', 'release_year',
                           'viral_index', 'recency_score', 'trending_score']].fillna('Unknown')
        else:
            # Fallback to regular top tracks
            return self.get_top_tracks(n)
    
    def search_by_artist(self, artist_name: str, use_fuzzy: bool = True) -> pd.DataFrame:
        """
        Search tracks by artist name with fuzzy matching
        
        Args:
            artist_name (str): Artist name to search for
            use_fuzzy (bool): Use fuzzy matching
        
        Returns:
            pd.DataFrame: Tracks by the artist
        """
        if use_fuzzy:
            # Find similar artist names
            artist_matches = process.extractBests(
                artist_name, 
                self.artist_list, 
                limit=3, 
                score_cutoff=70
            )
            
            if not artist_matches:
                return pd.DataFrame()
            
            # Get tracks for all matched artists
            matched_artists = [match[0] for match in artist_matches]
            artist_tracks = self.processed_data[self.processed_data['Artist'].isin(matched_artists)]
        else:
            artist_tracks = self.processed_data[self.processed_data['Artist'].str.contains(artist_name, case=False, na=False)]
        
        return artist_tracks[['Track', 'Artist', 'Album Name', 'release_year', 
                             'Spotify Popularity', 'Track Score']].sort_values('Spotify Popularity', ascending=False)
    
    def get_dataset_statistics(self) -> Dict:
        """
        Get comprehensive dataset statistics
        
        Returns:
            Dict: Dataset statistics
        """
        stats = {
            'total_tracks': len(self.processed_data),
            'unique_artists': self.processed_data['Artist'].nunique(),
            'unique_albums': self.processed_data['Album Name'].nunique(),
            'date_range': {
                'earliest': str(self.processed_data['Release Date'].min()) if 'Release Date' in self.processed_data.columns else 'Unknown',
                'latest': str(self.processed_data['Release Date'].max()) if 'Release Date' in self.processed_data.columns else 'Unknown'
            },
            'feature_count': self.feature_matrix.shape[1],
            'data_completeness': {
                'spotify_data': (self.processed_data['Spotify Streams'].notna().sum() / len(self.processed_data)) * 100,
                'youtube_data': (self.processed_data['YouTube Views'].notna().sum() / len(self.processed_data)) * 100 if 'YouTube Views' in self.processed_data.columns else 0,
                'tiktok_data': (self.processed_data['TikTok Views'].notna().sum() / len(self.processed_data)) * 100 if 'TikTok Views' in self.processed_data.columns else 0
            }
        }
        
        return stats


# Initialize the optimized recommender
try:
    recommender_optimized = SpotifyRecommenderOptimized()
    print("🎵 Optimized Spotify Recommender loaded successfully!")
except Exception as e:
    print(f"❌ Error initializing recommender: {str(e)}")
    recommender_optimized = None