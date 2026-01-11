# adoption_system_final.py
"""
Intelligent Dog Adoption System with Breed Analysis
Optimized version without external dependencies for encoding detection
"""

import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import json
import base64
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🐾 Intelligent Dog Adoption System",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ENHANCED DATA LOADER WITHOUT CHARDET
# ============================================================================

class EnhancedBreedDataLoader:
    """Enhanced breed data loader without external dependencies"""
    
    @staticmethod
    def try_load_csv(filepath, encodings, delimiters):
        """
        Tries to load CSV with different encodings and delimiters.
        
        Args:
            filepath (str): Path to the file
            encodings (list): List of encodings to try
            delimiters (list): List of delimiters to try
            
        Returns:
            pd.DataFrame: Loaded DataFrame or None
        """
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(filepath, sep=delimiter, encoding=encoding, on_bad_lines='skip')
                    # Check if we got reasonable data
                    if len(df) > 0 and len(df.columns) > 1:
                        st.info(f"✅ Successfully loaded with encoding '{encoding}' and delimiter '{delimiter}'")
                        return df
                except UnicodeDecodeError:
                    continue  # Try next encoding
                except pd.errors.ParserError:
                    continue  # Try next delimiter
                except Exception as e:
                    continue  # Try next combination
        return None
    
    @staticmethod
    def load_breed_data_with_fallback(filepath='breeds_final_dataset1.csv'):
        """
        Loads breed data with multiple fallback strategies for encoding.
        
        Args:
            filepath (str): Path to the breed data CSV file
            
        Returns:
            pd.DataFrame: Loaded breed data or empty DataFrame
        """
        # Common encodings to try (in order of likelihood)
        encodings_to_try = [
            'utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 
            'utf-16', 'ascii', 'utf-8-sig', 'windows-1252'
        ]
        
        # Common delimiters to try
        delimiters_to_try = [';', ',', '\t', '|', ' ']
        
        st.info("🔍 Attempting to load breed data...")
        
        # Try standard approach first
        result = EnhancedBreedDataLoader.try_load_csv(filepath, encodings_to_try, delimiters_to_try)
        
        if result is not None:
            return result
        
        # If standard approach fails, try more aggressive methods
        st.warning("Standard loading failed. Trying alternative methods...")
        
        # Method 1: Try reading as binary and decoding
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
                
                # Try different encodings with error handling
                for encoding in encodings_to_try:
                    try:
                        decoded = content.decode(encoding, errors='replace')
                        # Write to temporary string and load
                        from io import StringIO
                        temp_data = StringIO(decoded)
                        
                        for delimiter in delimiters_to_try:
                            try:
                                df = pd.read_csv(temp_data, sep=delimiter, on_bad_lines='skip')
                                if len(df) > 0 and len(df.columns) > 1:
                                    st.success(f"✅ Loaded via binary decode with {encoding}")
                                    return df
                            except:
                                temp_data.seek(0)  # Reset buffer
                                continue
                    except UnicodeDecodeError:
                        continue
        except Exception as e:
            st.error(f"Binary read failed: {str(e)}")
        
        # Method 2: Try with engine='python' for more flexible parsing
        try:
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(filepath, sep=None, engine='python', encoding=encoding, on_bad_lines='skip')
                    if len(df) > 0 and len(df.columns) > 1:
                        st.success(f"✅ Loaded with Python engine and {encoding}")
                        return df
                except:
                    continue
        except Exception as e:
            st.error(f"Python engine method failed: {str(e)}")
        
        # If all methods fail
        st.error("❌ All loading attempts failed.")
        return pd.DataFrame()

# ============================================================================
# UPDATED BREED DATA ANALYZER
# ============================================================================

class BreedDataAnalyzer:
    """Analyzes and clusters dog breed data."""
    
    @staticmethod
    def load_breed_data(filepath='breeds_final_dataset1.csv'):
        """
        Loads and processes breed data from CSV file.
        
        Args:
            filepath (str): Path to the breed data CSV file
            
        Returns:
            pd.DataFrame: Processed breed data or empty DataFrame on error
        """
        try:
            # Check if file exists
            import os
            if not os.path.exists(filepath):
                st.error(f"❌ File not found: {filepath}")
                st.info(f"💡 Current directory: {os.getcwd()}")
                st.info(f"💡 Looking for: {os.path.abspath(filepath)}")
                return pd.DataFrame()
            
            # Use enhanced loader
            breeds_df = EnhancedBreedDataLoader.load_breed_data_with_fallback(filepath)
            
            if breeds_df.empty or len(breeds_df) == 0:
                st.error("❌ Failed to load breed data or file is empty.")
                return pd.DataFrame()
            
            st.success(f"✅ Breed data loaded: {len(breeds_df)} breeds found")
            
            # Display basic info about the loaded data
            with st.expander("📊 Data Preview", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**First 5 rows:**")
                    st.dataframe(breeds_df.head())
                with col2:
                    st.write("**Data Info:**")
                    st.write(f"Shape: {breeds_df.shape}")
                    st.write(f"Columns: {list(breeds_df.columns)}")
            
            # Process the loaded data
            return BreedDataAnalyzer.process_breed_data(breeds_df)
            
        except Exception as e:
            st.error(f"❌ Error processing breed data: {str(e)}")
            # Try to provide helpful debugging info
            st.info("💡 **Troubleshooting tips:**")
            st.info("1. Check if the file exists at the specified path")
            st.info("2. Ensure the file is a valid CSV")
            st.info("3. Common encodings to try: utf-8, latin-1, cp1252")
            st.info("4. Common delimiters: semicolon (;), comma (,), tab")
            return pd.DataFrame()
    
    @staticmethod
    def process_breed_data(breeds_df):
        """
        Processes and enriches breed data for analysis.
        
        Args:
            breeds_df (pd.DataFrame): Raw breed data
            
        Returns:
            pd.DataFrame: Processed and enriched breed data
        """
        processed_df = breeds_df.copy()
        
        # Clean and standardize column names
        original_columns = list(processed_df.columns)
        processed_df.columns = [str(col).strip().replace(' ', '_').lower() for col in processed_df.columns]
        
        # Show column mapping info
        st.info(f"📝 Columns standardized from {original_columns} to {list(processed_df.columns)}")
        
        # Check for required columns and provide helpful mapping
        st.write("### 🔍 Column Identification")
        
        # Try to identify important columns by common names
        column_mapping = {}
        for col in processed_df.columns:
            col_lower = col.lower()
            if 'breed' in col_lower:
                column_mapping['breed'] = col
            elif 'size' in col_lower and not 'home' in col_lower:
                column_mapping['size'] = col
            elif 'exercise' in col_lower:
                column_mapping['exercise'] = col
            elif 'life' in col_lower or 'time' in col_lower:
                column_mapping['life_time'] = col
            elif 'groom' in col_lower:
                column_mapping['grooming'] = col
            elif 'vulner' in col_lower:
                column_mapping['vulnerable_breed'] = col
        
        if column_mapping:
            st.write("**Identified columns:**")
            for key, value in column_mapping.items():
                st.write(f"  - {key}: `{value}`")
        
        # Encode categorical features to numerical values
        processed_df = BreedDataAnalyzer.encode_categorical_features(processed_df, column_mapping)
        
        # Create derived features for better analysis
        processed_df = BreedDataAnalyzer.create_derived_features(processed_df)
        
        # Normalize features for clustering
        processed_df = BreedDataAnalyzer.normalize_features(processed_df)
        
        return processed_df
    
    @staticmethod
    def encode_categorical_features(df, column_mapping=None):
        """Encodes categorical features to numerical values for analysis."""
        if column_mapping is None:
            column_mapping = {}
        
        # Clean and standardize categorical values
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
        
        # Map size categories to numerical values
        size_col = column_mapping.get('size', 'size')
        if size_col in df.columns:
            size_mapping = {
                'small': 1, 'small-medium': 2, 'medium': 3, 
                'medium-large': 4, 'large': 5, 'extra large': 6,
                'extra_large': 6, 'extralarge': 6
            }
            
            # Try to map common size variations
            df['size_numeric'] = df[size_col].str.lower().map(size_mapping)
            
            # If mapping failed, create manual mapping
            if df['size_numeric'].isna().any():
                unique_sizes = df[size_col].unique()
                st.info(f"📏 Found size categories: {unique_sizes}")
                # Create automatic mapping for unique values
                auto_mapping = {size: i+1 for i, size in enumerate(sorted(set(unique_sizes)))}
                df['size_numeric'] = df[size_col].map(auto_mapping).fillna(3)
        
        # Map exercise requirements
        exercise_col = column_mapping.get('exercise', 'exercise')
        if exercise_col in df.columns:
            exercise_mapping = {
                'up to 30 minutes per day': 1,
                'up to 1 hour per day': 2,
                'more than 2 hours per day': 3,
                '30 minutes': 1,
                '1 hour': 2,
                '2 hours': 3
            }
            df['exercise_numeric'] = df[exercise_col].str.lower().map(exercise_mapping).fillna(2)
        
        # Convert vulnerable breed column
        vulner_col = column_mapping.get('vulnerable_breed', 'vulnerable_breed')
        if vulner_col in df.columns:
            df['vulnerable_numeric'] = df[vulner_col].str.lower().map({
                'yes': 1, 'y': 1, 'true': 1, '1': 1,
                'no': 0, 'n': 0, 'false': 0, '0': 0
            }).fillna(0)
        
        # Map grooming requirements if available
        groom_col = column_mapping.get('grooming', 'grooming')
        if groom_col in df.columns:
            grooming_mapping = {
                'once a week': 1,
                'more than once a week': 2,
                'every day': 3,
                'weekly': 1,
                'daily': 3
            }
            df['grooming_numeric'] = df[groom_col].str.lower().map(grooming_mapping).fillna(2)
        
        return df
    
    @staticmethod
    def create_derived_features(df):
        """Creates derived features for better breed analysis."""
        # Calculate care complexity if we have the components
        care_factors = []
        if 'size_numeric' in df.columns:
            care_factors.append('size_numeric')
        if 'grooming_numeric' in df.columns:
            care_factors.append('grooming_numeric')
        
        if len(care_factors) >= 2:
            weights = [0.5, 0.5] if len(care_factors) == 2 else [0.4, 0.3, 0.3]
            df['care_complexity'] = 0
            for i, factor in enumerate(care_factors):
                if i < len(weights):
                    df['care_complexity'] += df[factor] * weights[i]
        
        # Calculate activity level score
        if 'exercise_numeric' in df.columns and 'size_numeric' in df.columns:
            df['activity_level_score'] = (
                df['exercise_numeric'] * 0.6 + 
                df['size_numeric'] * 0.4
            )
        
        # Create apartment suitability score
        if 'size_numeric' in df.columns and 'exercise_numeric' in df.columns:
            # Smaller size and less exercise = better for apartments
            df['apartment_suitability'] = (
                (6 - df['size_numeric']) * 0.7 +  # Invert size (smaller is better)
                (4 - df['exercise_numeric']) * 0.3  # Less exercise is better
            )
        
        return df
    
    @staticmethod
    def normalize_features(df):
        """Normalizes features for clustering algorithms."""
        # Find all numeric columns (excluding any that might be IDs or categories)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out columns that shouldn't be normalized
        exclude_cols = ['cluster', 'pca1', 'pca2', 'dim1', 'dim2']  # These are results, not inputs
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if numeric_cols:
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(df[numeric_cols])
            
            for i, col in enumerate(numeric_cols):
                df[f'{col}_norm'] = normalized_data[:, i]
            
            st.info(f"📊 Normalized {len(numeric_cols)} numeric features for clustering")
        
        return df

# ============================================================================
# SIMPLIFIED STREAMLIT APP
# ============================================================================

def main():
    """Main function to run the Streamlit application."""
    st.title("🐾 Intelligent Dog Adoption System")
    st.markdown("---")
    
    # Initialize session state
    if 'breeds_data' not in st.session_state:
        st.session_state.breeds_data = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Home", "Breed Analysis", "Find Matches", "About"]
    )
    
    if app_mode == "Home":
        render_home_page()
    elif app_mode == "Breed Analysis":
        render_breed_analysis()
    elif app_mode == "Find Matches":
        render_find_matches()
    elif app_mode == "About":
        render_about_page()

def render_home_page():
    """Renders the home page of the application."""
    st.header("Welcome to the Intelligent Dog Adoption System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Find Your Perfect Canine Companion
        
        This intelligent system helps you:
        
        🔍 **Analyze dog breeds** based on characteristics and needs
        🎯 **Match with suitable dogs** based on your lifestyle
        📊 **Visualize breed clusters** using machine learning
        📋 **Make informed decisions** about dog adoption
        
        ### Quick Start Guide
        
        1. **Load Data**: Go to 'Breed Analysis' and load the breed data
        2. **Explore**: View statistics, distributions, and clusters
        3. **Find Matches**: Create your profile in 'Find Matches'
        4. **Get Recommendations**: Receive personalized breed suggestions
        
        ### Features
        
        - **Comprehensive Breed Database**: 200+ breeds with detailed characteristics
        - **Advanced Analytics**: Machine learning clustering and PCA visualization
        - **Interactive Interface**: Easy-to-use forms and visualizations
        - **Educational Resources**: Learn about breed requirements and care
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1552053831-71594a27632d?w=400", 
                caption="Find Your Perfect Match", use_column_width=True)
        
        st.info("""
        🐾 **Did You Know?**
        
        - There are over 340 recognized dog breeds worldwide
        - Different breeds have vastly different exercise needs
        - Small breeds often live longer than large breeds
        - Adoption saves lives!
        """)

def render_breed_analysis():
    """Renders the breed analysis section."""
    st.header("🔬 Dog Breed Analysis")
    
    # File upload option for flexibility
    st.subheader("📁 Load Breed Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Provide options for file loading
        option = st.radio(
            "Select data source:",
            ["Use default file", "Upload custom CSV file"],
            index=0
        )
    
    with col2:
        if option == "Use default file":
            filepath = "breeds_final_dataset1.csv"
            st.info(f"Using: {filepath}")
        else:
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with open("uploaded_breeds.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                filepath = "uploaded_breeds.csv"
                st.success("File uploaded successfully!")
            else:
                st.warning("Please upload a CSV file")
                return
    
    # Load data button
    if st.button("📥 Load Breed Data", type="primary", use_container_width=True):
        with st.spinner("Loading and processing breed data..."):
            st.session_state.breeds_data = BreedDataAnalyzer.load_breed_data(filepath)
            if st.session_state.breeds_data is not None and not st.session_state.breeds_data.empty:
                st.session_state.data_loaded = True
                st.balloons()  # Celebration animation
    
    # Display analysis if data is loaded
    if st.session_state.data_loaded and st.session_state.breeds_data is not None:
        breeds_data = st.session_state.breeds_data
        
        st.success(f"✅ Analysis ready! Loaded {len(breeds_data)} breeds with {len(breeds_data.columns)} features.")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "📈 Statistics", 
            "🎯 Clustering",
            "🔍 Search"
        ])
        
        with tab1:
            render_breed_overview(breeds_data)
        
        with tab2:
            render_breed_statistics(breeds_data)
        
        with tab3:
            render_clustering_analysis(breeds_data)
        
        with tab4:
            render_breed_search(breeds_data)
    elif st.session_state.breeds_data is not None and st.session_state.breeds_data.empty:
        st.error("Data was loaded but appears to be empty. Please check your file.")

def render_breed_overview(breeds_data):
    """Renders overview of breed data."""
    st.subheader("📊 Breed Data Overview")
    
    # Display first 20 rows
    st.write("### Sample Data (First 20 rows)")
    st.dataframe(breeds_data.head(20), use_container_width=True)
    
    # Summary statistics
    st.subheader("📈 Quick Statistics")
    
    # Create metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Breeds", len(breeds_data))
    
    with col2:
        # Find size column
        size_col = None
        for col in breeds_data.columns:
            if 'size' in col.lower() and not 'numeric' in col:
                size_col = col
                break
        
        if size_col and size_col in breeds_data.columns:
            unique_sizes = breeds_data[size_col].nunique()
            st.metric("Size Categories", unique_sizes)
        else:
            st.metric("Features", len(breeds_data.columns))
    
    with col3:
        # Find breed column
        breed_col = None
        for col in breeds_data.columns:
            if 'breed' in col.lower():
                breed_col = col
                break
        
        if breed_col and breed_col in breeds_data.columns:
            st.metric("Breed Column", breed_col)
        else:
            st.metric("Numeric Columns", len(breeds_data.select_dtypes(include=[np.number]).columns))
    
    with col4:
        if 'vulnerable_numeric' in breeds_data.columns:
            vulnerable = breeds_data['vulnerable_numeric'].sum()
            st.metric("Vulnerable Breeds", int(vulnerable))
        else:
            st.metric("Missing Values", breeds_data.isna().sum().sum())
    
    # Size distribution visualization
    size_col = None
    for col in breeds_data.columns:
        if 'size' in col.lower() and not 'numeric' in col and col != 'size_home':
            size_col = col
            break
    
    if size_col and size_col in breeds_data.columns:
        st.subheader("📏 Size Distribution")
        
        size_counts = breeds_data[size_col].value_counts().sort_index()
        
        # Create two visualizations side by side
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            bars = ax1.bar(size_counts.index, size_counts.values, color='skyblue', alpha=0.8)
            ax1.set_xlabel('Size Category')
            ax1.set_ylabel('Number of Breeds')
            ax1.set_title('Breed Distribution by Size')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig1)
        
        with col2:
            # Pie chart
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            wedges, texts, autotexts = ax2.pie(size_counts.values, labels=size_counts.index, 
                                              autopct='%1.1f%%', startangle=90,
                                              colors=plt.cm.Set3(np.linspace(0, 1, len(size_counts))))
            ax2.set_title('Size Distribution Percentage')
            st.pyplot(fig2)

def render_breed_statistics(breeds_data):
    """Renders statistical analysis of breeds."""
    st.subheader("📈 Detailed Statistical Analysis")
    
    # Select numeric columns for analysis
    numeric_cols = breeds_data.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        # Display descriptive statistics
        st.write("### Descriptive Statistics")
        st.dataframe(breeds_data[numeric_cols].describe().round(2), use_container_width=True)
        
        # Correlation matrix
        st.write("### Correlation Matrix")
        
        # Filter to reasonable number of columns for correlation matrix
        if len(numeric_cols) > 15:
            # Select most important numeric columns
            important_cols = [col for col in numeric_cols if 'norm' not in col and col not in ['cluster', 'pca1', 'pca2']]
            important_cols = important_cols[:15]  # Limit to 15 columns
            corr_data = breeds_data[important_cols]
        else:
            corr_data = breeds_data[numeric_cols]
        
        corr_matrix = corr_data.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True, cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Matrix of Breed Features')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
        
        # Most correlated features
        st.write("### Top Feature Correlations")
        
        # Create a list of correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
        top_correlations = corr_df.sort_values('Abs_Correlation', ascending=False).head(10)
        
        st.dataframe(top_correlations[['Feature 1', 'Feature 2', 'Correlation']], use_container_width=True)
        
    else:
        st.warning("No numeric columns found for statistical analysis.")

def render_clustering_analysis(breeds_data):
    """Renders clustering analysis."""
    st.subheader("🎯 Breed Clustering Analysis")
    
    # Check if we have normalized features
    norm_cols = [col for col in breeds_data.columns if col.endswith('_norm')]
    
    if not norm_cols:
        st.warning("No normalized features found for clustering. Try loading the data again.")
        
        # Show available columns
        st.write("**Available columns:**", list(breeds_data.columns))
        return
    
    # Clustering parameters
    st.write("### Clustering Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)
    
    with col2:
        algorithm = st.selectbox("Clustering Algorithm", 
                                ["K-Means", "Agglomerative", "DBSCAN"])
    
    with col3:
        visualization = st.selectbox("Visualization Method", 
                                    ["PCA (2D)", "PCA (3D)", "t-SNE (2D)"])
    
    if st.button("🔍 Run Clustering Analysis", type="primary", use_container_width=True):
        with st.spinner("Clustering in progress. This may take a moment..."):
            try:
                # Prepare data
                X = breeds_data[norm_cols].values
                
                # Apply clustering
                if algorithm == "K-Means":
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = clusterer.fit_predict(X)
                elif algorithm == "Agglomerative":
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                    clusters = clusterer.fit_predict(X)
                else:  # DBSCAN
                    clusterer = DBSCAN(eps=0.5, min_samples=5)
                    clusters = clusterer.fit_predict(X)
                
                breeds_data['cluster'] = clusters
                
                # Calculate clustering quality metrics
                unique_clusters = len(set(clusters))
                if unique_clusters > 1 and algorithm != "DBSCAN":
                    try:
                        silhouette = silhouette_score(X, clusters)
                        st.info(f"Silhouette Score: {silhouette:.3f}")
                    except:
                        pass
                
                # Reduce dimensionality for visualization
                if visualization == "PCA (2D)":
                    reducer = PCA(n_components=2, random_state=42)
                    reduced_data = reducer.fit_transform(X)
                    breeds_data['dim1'] = reduced_data[:, 0]
                    breeds_data['dim2'] = reduced_data[:, 1]
                    
                    # Calculate explained variance
                    variance = reducer.explained_variance_ratio_.sum() * 100
                    st.info(f"PCA explains {variance:.1f}% of variance")
                    
                    # Visualize clusters
                    fig = px.scatter(breeds_data, x='dim1', y='dim2', 
                                    color='cluster', hover_data=['breed', 'size', 'exercise'],
                                    title=f'Breed Clusters ({algorithm}) - PCA 2D',
                                    labels={'cluster': 'Cluster'},
                                    color_continuous_scale=px.colors.qualitative.Set3)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif visualization == "PCA (3D)":
                    reducer = PCA(n_components=3, random_state=42)
                    reduced_data = reducer.fit_transform(X)
                    breeds_data['pca1'] = reduced_data[:, 0]
                    breeds_data['pca2'] = reduced_data[:, 1]
                    breeds_data['pca3'] = reduced_data[:, 2]
                    
                    fig = px.scatter_3d(breeds_data, x='pca1', y='pca2', z='pca3',
                                       color='cluster', hover_data=['breed', 'size'],
                                       title=f'Breed Clusters ({algorithm}) - PCA 3D',
                                       labels={'cluster': 'Cluster'})
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:  # t-SNE
                    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                    reduced_data = reducer.fit_transform(X)
                    breeds_data['tsne1'] = reduced_data[:, 0]
                    breeds_data['tsne2'] = reduced_data[:, 1]
                    
                    fig = px.scatter(breeds_data, x='tsne1', y='tsne2', 
                                    color='cluster', hover_data=['breed', 'size', 'exercise'],
                                    title=f'Breed Clusters ({algorithm}) - t-SNE 2D',
                                    labels={'cluster': 'Cluster'},
                                    color_continuous_scale=px.colors.qualitative.Set2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster analysis
                st.subheader("📊 Cluster Analysis")
                
                for cluster_id in sorted(breeds_data['cluster'].unique()):
                    if cluster_id == -1 and algorithm == "DBSCAN":
                        cluster_name = "Noise/Outliers"
                    else:
                        cluster_name = f"Cluster {cluster_id}"
                    
                    cluster_breeds = breeds_data[breeds_data['cluster'] == cluster_id]
                    
                    with st.expander(f"{cluster_name} ({len(cluster_breeds)} breeds)"):
                        # Find breed column
                        breed_col = None
                        for col in cluster_breeds.columns:
                            if 'breed' in col.lower():
                                breed_col = col
                                break
                        
                        if breed_col:
                            sample_breeds = cluster_breeds[breed_col].head(5).tolist()
                            st.write(f"**Sample breeds:** {', '.join(sample_breeds)}")
                        
                        # Size distribution if available
                        size_col = None
                        for col in cluster_breeds.columns:
                            if 'size' in col.lower() and not 'numeric' in col and col != 'size_home':
                                size_col = col
                                break
                        
                        if size_col and size_col in cluster_breeds.columns:
                            size_dist = cluster_breeds[size_col].value_counts()
                            if not size_dist.empty:
                                st.write("**Size distribution:**")
                                for size, count in size_dist.items():
                                    st.write(f"  - {size}: {count} breeds")
                
            except Exception as e:
                st.error(f"Clustering failed: {str(e)}")
                st.info("Try reducing the number of clusters or using a different algorithm.")

def render_breed_search(breeds_data):
    """Renders breed search functionality."""
    st.subheader("🔍 Search and Filter Breeds")
    
    # Find breed column
    breed_col = None
    for col in breeds_data.columns:
        if 'breed' in col.lower():
            breed_col = col
            break
    
    if not breed_col:
        st.warning("No breed column found in the data.")
        return
    
    # Search and filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("Search breed by name", "")
    
    with col2:
        # Find size column
        size_col = None
        for col in breeds_data.columns:
            if 'size' in col.lower() and not 'numeric' in col and col != 'size_home':
                size_col = col
                break
        
        if size_col:
            unique_sizes = ["All"] + sorted(breeds_data[size_col].dropna().unique().tolist())
            size_filter = st.selectbox("Filter by size", unique_sizes)
        else:
            size_filter = "All"
    
    with col3:
        # Filter by vulnerable breeds
        if 'vulnerable_numeric' in breeds_data.columns:
            vulner_filter = st.selectbox("Vulnerable breeds", ["All", "Yes", "No"])
        else:
            vulner_filter = "All"
    
    # Filter data
    filtered_data = breeds_data.copy()
    
    if search_term:
        filtered_data = filtered_data[
            filtered_data[breed_col].str.contains(search_term, case=False, na=False)
        ]
    
    if size_filter != "All" and size_col:
        filtered_data = filtered_data[filtered_data[size_col] == size_filter]
    
    if vulner_filter != "All" and 'vulnerable_numeric' in filtered_data.columns:
        if vulner_filter == "Yes":
            filtered_data = filtered_data[filtered_data['vulnerable_numeric'] == 1]
        else:
            filtered_data = filtered_data[filtered_data['vulnerable_numeric'] == 0]
    
    # Display results
    if len(filtered_data) > 0:
        st.success(f"**Found {len(filtered_data)} breeds:**")
        
        # Display as expandable cards
        for idx, breed in filtered_data.iterrows():
            with st.expander(f"🐕 {breed.get(breed_col, 'Unknown')}", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # Basic info
                    if size_col and size_col in breed:
                        st.write(f"**Size:** {breed[size_col]}")
                    
                    exercise_col = None
                    for col in breed.index:
                        if 'exercise' in str(col).lower() and not 'numeric' in str(col):
                            exercise_col = col
                            break
                    
                    if exercise_col and exercise_col in breed:
                        st.write(f"**Exercise:** {breed[exercise_col]}")
                    
                    if 'grooming' in breed:
                        st.write(f"**Grooming:** {breed['grooming']}")
                
                with col_b:
                    # Additional info
                    if 'life_time' in breed:
                        st.write(f"**Life Expectancy:** {breed['life_time']}")
                    
                    if 'breed_category' in breed:
                        st.write(f"**Category:** {breed['breed_category']}")
                    
                    if 'vulnerable_numeric' in breed and breed['vulnerable_numeric'] == 1:
                        st.warning("⚠️ Vulnerable Breed - May have health issues")
                    
                    # Show cluster if available
                    if 'cluster' in breed and not pd.isna(breed['cluster']):
                        if breed['cluster'] == -1:
                            st.info("📊 Cluster: Noise/Outlier")
                        else:
                            st.info(f"📊 Cluster: {int(breed['cluster'])}")
                
                # Show health issues if available
                if 'breed_issues' in breed and pd.notna(breed['breed_issues']):
                    with st.expander("Health Considerations"):
                        st.write(breed['breed_issues'][:500] + "..." if len(str(breed['breed_issues'])) > 500 else breed['breed_issues'])
    else:
        st.info("No breeds match your search criteria. Try broadening your search.")

def render_find_matches():
    """Renders the dog matching section."""
    st.header("🎯 Find Your Perfect Match")
    
    if st.session_state.breeds_data is None or st.session_state.breeds_data.empty:
        st.warning("⚠️ Please load breed data first in the 'Breed Analysis' section.")
        return
    
    breeds_data = st.session_state.breeds_data
    
    st.subheader("Create Your Adoption Profile")
    
    with st.form("match_profile"):
        st.write("### 🏠 Your Lifestyle & Environment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lifestyle = st.select_slider(
                "Your Activity Level",
                options=["Low (sedentary)", "Moderate", "High (active)", "Very High (athletic)"],
                value="Moderate"
            )
            
            home_type = st.selectbox(
                "Home Type",
                ["Apartment", "House with Yard", "House without Yard", "Farm/Rural Area"],
                index=0
            )
        
        with col2:
            experience = st.selectbox(
                "Dog Ownership Experience",
                ["First-time owner", "Some experience", "Experienced", "Professional"],
                index=0
            )
            
            time_available = st.slider(
                "Daily time for dog (hours)",
                1, 4, 2
            )
        
        st.write("### 🐕 Your Preferences")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Find size column
            size_col = None
            for col in breeds_data.columns:
                if 'size' in col.lower() and not 'numeric' in col and col != 'size_home':
                    size_col = col
                    break
            
            if size_col:
                unique_sizes = breeds_data[size_col].dropna().unique().tolist()
                size_pref = st.multiselect(
                    "Preferred Size(s)",
                    unique_sizes,
                    default=unique_sizes[:1] if unique_sizes else []
                )
        
        with col4:
            grooming_tolerance = st.select_slider(
                "Grooming Tolerance",
                options=["Low (minimal grooming)", "Moderate", "High (regular grooming)"],
                value="Moderate"
            )
        
        submitted = st.form_submit_button("Find My Matches 🐾", use_container_width=True)
        
        if submitted:
            # Simple matching algorithm
            matches = []
            
            for idx, breed in breeds_data.iterrows():
                score = 0
                notes = []
                
                # Size preference matching
                if size_pref and size_col and size_col in breed:
                    if breed[size_col] in size_pref:
                        score += 30
                        notes.append(f"✅ Size preference matched")
                    else:
                        score += 10
                
                # Lifestyle matching based on exercise needs
                exercise_col = None
                for col in breed.index:
                    if 'exercise' in str(col).lower() and not 'numeric' in str(col):
                        exercise_col = col
                        break
                
                if exercise_col and exercise_col in breed:
                    breed_exercise = str(breed[exercise_col]).lower()
                    
                    if lifestyle == "Low (sedentary)" and "30 minutes" in breed_exercise:
                        score += 25
                        notes.append("✅ Activity level well-matched")
                    elif lifestyle == "Moderate" and "1 hour" in breed_exercise:
                        score += 25
                        notes.append("✅ Activity level well-matched")
                    elif lifestyle == "High (active)" and "2 hours" in breed_exercise:
                        score += 25
                        notes.append("✅ Activity level well-matched")
                    elif lifestyle == "Very High (athletic)" and "2 hours" in breed_exercise:
                        score += 20
                        notes.append("⚡ Good activity match")
                    else:
                        score += 10
                        notes.append("⚠️ Activity level may require adjustment")
                
                # Home type matching
                if home_type == "Apartment":
                    # Check if breed is suitable for apartments
                    if 'apartment_suitability' in breed:
                        if breed['apartment_suitability'] > 3:  # Higher score = better for apartments
                            score += 20
                            notes.append("✅ Suitable for apartment living")
                        else:
                            score += 5
                            notes.append("⚠️ May not be ideal for apartments")
                
                # Experience matching
                if 'vulnerable_numeric' in breed and breed['vulnerable_numeric'] == 1:
                    if experience in ["Experienced", "Professional"]:
                        score += 25
                        notes.append("✅ Experience adequate for specialized breed")
                    else:
                        score += 5
                        notes.append("⚠️ May require more experience")
                else:
                    score += 15
                    notes.append("✅ Suitable for your experience level")
                
                # Time availability matching
                if time_available >= 2:
                    score += 15
                    notes.append("✅ Adequate time for care")
                else:
                    score += 5
                    notes.append("⚠️ Consider time commitment")
                
                # Grooming tolerance
                if 'grooming_numeric' in breed:
                    if grooming_tolerance == "Low (minimal grooming)" and breed['grooming_numeric'] <= 1:
                        score += 15
                    elif grooming_tolerance == "Moderate" and breed['grooming_numeric'] <= 2:
                        score += 15
                    elif grooming_tolerance == "High (regular grooming)":
                        score += 15
                
                # Find breed name
                breed_name = "Unknown"
                for col in breed.index:
                    if 'breed' in str(col).lower():
                        breed_name = breed[col]
                        break
                
                matches.append({
                    'breed': breed_name,
                    'size': breed[size_col] if size_col and size_col in breed else "Unknown",
                    'exercise': breed[exercise_col] if exercise_col and exercise_col in breed else "Unknown",
                    'score': min(100, score),  # Cap at 100
                    'notes': notes[:3],  # Limit to 3 notes
                    'match_level': get_match_level(score)
                })
            
            # Sort and display top matches
            matches.sort(key=lambda x: x['score'], reverse=True)
            top_matches = matches[:12]  # Show top 12
            
            st.subheader("🎯 Your Top Breed Matches")
            
            # Display in a grid
            cols = st.columns(3)
            
            for i, match in enumerate(top_matches):
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"### {match['breed']}")
                        st.markdown(f"**Match Score:** {match['score']}/100")
                        st.markdown(f"**{match['match_level']}**")
                        
                        st.write(f"**Size:** {match['size']}")
                        st.write(f"**Exercise:** {match['exercise']}")
                        
                        # Show match notes
                        if match['notes']:
                            st.write("**Key Points:**")
                            for note in match['notes']:
                                st.write(f"- {note}")
                        
                        st.markdown("---")

def get_match_level(score):
    """Determines match level based on score."""
    if score >= 80:
        return "⭐ EXCELLENT MATCH"
    elif score >= 65:
        return "👍 GOOD MATCH"
    elif score >= 50:
        return "🤔 MODERATE MATCH"
    else:
        return "⚠️ LOW MATCH"

def render_about_page():
    """Renders the about page."""
    st.header("About This Application")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🐾 Intelligent Dog Adoption System
        
        **Version:** 2.0.2 (Optimized & Robust)
        
        ### Purpose
        
        This application is designed to help prospective dog owners make informed decisions 
        about dog adoption by providing:
        
        1. **Comprehensive breed analysis** - Understand different breeds and their requirements
        2. **Intelligent matching** - Find breeds that suit your lifestyle
        3. **Educational resources** - Learn about breed characteristics and care needs
        4. **Data-driven insights** - Use analytics to make better decisions
        
        ### How It Works
        
        1. **Data Loading** - Load breed data from CSV files with automatic encoding detection
        2. **Data Processing** - Clean, encode, and normalize data for analysis
        3. **Analysis** - View statistics, distributions, and correlations
        4. **Clustering** - Use machine learning to group similar breeds
        5. **Matching** - Get personalized breed recommendations based on your profile
        
        ### Technical Features
        
        - **Robust Data Loading**: Handles various CSV formats and encodings
        - **Machine Learning**: K-Means, Agglomerative, and DBSCAN clustering
        - **Dimensionality Reduction**: PCA and t-SNE for visualization
        - **Interactive Visualizations**: Plotly charts with hover information
        - **Responsive Design**: Works on desktop and mobile devices
        
        ### Data Requirements
        
        The system expects CSV files with breed information including:
        - Breed names
        - Size categories
        - Exercise requirements
        - Grooming needs
        - Life expectancy
        - Breed categories
        - Health information (optional)
        
        ### Common Issues & Solutions
        
        **1. Encoding Problems:**
        - Try saving your CSV with UTF-8 encoding
        - Check if file uses semicolon (;) or comma (,) separators
        
        **2. File Not Found:**
        - Ensure file is in the same directory as the script
        - Check file name spelling
        - Use absolute path if needed
        
        **3. Data Format Issues:**
        - Ensure CSV has proper headers
        - Check for missing values
        - Verify column names are consistent
        """)
    
    with col2:
        st.info("""
        ### Quick Tips
        
        🚀 **For Best Results:**
        
        1. Load data in 'Breed Analysis' first
        2. Explore the statistics to understand the data
        3. Try different clustering configurations
        4. Be honest in your profile for better matches
        
        🔧 **Troubleshooting:**
        
        - If data won't load, try a different encoding
        - If clustering fails, reduce cluster count
        - For slow performance, use smaller datasets
        
        📞 **Support:**
        
        For issues or questions, check the documentation or contact support.
        """)
        
        st.image("https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400", 
                caption="Happy Adoption!", use_container_width=True)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()