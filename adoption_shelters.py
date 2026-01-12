# adoption_system_final.py
"""
INTELLIGENT DOG ADOPTION SYSTEM WITH BREED ANALYSIS AND SHELTER DOG MATCHING

This comprehensive application provides two integrated systems for dog adoption:
1. Breed Analysis System - For understanding dog breed characteristics and finding suitable breeds
2. Shelter Adoption System - For matching adopters with specific dogs in shelters

DEVELOPED BY: [Your Name/Team Name]
VERSION: 3.0.0
TECHNOLOGY STACK: Python, Streamlit, Pandas, Scikit-learn, Plotly, Matplotlib, Seaborn
USE CASE: Capstone Project for Data Science/AI Program
"""

# ============================================================================
# IMPORT SECTION - ALL NECESSARY LIBRARIES AND DEPENDENCIES
# ============================================================================

import pandas as pd              # Primary data manipulation library for handling CSV files and dataframes
# WHY: Pandas provides efficient data structures (DataFrames) for tabular data manipulation and analysis
# USE: Loading CSV files, data cleaning, transformation, filtering, and statistical analysis

import streamlit as st           # Web application framework for creating interactive data apps
# WHY: Streamlit enables rapid development of data applications without frontend coding
# USE: Building the entire user interface - forms, charts, navigation, and interactive elements

from datetime import datetime    # Date and time manipulation for timestamps and scheduling
# WHY: Needed for recording submission dates, creating timestamps for exports, and time-based calculations
# USE: Timestamping adoption form submissions, naming export files with dates

import numpy as np               # Numerical computing library for mathematical operations and array manipulation
# WHY: Essential for numerical calculations, random number generation, and array operations
# USE: Random simulation of days in shelter, numerical transformations, mathematical operations

import plotly.express as px      # High-level interface for creating interactive visualizations
# WHY: Plotly creates publication-quality, interactive charts that work well with Streamlit
# USE: Creating bar charts, pie charts, scatter plots, and 3D visualizations

import plotly.graph_objects as go # Lower-level Plotly API for more customized visualizations
# WHY: Provides finer control over chart elements when Plotly Express is insufficient
# USE: Reserved for advanced custom visualizations if needed

from plotly.subplots import make_subplots # For creating multi-plot figures with Plotly
# WHY: Enables combining multiple charts in a single figure for comparative analysis
# USE: Creating dashboard-like visualizations with multiple coordinated charts

import re                        # Regular expressions library for text pattern matching and extraction
# WHY: Essential for parsing unstructured text data (age descriptions, keyword extraction)
# USE: Extracting numbers from age strings, finding keywords in dog descriptions

import json                      # JavaScript Object Notation library for data serialization
# WHY: Standard format for data interchange; useful for saving/loading complex data structures
# USE: Potential future use for saving adopter profiles or configuration settings

import base64                    # Encoding library for converting binary data to text format
# WHY: Required for creating downloadable links for CSV exports in HTML format
# USE: Encoding CSV data for download links in the Streamlit interface

from io import BytesIO           # In-memory byte stream handling for file operations
# WHY: Creates temporary in-memory file-like objects for data manipulation
# USE: Handling data exports without writing to disk

from sklearn.preprocessing import StandardScaler # Feature scaling tool from scikit-learn
# WHY: Essential for normalizing features before clustering algorithms
# USE: Scaling numerical features to have zero mean and unit variance for clustering

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering # Clustering algorithms
# WHY: Three different clustering approaches for breed segmentation analysis
# - KMeans: Centroid-based, good for spherical clusters, requires specifying number of clusters
# - DBSCAN: Density-based, finds arbitrary shaped clusters, identifies outliers
# - Agglomerative: Hierarchical bottom-up clustering, creates dendrogram
# USE: Grouping similar dog breeds based on multiple characteristics

from sklearn.decomposition import PCA # Principal Component Analysis for dimensionality reduction
# WHY: Reduces high-dimensional data to 2D/3D for visualization while preserving variance
# USE: Visualizing high-dimensional breed data in 2D or 3D scatter plots

from sklearn.manifold import TSNE # t-Distributed Stochastic Neighbor Embedding
# WHY: Nonlinear dimensionality reduction that preserves local structure better than PCA
# USE: Alternative visualization method for complex cluster structures

from sklearn.metrics import silhouette_score, davies_bouldin_score # Clustering quality metrics
# WHY: Quantitative measures to evaluate clustering performance
# - Silhouette Score: Measures how similar objects are to their own cluster vs other clusters (-1 to 1, higher is better)
# - Davies-Bouldin Score: Ratio of within-cluster to between-cluster distances (lower is better)
# USE: Evaluating which clustering configuration works best for the breed data

import seaborn as sns            # Statistical data visualization library based on matplotlib
# WHY: Provides high-level interface for attractive statistical graphics
# USE: Creating correlation heatmaps with better aesthetics than matplotlib

import matplotlib.pyplot as plt  # Fundamental plotting library for creating static visualizations
# WHY: Industry-standard plotting library with extensive customization options
# USE: Creating bar charts, pie charts, and histograms for breed analysis

import warnings                  # Warning control system for managing runtime warnings
# WHY: To suppress non-critical warnings that might confuse end users
# USE: Creating cleaner output by ignoring deprecation warnings or convergence warnings

# ============================================================================
# WARNING CONFIGURATION
# ============================================================================

warnings.filterwarnings('ignore')
# WHY: Improves user experience by suppressing non-critical warnings that might appear in the Streamlit interface
# RISK: Could hide important issues. In production, warnings should be logged but not shown to users.
# BEST PRACTICE: In development, keep warnings enabled; in deployment, suppress for cleaner UX.

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🐾 Intelligent Dog Adoption System",  # Browser tab title
    page_icon="🐕",                                   # Browser tab icon (dog emoji)
    layout="wide",                                    # Uses full width of the browser
    initial_sidebar_state="expanded"                  # Sidebar starts expanded (visible)
)
# WHY: Sets the fundamental appearance and behavior of the Streamlit application
# - Wide layout: Better utilization of screen real estate for data visualizations
# - Expanded sidebar: Navigation is immediately visible to users
# - Dog emoji: Branding and visual appeal

# ============================================================================
# ENHANCED DATA LOADER WITHOUT CHARDET
# ============================================================================

class EnhancedBreedDataLoader:
    """
    ENHANCED BREED DATA LOADER - ROBUST CSV LOADING WITHOUT EXTERNAL DEPENDENCIES
    
    PURPOSE: Solves common CSV loading problems (encoding issues, delimiter confusion)
    PROBLEM: CSV files can have different encodings (UTF-8, Latin-1, etc.) and delimiters (comma, semicolon, tab)
    SOLUTION: Tries multiple combinations automatically to find the correct one
    ALTERNATIVE: Normally would use `chardet` library, but this avoids external dependencies
    """
    
    @staticmethod
    def try_load_csv(filepath, encodings, delimiters):
        """
        ATTEMPT CSV LOADING WITH DIFFERENT ENCODING-DELIMITER COMBINATIONS
        
        METHODOLOGY: Brute force approach trying all possible combinations
        DESIGN PATTERN: Factory method pattern for flexible loading strategies
        
        Args:
            filepath (str): Path to the CSV file to load
            encodings (list): List of character encodings to try (e.g., ['utf-8', 'latin-1'])
            delimiters (list): List of column separators to try (e.g., [',', ';', '\t'])
            
        Returns:
            pd.DataFrame: Successfully loaded DataFrame, or None if all attempts fail
        """
        # Nested loops: Outer loop tries encodings, inner loop tries delimiters
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    # Attempt to load CSV with current encoding and delimiter
                    df = pd.read_csv(filepath, sep=delimiter, encoding=encoding, on_bad_lines='skip')
                    # Validation: Ensure we got meaningful data (not empty, has multiple columns)
                    if len(df) > 0 and len(df.columns) > 1:
                        # User feedback: Inform which combination worked
                        st.info(f"✅ Successfully loaded with encoding '{encoding}' and delimiter '{delimiter}'")
                        return df  # Exit early with successful load
                except UnicodeDecodeError:
                    continue  # Wrong encoding - try next one
                except pd.errors.ParserError:
                    continue  # Wrong delimiter - try next one
                except Exception as e:
                    continue  # Any other error - try next combination
        return None  # All combinations failed
    
    @staticmethod
    def load_breed_data_with_fallback(filepath='breeds_final_dataset1.csv'):
        """
        COMPREHENSIVE DATA LOADING WITH MULTI-LEVEL FALLBACK STRATEGIES
        
        STRATEGY: Three-tier approach:
        1. Standard pandas loading with common combinations
        2. Binary file reading with manual decoding
        3. Python engine with flexible parsing
        
        Args:
            filepath (str): Path to the breed data CSV file
            
        Returns:
            pd.DataFrame: Loaded breed data or empty DataFrame if all methods fail
        """
        # TIER 1: Common encodings (ordered by likelihood based on real-world usage)
        encodings_to_try = [
            'utf-8',      # Most common modern encoding (international)
            'latin-1',    # Common for Western European languages
            'iso-8859-1', # ISO standard for Western Europe
            'cp1252',     # Windows Western European encoding
            'utf-16',     # Unicode with 16-bit encoding
            'ascii',      # Basic 128-character encoding
            'utf-8-sig',  # UTF-8 with BOM (Byte Order Mark)
            'windows-1252' # Another common Windows encoding
        ]
        
        # TIER 1: Common delimiters (CSV can mean Comma OR Character Separated Values)
        delimiters_to_try = [';', ',', '\t', '|', ' ']
        # ';' common in European CSVs, ',' standard, '\t' for TSV, '|' pipe-separated, ' ' space-separated
        
        st.info("🔍 Attempting to load breed data...")  # User feedback
        
        # ATTEMPT 1: Standard combination approach
        result = EnhancedBreedDataLoader.try_load_csv(filepath, encodings_to_try, delimiters_to_try)
        
        if result is not None:
            return result  # Success with standard method
        
        # TIER 2: Binary decoding approach (if standard method fails)
        st.warning("Standard loading failed. Trying alternative methods...")
        
        # METHOD 1: Read as binary and attempt manual decoding
        try:
            with open(filepath, 'rb') as f:  # 'rb' = read binary mode
                content = f.read()  # Read entire file as bytes
                
                # Try each encoding on the binary content
                for encoding in encodings_to_try:
                    try:
                        # Decode bytes to string with error replacement
                        decoded = content.decode(encoding, errors='replace')
                        # 'errors='replace'' replaces undecodable bytes with replacement character
                        
                        # Create in-memory string buffer for pandas
                        from io import StringIO
                        temp_data = StringIO(decoded)
                        
                        # Try different delimiters on the decoded string
                        for delimiter in delimiters_to_try:
                            try:
                                df = pd.read_csv(temp_data, sep=delimiter, on_bad_lines='skip')
                                if len(df) > 0 and len(df.columns) > 1:
                                    st.success(f"✅ Loaded via binary decode with {encoding}")
                                    return df
                            except:
                                temp_data.seek(0)  # Reset buffer position for next attempt
                                continue
                    except UnicodeDecodeError:
                        continue  # This encoding doesn't work
        except Exception as e:
            st.error(f"Binary read failed: {str(e)}")
        
        # TIER 3: Python engine with flexible parsing
        try:
            for encoding in encodings_to_try:
                try:
                    # sep=None lets pandas auto-detect delimiter
                    # engine='python' uses Python parser (slower but more flexible)
                    df = pd.read_csv(filepath, sep=None, engine='python', 
                                     encoding=encoding, on_bad_lines='skip')
                    if len(df) > 0 and len(df.columns) > 1:
                        st.success(f"✅ Loaded with Python engine and {encoding}")
                        return df
                except:
                    continue
        except Exception as e:
            st.error(f"Python engine method failed: {str(e)}")
        
        # ALL METHODS FAILED
        st.error("❌ All loading attempts failed.")
        return pd.DataFrame()  # Return empty DataFrame as failure indicator

# ============================================================================
# UPDATED BREED DATA ANALYZER
# ============================================================================

class BreedDataAnalyzer:
    """
    BREED DATA ANALYZER - COMPREHENSIVE PROCESSING AND ANALYSIS OF DOG BREED DATA
    
    RESPONSIBILITIES:
    1. Load and validate breed data from CSV
    2. Clean and standardize column names and values
    3. Encode categorical variables to numerical representations
    4. Create derived features for enhanced analysis
    5. Normalize features for machine learning algorithms
    6. Provide statistical summaries and visualizations
    
    DESIGN PATTERN: Pipeline pattern - Sequential data transformation steps
    """
    
    @staticmethod
    def load_breed_data(filepath='breeds_final_dataset1.csv'):
        """
        MAIN ENTRY POINT FOR BREED DATA LOADING AND PROCESSING
        
        WORKFLOW:
        1. File existence check
        2. Enhanced CSV loading with fallbacks
        3. Data preview and validation
        4. Full data processing pipeline
        
        Args:
            filepath (str): Path to the breed data CSV file
            
        Returns:
            pd.DataFrame: Fully processed and enriched breed data for analysis
        """
        try:
            # STEP 1: VERIFY FILE EXISTS
            import os  # Imported inside method to avoid global dependency
            if not os.path.exists(filepath):
                # Comprehensive error message with troubleshooting information
                st.error(f"❌ File not found: {filepath}")
                st.info(f"💡 Current directory: {os.getcwd()}")
                st.info(f"💡 Looking for: {os.path.abspath(filepath)}")
                return pd.DataFrame()  # Early return on failure
            
            # STEP 2: LOAD DATA USING ENHANCED LOADER
            breeds_df = EnhancedBreedDataLoader.load_breed_data_with_fallback(filepath)
            
            # STEP 3: VALIDATE LOADED DATA
            if breeds_df.empty or len(breeds_df) == 0:
                st.error("❌ Failed to load breed data or file is empty.")
                return pd.DataFrame()
            
            # STEP 4: USER FEEDBACK AND DATA PREVIEW
            st.success(f"✅ Breed data loaded: {len(breeds_df)} breeds found")
            
            # Interactive expander for data inspection
            with st.expander("📊 Data Preview", expanded=False):
                col1, col2 = st.columns(2)  # Two-column layout for better presentation
                with col1:
                    st.write("**First 5 rows:**")
                    st.dataframe(breeds_df.head())  # Show sample data
                with col2:
                    st.write("**Data Info:**")
                    st.write(f"Shape: {breeds_df.shape}")  # Rows × Columns
                    st.write(f"Columns: {list(breeds_df.columns)}")  # Column names
            
            # STEP 5: PROCESS DATA THROUGH FULL PIPELINE
            return BreedDataAnalyzer.process_breed_data(breeds_df)
            
        except Exception as e:
            # COMPREHENSIVE ERROR HANDLING WITH TROUBLESHOOTING TIPS
            st.error(f"❌ Error processing breed data: {str(e)}")
            # User-friendly troubleshooting guide
            st.info("💡 **Troubleshooting tips:**")
            st.info("1. Check if the file exists at the specified path")
            st.info("2. Ensure the file is a valid CSV")
            st.info("3. Common encodings to try: utf-8, latin-1, cp1252")
            st.info("4. Common delimiters: semicolon (;), comma (,), tab")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    @staticmethod
    def process_breed_data(breeds_df):
        """
        COMPREHENSIVE DATA PROCESSING PIPELINE FOR BREED DATA
        
        PROCESSING STEPS:
        1. Column name standardization
        2. Column identification and mapping
        3. Categorical feature encoding
        4. Derived feature creation
        5. Feature normalization for ML
        
        Args:
            breeds_df (pd.DataFrame): Raw breed data from CSV
            
        Returns:
            pd.DataFrame: Processed and enriched breed data ready for analysis
        """
        processed_df = breeds_df.copy()  # Create copy to avoid modifying original
        
        # STEP 1: COLUMN NAME STANDARDIZATION
        original_columns = list(processed_df.columns)  # Store original for comparison
        # Clean column names: lowercase, replace spaces with underscores, trim whitespace
        processed_df.columns = [str(col).strip().replace(' ', '_').lower() for col in processed_df.columns]
        
        # User feedback on column transformation
        st.info(f"📝 Columns standardized from {original_columns} to {list(processed_df.columns)}")
        
        # STEP 2: COLUMN IDENTIFICATION AND MAPPING
        st.write("### 🔍 Column Identification")
        
        # Heuristic column mapping based on common column name patterns
        column_mapping = {}
        for col in processed_df.columns:
            col_lower = col.lower()  # Case-insensitive matching
            
            # Pattern matching for common breed data columns
            if 'breed' in col_lower:
                column_mapping['breed'] = col  # Primary breed name column
            elif 'size' in col_lower and not 'home' in col_lower:
                column_mapping['size'] = col  # Size category (excluding 'size_home')
            elif 'exercise' in col_lower:
                column_mapping['exercise'] = col  # Exercise requirements
            elif 'life' in col_lower or 'time' in col_lower:
                column_mapping['life_time'] = col  # Life expectancy
            elif 'groom' in col_lower:
                column_mapping['grooming'] = col  # Grooming requirements
            elif 'vulner' in col_lower:
                column_mapping['vulnerable_breed'] = col  # Vulnerability status
        
        # Display identified columns to user
        if column_mapping:
            st.write("**Identified columns:**")
            for key, value in column_mapping.items():
                st.write(f"  - {key}: `{value}`")  # Show mapping
        
        # STEP 3: CATEGORICAL FEATURE ENCODING
        processed_df = BreedDataAnalyzer.encode_categorical_features(processed_df, column_mapping)
        
        # STEP 4: DERIVED FEATURE CREATION
        processed_df = BreedDataAnalyzer.create_derived_features(processed_df)
        
        # STEP 5: FEATURE NORMALIZATION FOR CLUSTERING
        processed_df = BreedDataAnalyzer.normalize_features(processed_df)
        
        return processed_df  # Return fully processed dataframe
    
    @staticmethod
    def encode_categorical_features(df, column_mapping=None):
        """
        TRANSFORM CATEGORICAL TEXT DATA TO NUMERICAL VALUES FOR ANALYSIS
        
        ENCODING STRATEGIES:
        1. Size categories: Map to ordinal numbers (1=small, 6=extra large)
        2. Exercise requirements: Map to ordinal scale (1=low, 3=high)
        3. Vulnerable breed: Binary encoding (1=yes, 0=no)
        4. Grooming needs: Ordinal scale (1=low, 3=high)
        
        Args:
            df (pd.DataFrame): Dataframe with categorical columns
            column_mapping (dict): Mapping of standard names to actual column names
            
        Returns:
            pd.DataFrame: Dataframe with added numerical columns
        """
        if column_mapping is None:
            column_mapping = {}  # Default to empty dict
        
        # PREPROCESSING: Clean all categorical (object) columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()  # Convert to string and trim whitespace
        
        # ENCODING 1: SIZE CATEGORIES TO NUMERICAL
        size_col = column_mapping.get('size', 'size')  # Get column name or default
        if size_col in df.columns:
            # Comprehensive size mapping dictionary
            size_mapping = {
                'small': 1, 
                'small-medium': 2, 
                'medium': 3, 
                'medium-large': 4, 
                'large': 5, 
                'extra large': 6,
                'extra_large': 6,  # Handle variations
                'extralarge': 6    # Handle variations
            }
            
            # Apply mapping with case normalization
            df['size_numeric'] = df[size_col].str.lower().map(size_mapping)
            
            # Fallback: Create automatic mapping if predefined mapping fails
            if df['size_numeric'].isna().any():  # Check for unmapped values
                unique_sizes = df[size_col].unique()
                st.info(f"📏 Found size categories: {unique_sizes}")
                # Create dynamic mapping: each unique size gets sequential number
                auto_mapping = {size: i+1 for i, size in enumerate(sorted(set(unique_sizes)))}
                df['size_numeric'] = df[size_col].map(auto_mapping).fillna(3)  # Default to medium
        
        # ENCODING 2: EXERCISE REQUIREMENTS
        exercise_col = column_mapping.get('exercise', 'exercise')
        if exercise_col in df.columns:
            exercise_mapping = {
                'up to 30 minutes per day': 1,  # Low exercise
                'up to 1 hour per day': 2,      # Moderate exercise
                'more than 2 hours per day': 3, # High exercise
                '30 minutes': 1,                # Abbreviated forms
                '1 hour': 2,
                '2 hours': 3
            }
            df['exercise_numeric'] = df[exercise_col].str.lower().map(exercise_mapping).fillna(2)  # Default moderate
        
        # ENCODING 3: VULNERABLE BREED STATUS (BINARY)
        vulner_col = column_mapping.get('vulnerable_breed', 'vulnerable_breed')
        if vulner_col in df.columns:
            # Comprehensive yes/no mapping with variations
            df['vulnerable_numeric'] = df[vulner_col].str.lower().map({
                'yes': 1, 'y': 1, 'true': 1, '1': 1,    # All yes variations
                'no': 0, 'n': 0, 'false': 0, '0': 0     # All no variations
            }).fillna(0)  # Default to not vulnerable
        
        # ENCODING 4: GROOMING REQUIREMENTS
        groom_col = column_mapping.get('grooming', 'grooming')
        if groom_col in df.columns:
            grooming_mapping = {
                'once a week': 1,          # Low grooming
                'more than once a week': 2, # Moderate grooming
                'every day': 3,             # High grooming
                'weekly': 1,                # Abbreviated forms
                'daily': 3
            }
            df['grooming_numeric'] = df[groom_col].str.lower().map(grooming_mapping).fillna(2)  # Default moderate
        
        return df
    
    @staticmethod
    def create_derived_features(df):
        """
        CREATE SYNTHETIC FEATURES FOR ENHANCED ANALYSIS
        
        DERIVED FEATURES:
        1. Care Complexity: Weighted combination of size and grooming needs
        2. Activity Level Score: Combination of exercise and size
        3. Apartment Suitability: Inverse scoring for apartment living
        
        Args:
            df (pd.DataFrame): Dataframe with basic numerical features
            
        Returns:
            pd.DataFrame: Dataframe with added derived features
        """
        # FEATURE 1: CARE COMPLEXITY SCORE
        care_factors = []
        if 'size_numeric' in df.columns:
            care_factors.append('size_numeric')  # Larger dogs generally need more care
        if 'grooming_numeric' in df.columns:
            care_factors.append('grooming_numeric')  # High grooming needs increase complexity
        
        if len(care_factors) >= 2:
            # Dynamic weight assignment based on available factors
            weights = [0.5, 0.5] if len(care_factors) == 2 else [0.4, 0.3, 0.3]
            df['care_complexity'] = 0  # Initialize column
            
            # Weighted sum calculation
            for i, factor in enumerate(care_factors):
                if i < len(weights):
                    df['care_complexity'] += df[factor] * weights[i]
        
        # FEATURE 2: ACTIVITY LEVEL SCORE
        if 'exercise_numeric' in df.columns and 'size_numeric' in df.columns:
            # Formula: 60% exercise + 40% size (larger dogs often more active)
            df['activity_level_score'] = (
                df['exercise_numeric'] * 0.6 + 
                df['size_numeric'] * 0.4
            )
        
        # FEATURE 3: APARTMENT SUITABILITY SCORE
        if 'size_numeric' in df.columns and 'exercise_numeric' in df.columns:
            # Reverse scoring: Smaller and less active = better for apartments
            # Size: 6 - size_numeric (inverts so smaller = higher score)
            # Exercise: 4 - exercise_numeric (inverts so less exercise = higher score)
            df['apartment_suitability'] = (
                (6 - df['size_numeric']) * 0.7 +  # 70% weight to size
                (4 - df['exercise_numeric']) * 0.3  # 30% weight to exercise
            )
        
        return df
    
    @staticmethod
    def normalize_features(df):
        """
        STANDARDIZE NUMERICAL FEATURES FOR MACHINE LEARNING ALGORITHMS
        
        WHY NORMALIZE:
        1. Clustering algorithms are distance-based and sensitive to scale
        2. Ensures all features contribute equally to distance calculations
        3. Improves convergence and performance of ML algorithms
        
        METHOD: StandardScaler (z-score normalization)
        Formula: (x - mean) / standard deviation
        
        Args:
            df (pd.DataFrame): Dataframe with numerical features
            
        Returns:
            pd.DataFrame: Dataframe with normalized features appended
        """
        # Identify all numerical columns for normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude columns that are results, not features (to avoid data leakage)
        exclude_cols = ['cluster', 'pca1', 'pca2', 'dim1', 'dim2']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if numeric_cols:  # Only normalize if we have numeric columns
            # Initialize StandardScaler (z-score normalizer)
            scaler = StandardScaler()
            # Fit to data and transform in one step
            normalized_data = scaler.fit_transform(df[numeric_cols])
            
            # Add normalized columns with '_norm' suffix
            for i, col in enumerate(numeric_cols):
                df[f'{col}_norm'] = normalized_data[:, i]
            
            # User feedback
            st.info(f"📊 Normalized {len(numeric_cols)} numeric features for clustering")
        
        return df

# ============================================================================
# SHELTER DOG ADOPTION SYSTEM - INTEGRATED COMPONENTS
# ============================================================================

class DogDataProcessor:
    """
    SHELTER DOG DATA PROCESSOR - TRANSFORMS RAW SHELTER DATA FOR MATCHING
    
    RESPONSIBILITIES:
    1. Load shelter dog data from CSV
    2. Extract features from unstructured descriptions
    3. Create derived categorical features
    4. Calculate adoption priority scores
    5. Prepare data for matching engine
    
    KEY INNOVATION: NLP-like feature extraction from free-text descriptions
    """
    
    @staticmethod
    def load_and_process_data(filepath='shelters_final_dataset.csv'):
        """
        MAIN ENTRY POINT FOR SHELTER DATA LOADING
        
        Args:
            filepath (str): Path to shelter data CSV
            
        Returns:
            pd.DataFrame: Processed shelter dog data ready for matching
        """
        try:
            # Load CSV with UTF-8 encoding (standard for multilingual data)
            df = pd.read_csv(filepath, encoding='utf-8')
            st.success(f"✅ Shelter data loaded: {len(df)} dogs found")
            
            # Process through full pipeline
            return DogDataProcessor.process_dataframe(df)
            
        except FileNotFoundError:
            # Comprehensive error handling with user guidance
            st.error("❌ File 'shelters_final_dataset.csv' not found!")
            st.info("Please place the CSV file in the same directory as this script.")
            return pd.DataFrame()  # Return empty dataframe
        except Exception as e:
            st.error(f"❌ Error loading shelter data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def process_dataframe(df):
        """
        COMPREHENSIVE SHELTER DATA PROCESSING PIPELINE
        
        TRANSFORMATIONS:
        1. Size estimation from breed names
        2. Age categorization from various formats
        3. Compatibility extraction from descriptions
        4. Energy level estimation
        5. Difficulty assessment
        6. Priority calculation
        7. Unique ID assignment
        
        Args:
            df (pd.DataFrame): Raw shelter data
            
        Returns:
            pd.DataFrame: Enriched shelter data with derived features
        """
        processed_df = df.copy()  # Preserve original data
        
        # FEATURE 1: SIZE CATEGORY ESTIMATION
        processed_df['size_category'] = processed_df['breed'].apply(DogDataProcessor.estimate_size)
        
        # FEATURE 2: AGE CATEGORIZATION
        processed_df['age_category'] = processed_df['age'].apply(DogDataProcessor.categorize_age)
        
        # FEATURE 3: CHILD COMPATIBILITY (from description)
        processed_df['child_compatibility'] = processed_df['description'].apply(DogDataProcessor.check_child_friendly)
        
        # FEATURE 4: DOG COMPATIBILITY (from description)
        processed_df['dog_compatibility'] = processed_df['description'].apply(DogDataProcessor.check_dog_friendly)
        
        # FEATURE 5: ENERGY LEVEL ESTIMATION
        processed_df['energy_level'] = processed_df['description'].apply(DogDataProcessor.estimate_energy)
        
        # FEATURE 6: DIFFICULTY LEVEL ASSESSMENT
        processed_df['difficulty_level'] = processed_df['description'].apply(DogDataProcessor.estimate_difficulty)
        
        # FEATURE 7: DAYS IN SHELTER (simulated for demonstration)
        # In real system, this would come from shelter database
        processed_df['days_in_shelter'] = np.random.randint(7, 365, len(processed_df))
        
        # FEATURE 8: ADOPTION PRIORITY CALCULATION
        processed_df['adoption_priority'] = DogDataProcessor.calculate_priority(processed_df)
        
        # FEATURE 9: MATCH SCORE PLACEHOLDER
        processed_df['match_score'] = 0  # Will be populated by matching engine
        
        # FEATURE 10: UNIQUE ID FOR EACH DOG
        processed_df['dog_id'] = range(1, len(processed_df) + 1)
        
        return processed_df
    
    @staticmethod
    def estimate_size(breed):
        """
        ESTIMATE DOG SIZE FROM BREED NAME USING KEYWORD MATCHING
        
        METHOD: Heuristic classification based on breed name keywords
        LIMITATION: Not 100% accurate but works for common breeds
        IMPROVEMENT: Could use breed database API for accuracy
        
        Args:
            breed (str): Breed name (may be None or NaN)
            
        Returns:
            str: Size category ('Small', 'Medium', or 'Large')
        """
        if pd.isna(breed):
            return 'Medium'  # Default when breed is unknown
        
        breed_lower = str(breed).lower()  # Case-insensitive matching
        
        # SMALL BREEDS DATABASE (common small breed keywords)
        small_breeds = ['chihuahua', 'pinscher', 'dachshund', 'zwergpinscher', 
                       'jack russell', 'terrier', 'pug', 'shih tzu', 'maltese',
                       'yorkshire', 'pomeranian', 'bichon', 'pekingese']
        
        # LARGE BREEDS DATABASE (common large breed keywords)
        large_breeds = ['kangal', 'rottweiler', 'shepherd', 'schäferhund', 'malinois',
                       'cane corso', 'mastiff', 'dogge', 'ridgeback', 'husky',
                       'akita', 'owtscharka', 'bullmastiff', 'great dane', 'leonberger',
                       'bernhardiner', 'dogge', 'pyrenäenberghund', 'maremmano']
        
        # Check if any small breed keyword appears in breed name
        if any(small_breed in breed_lower for small_breed in small_breeds):
            return 'Small'
        # Check if any large breed keyword appears in breed name
        elif any(large_breed in breed_lower for large_breed in large_breeds):
            return 'Large'
        else:
            return 'Medium'  # Default/fallback category
    
    @staticmethod
    def categorize_age(age_str):
        """
        CATEGORIZE AGE FROM VARIOUS TEXT FORMATS
        
        HANDLES MULTIPLE FORMATS:
        1. Numerical with units: "2 years", "6 months", "3 jahren"
        2. Text descriptions: "puppy", "senior", "adult", "young"
        3. Mixed languages: English, German, Portuguese
        
        Args:
            age_str (str): Age description in various formats
            
        Returns:
            str: Age category ('Puppy', 'Young', 'Adult', 'Senior')
        """
        if pd.isna(age_str):
            return 'Adult'  # Default when age unknown
        
        age_str = str(age_str).lower()  # Normalize case
        
        # METHOD 1: EXTRACT NUMERICAL VALUES WITH REGEX
        numbers = re.findall(r'\d+', age_str)  # Find all digit sequences
        
        if numbers:
            age_num = int(numbers[0])  # Take first number found
            
            # HANDLE MONTHS: Convert to years if necessary
            if 'month' in age_str or 'monat' in age_str:  # English and German
                if age_num <= 12:  # Up to 12 months = puppy
                    return 'Puppy'
                else:
                    age_num = age_num // 12  # Convert months to years
            
            # HANDLE YEARS: Categorize by age in years
            if 'year' in age_str or 'jahren' in age_str or 'years' in age_str:
                if age_num <= 1:
                    return 'Puppy'
                elif age_num <= 3:
                    return 'Young'
                elif age_num <= 8:
                    return 'Adult'
                else:
                    return 'Senior'
        
        # METHOD 2: KEYWORD MATCHING FOR TEXT DESCRIPTIONS
        if 'puppy' in age_str or 'welpe' in age_str or 'filhote' in age_str:
            return 'Puppy'
        elif 'senior' in age_str or 'old' in age_str or 'older' in age_str or 'idoso' in age_str:
            return 'Senior'
        elif 'adult' in age_str or 'adulto' in age_str:
            return 'Adult'
        elif 'young' in age_str or 'jovem' in age_str:
            return 'Young'
        
        return 'Adult'  # Default fallback
    
    @staticmethod
    def check_child_friendly(description):
        """
        EXTRACT CHILD-FRIENDLINESS FROM DOG DESCRIPTION
        
        METHOD: Keyword scanning of free-text descriptions
        ACCURACY: Good for explicit statements, limited for implicit information
        
        Args:
            description (str): Dog description text
            
        Returns:
            str: 'yes', 'no', or 'unknown'
        """
        if pd.isna(description):
            return 'unknown'  # No information
        
        desc = str(description).lower()
        
        # NEGATIVE KEYWORDS (indicates not child-friendly)
        negative = ['not suitable for children', 'not for children', 'keine kinder',
                   'nicht für kinder', 'should not live with children', 'no children',
                   'not child friendly', 'nicht geeignet für kinder', 'children over']
        
        # POSITIVE KEYWORDS (indicates child-friendly)
        positive = ['good with children', 'loves children', 'child friendly',
                   'family dog', 'family friendly', 'gute mit kindern',
                   'kindern gegenüber freundlich', 'children ok']
        
        # Check for negative indicators first (safety first)
        if any(keyword in desc for keyword in negative):
            return 'no'
        # Check for positive indicators
        elif any(keyword in desc for keyword in positive):
            return 'yes'
        
        return 'unknown'  # No clear indication
    
    @staticmethod
    def check_dog_friendly(description):
        """
        EXTRACT DOG-FRIENDLINESS (COMPATIBILITY WITH OTHER DOGS)
        
        Similar approach to child-friendliness check
        """
        if pd.isna(description):
            return 'unknown'
        
        desc = str(description).lower()
        
        negative = ['not dog friendly', 'does not like other dogs', 'should be only dog',
                   'single dog only', 'keine anderen hunde', 'doesn\'t get along with dogs',
                   'no other dogs', 'nicht mit anderen hunden']
        
        positive = ['dog friendly', 'gets along with dogs', 'good with other dogs',
                   'likes other dogs', 'verträglich mit hunden', 'other dogs ok']
        
        if any(keyword in desc for keyword in negative):
            return 'no'
        elif any(keyword in desc for keyword in positive):
            return 'yes'
        
        return 'unknown'
    
    @staticmethod
    def estimate_energy(description):
        """
        ESTIMATE ENERGY LEVEL FROM DESCRIPTION
        
        SCALE: 1 (low energy) to 3 (high energy)
        METHOD: Keyword matching with weighted scoring
        
        Args:
            description (str): Dog description
            
        Returns:
            int: Energy level (1, 2, or 3)
        """
        if pd.isna(description):
            return 2  # Default medium energy
        
        desc = str(description).lower()
        
        # HIGH ENERGY INDICATORS
        high_energy = ['high energy', 'very active', 'needs lots of exercise', 'energetic',
                      'lively', 'athletic', 'sporty', 'working dog', 'aktiv', 'viel bewegung']
        
        # LOW ENERGY INDICATORS
        low_energy = ['low energy', 'calm', 'quiet', 'senior', 'older', 'relaxed',
                     'laid back', 'couch potato', 'ruhig', 'entspannt']
        
        if any(keyword in desc for keyword in high_energy):
            return 3  # High energy
        elif any(keyword in desc for keyword in low_energy):
            return 1  # Low energy
        
        return 2  # Default medium energy
    
    @staticmethod
    def estimate_difficulty(description):
        """
        ESTIMATE DIFFICULTY LEVEL FOR CARE AND TRAINING
        
        SCALE: 1 (easy) to 5 (very difficult)
        METHOD: Count positive and negative indicators
        
        Args:
            description (str): Dog description
            
        Returns:
            int: Difficulty level (1-5)
        """
        if pd.isna(description):
            return 3  # Default medium difficulty
        
        desc = str(description).lower()
        
        difficulty = 3  # Start with neutral baseline
        
        # FACTORS THAT INCREASE DIFFICULTY
        hard_factors = ['biting incident', 'bite history', 'aggressive', 'behavioral issues',
                       'not for beginners', 'experience required', 'difficult', 'challenging',
                       'resource guarding', 'needs training', 'professional help', 'schwierig',
                       'erfahrung erforderlich', 'beißvorfall']
        
        # FACTORS THAT DECREASE DIFFICULTY
        easy_factors = ['easy', 'beginner friendly', 'family dog', 'calm', 'gentle',
                       'well trained', 'obedient', 'good listener', 'einfach', 'anfängerfreundlich']
        
        # Count occurrences of difficulty factors
        hard_count = sum(1 for factor in hard_factors if factor in desc)
        easy_count = sum(1 for factor in easy_factors if factor in desc)
        
        # Adjust difficulty based on factor counts
        if hard_count > 0:
            difficulty = min(5, 3 + hard_count)  # Increase difficulty, cap at 5
        elif easy_count > 0:
            difficulty = max(1, 3 - easy_count)  # Decrease difficulty, minimum 1
        
        return difficulty
    
    @staticmethod
    def calculate_priority(df):
        """
        CALCULATE ADOPTION PRIORITY SCORE FOR EACH DOG
        
        PRIORITY FACTORS:
        1. Age: Senior dogs get higher priority (harder to adopt)
        2. Difficulty: Difficult dogs get lower priority (need experienced owners)
        3. Time in shelter: Longer stays get higher priority
        
        SCALE: 1 (low priority) to 10 (high priority)
        
        Args:
            df (pd.DataFrame): Shelter dog data
            
        Returns:
            list: Priority scores for each dog
        """
        priorities = []
        
        for _, row in df.iterrows():
            priority = 5  # Base priority score
            
            # AGE FACTOR: Senior dogs have higher adoption priority
            if row.get('age_category') == 'Senior':
                priority += 2  # Significant boost for seniors
            elif row.get('age_category') == 'Puppy':
                priority += 1  # Small boost for puppies
            
            # DIFFICULTY FACTOR: Difficult dogs have lower priority
            if row.get('difficulty_level', 3) >= 4:  # High difficulty
                priority -= 1  # Penalty for difficult dogs
            
            # TIME FACTOR: Longer shelter stays increase priority
            if row.get('days_in_shelter', 0) > 180:  # More than 6 months
                priority += 1  # Boost for long-term residents
            
            # ENSURE SCORE STAYS WITHIN 1-10 RANGE
            priorities.append(max(1, min(10, priority)))
        
        return priorities

class AdoptionMatchingEngine:
    """
    INTELLIGENT MATCHING ENGINE FOR ADOPTION COMPATIBILITY
    
    ARCHITECTURE: Multi-criteria decision analysis with weighted scoring
    SCORING CATEGORIES (100 points total):
    1. Family Compatibility: 25 points (children, other pets)
    2. Space Compatibility: 20 points (housing type vs dog size)
    3. Experience Compatibility: 25 points (owner experience vs dog difficulty)
    4. Lifestyle Compatibility: 20 points (activity levels, time available)
    5. Preferences: 10 points (size, age preferences)
    
    DESIGN PATTERN: Strategy pattern for different compatibility checks
    """
    
    def __init__(self, dogs_data):
        """
        INITIALIZE MATCHING ENGINE WITH SHELTER DATA
        
        Args:
            dogs_data (pd.DataFrame): Processed shelter dog data
        """
        self.dogs_data = dogs_data
    
    def calculate_matches(self, adopter_profile):
        """
        CALCULATE COMPATIBILITY SCORES FOR ALL DOGS
        
        PROCESS:
        1. Calculate individual match score for each dog
        2. Sort dogs by compatibility score (highest first)
        3. Return top matches for user
        
        PERFORMANCE: O(n) where n = number of dogs
        OPTIMIZATION: Could be parallelized for large datasets
        
        Args:
            adopter_profile (dict): Adopter's information and preferences
            
        Returns:
            list: Top matching dogs with scores and details
        """
        matches = []
        
        # ITERATE THROUGH ALL DOGS IN SHELTER
        for _, dog in self.dogs_data.iterrows():
            # Calculate compatibility for this specific dog
            match_result = self.calculate_single_match(adopter_profile, dog)
            matches.append(match_result)
        
        # SORT BY MATCH SCORE (DESCENDING) - Highest compatibility first
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return matches[:15]  # Return top 15 matches (configurable)
    
    def calculate_single_match(self, adopter_profile, dog):
        """
        CALCULATE COMPREHENSIVE MATCH SCORE FOR SINGLE DOG
        
        METHOD: Weighted sum of five compatibility categories
        MAX SCORE: 100 points
        
        Args:
            adopter_profile (dict): Adopter information
            dog (pd.Series): Individual dog data
            
        Returns:
            dict: Complete match information including score, percentage, and notes
        """
        score = 0
        max_score = 100  # Maximum possible score
        notes = []  # Explanatory notes for user
        
        # CATEGORY 1: FAMILY COMPATIBILITY (25 points)
        family_score, family_notes = self._check_family_compatibility(adopter_profile, dog)
        score += family_score
        notes.extend(family_notes)
        
        # CATEGORY 2: SPACE COMPATIBILITY (20 points)
        space_score, space_notes = self._check_space_compatibility(adopter_profile, dog)
        score += space_score
        notes.extend(space_notes)
        
        # CATEGORY 3: EXPERIENCE COMPATIBILITY (25 points)
        experience_score, experience_notes = self._check_experience_compatibility(adopter_profile, dog)
        score += experience_score
        notes.extend(experience_notes)
        
        # CATEGORY 4: LIFESTYLE COMPATIBILITY (20 points)
        lifestyle_score, lifestyle_notes = self._check_lifestyle_compatibility(adopter_profile, dog)
        score += lifestyle_score
        notes.extend(lifestyle_notes)
        
        # CATEGORY 5: PREFERENCES (10 points)
        preference_score, preference_notes = self._check_preferences(adopter_profile, dog)
        score += preference_score
        notes.extend(preference_notes)
        
        # CONVERT TO PERCENTAGE FOR USER-FRIENDLY DISPLAY
        percentage = (score / max_score) * 100
        
        # COMPREHENSIVE RESULT DICTIONARY
        return {
            'dog_id': dog.get('dog_id'),  # Unique identifier
            'dog_name': dog.get('DOG_NAME'),  # Dog's name
            'dog_breed': dog.get('breed'),  # Breed information
            'dog_age': dog.get('age'),  # Age description
            'dog_age_category': dog.get('age_category'),  # Age category
            'dog_size': dog.get('size_category'),  # Size category
            'dog_gender': dog.get('gender'),  # Gender
            'shelter_name': dog.get('shelter_name'),  # Shelter name
            'shelter_address': dog.get('shelter_address'),  # Shelter address
            'match_score': score,  # Raw score (0-100)
            'match_percentage': percentage,  # Percentage (0-100%)
            'match_level': self._get_match_level(percentage),  # Textual rating
            'compatibility_notes': notes[:5],  # Top 5 notes (for display)
            'priority_level': dog.get('adoption_priority', 5),  # Adoption urgency
            'difficulty_level': dog.get('difficulty_level', 3),  # Care difficulty
            'energy_level': dog.get('energy_level', 2)  # Energy level
        }
    
    def _check_family_compatibility(self, adopter, dog):
        """
        ASSESS FAMILY COMPATIBILITY (CHILDREN AND OTHER PETS)
        
        LOGIC:
        - If adopter has children and dog is not child-friendly: DISQUALIFY (0 points)
        - If adopter has other dogs and dog is not dog-friendly: Penalty
        - Otherwise: Award points based on compatibility
        
        Args:
            adopter (dict): Adopter information
            dog (pd.Series): Dog information
            
        Returns:
            tuple: (score, notes) for family compatibility
        """
        score = 0
        notes = []
        
        # CHILDREN COMPATIBILITY CHECK
        if adopter.get('has_children', False):
            child_friendly = dog.get('child_compatibility', 'unknown')
            
            if child_friendly == 'no':
                notes.append("❌ Not recommended for children")
                return 0, notes  # DISQUALIFICATION - safety first
            elif child_friendly == 'yes':
                score += 12
                notes.append("✅ Good with children")
            else:  # Unknown compatibility
                score += 8
                notes.append("⚠️ Child compatibility unknown")
        else:  # No children in household
            score += 12
            notes.append("✅ No children in household")
        
        # OTHER DOGS COMPATIBILITY CHECK
        if adopter.get('has_other_dogs', False):
            dog_friendly = dog.get('dog_compatibility', 'unknown')
            
            if dog_friendly == 'no':
                notes.append("❌ Doesn't get along with other dogs")
                return score, notes  # Return current score (not disqualifying)
            elif dog_friendly == 'yes':
                score += 13
                notes.append("✅ Good with other dogs")
            else:  # Unknown
                score += 8
                notes.append("⚠️ Dog compatibility unknown")
        else:  # No other dogs
            score += 13
        
        return score, notes
    
    def _check_space_compatibility(self, adopter, dog):
        """
        ASSESS LIVING SPACE COMPATIBILITY
        
        METHOD: Lookup table scoring based on dog size and housing type
        PRINCIPLE: Right-size dog for available space
        
        Args:
            adopter (dict): Adopter housing information
            dog (pd.Series): Dog size information
            
        Returns:
            tuple: (score, notes) for space compatibility
        """
        score = 0
        notes = []
        
        housing_type = adopter.get('housing_type', 'Apartment')
        dog_size = dog.get('size_category', 'Medium')
        
        # COMPREHENSIVE COMPATIBILITY MATRIX
        compatibility_matrix = {
            # Format: (Dog Size, Housing Type): (Score, Note)
            ('Small', 'Apartment'): (20, "✅ Ideal space"),
            ('Small', 'House with Yard'): (18, "✅ Plenty of space"),
            ('Small', 'House without Yard'): (15, "⚡ Adequate space"),
            ('Medium', 'Apartment'): (12, "⚠️ Needs frequent walks"),
            ('Medium', 'House with Yard'): (20, "✅ Ideal space"),
            ('Medium', 'House without Yard'): (18, "✅ Adequate space"),
            ('Large', 'Apartment'): (5, "❌ Limited space - needs lots of activity"),
            ('Large', 'House with Yard'): (20, "✅ Ideal space"),
            ('Large', 'House without Yard'): (15, "⚠️ Needs additional space"),
            ('Large', 'Farm/Rural'): (20, "✅ Perfect space"),
        }
        
        # LOOKUP SCORE IN MATRIX
        key = (dog_size, housing_type)
        if key in compatibility_matrix:
            score, note = compatibility_matrix[key]
            notes.append(f"{note} ({dog_size} in {housing_type})")
        else:  # Fallback for unanticipated combinations
            score = 10
            notes.append("⚡ Reasonable space")
        
        return score, notes
    
    def _check_experience_compatibility(self, adopter, dog):
        """
        ASSESS OWNER EXPERIENCE VS DOG DIFFICULTY
        
        METHOD: Compare adopter experience level with dog difficulty
        PRINCIPLE: Match experience to challenge level
        
        Args:
            adopter (dict): Adopter experience information
            dog (pd.Series): Dog difficulty information
            
        Returns:
            tuple: (score, notes) for experience compatibility
        """
        score = 0
        notes = []
        
        # MAP ADOPTER EXPERIENCE TO NUMERICAL SCALE (1-5)
        experience_map = {
            'No experience': 1,
            'Little experience': 2,
            'Moderate experience': 3,
            'Lots of experience': 4,
            'Professional': 5
        }
        
        adopter_exp = experience_map.get(adopter.get('experience_level', 'No experience'), 1)
        dog_difficulty = dog.get('difficulty_level', 3)
        
        # CALCULATE EXPERIENCE GAP
        diff = adopter_exp - dog_difficulty
        
        # SCORE BASED ON EXPERIENCE GAP
        if diff >= 2:  # Significantly more experience than needed
            score = 25
            notes.append(f"✅ Excellent experience (level {adopter_exp})")
        elif diff >= 0:  # Adequate or slightly more experience
            score = 20
            notes.append(f"✅ Adequate experience")
        elif diff >= -1:  # Slightly less experience than ideal
            score = 15
            notes.append(f"⚠️ Slightly below ideal experience")
        elif diff >= -2:  # Insufficient experience
            score = 10
            notes.append(f"⚠️ Insufficient experience - training needed")
        else:  # Significantly insufficient experience
            score = 5
            notes.append(f"❌ Very low experience - not recommended")
        
        return score, notes
    
    def _check_lifestyle_compatibility(self, adopter, dog):
        """
        ASSESS LIFESTYLE COMPATIBILITY (ACTIVITY AND TIME)
        
        FACTORS:
        1. Activity level matching (adopter vs dog energy)
        2. Time available for dog care
        
        Args:
            adopter (dict): Adopter lifestyle information
            dog (pd.Series): Dog energy information
            
        Returns:
            tuple: (score, notes) for lifestyle compatibility
        """
        score = 0
        notes = []
        
        # ACTIVITY LEVEL COMPATIBILITY
        activity_map = {
            'Sedentary': 1,
            'Moderate': 2,
            'Active': 3,
            'Very active': 4
        }
        
        adopter_activity = activity_map.get(adopter.get('activity_level', 'Moderate'), 2)
        dog_energy = dog.get('energy_level', 2)
        
        # CALCULATE ENERGY LEVEL DIFFERENCE
        energy_diff = abs(adopter_activity - dog_energy)
        
        if energy_diff == 0:  # Perfect match
            score += 10
            notes.append("✅ Compatible energy levels")
        elif energy_diff == 1:  # Good match
            score += 7
            notes.append("⚡ Reasonably compatible energy levels")
        else:  # Poor match
            score += 3
            notes.append("⚠️ Significant difference in energy levels")
        
        # TIME AVAILABILITY COMPATIBILITY
        daily_time = adopter.get('daily_time_available', 3)
        
        if daily_time >= 4:  # Ample time
            score += 10
            notes.append("✅ Sufficient time available")
        elif daily_time >= 2:  # Reasonable time
            score += 7
            notes.append("⚡ Reasonable time available")
        else:  # Limited time
            score += 3
            notes.append("⚠️ Limited time available")
        
        return score, notes
    
    def _check_preferences(self, adopter, dog):
        """
        ASSESS PREFERENCE MATCHING (SIZE AND AGE)
        
        SCORING:
        - Exact match: Full points
        - No preference: Baseline points
        - Mismatch: Reduced points
        
        Args:
            adopter (dict): Adopter preferences
            dog (pd.Series): Dog characteristics
            
        Returns:
            tuple: (score, notes) for preference matching
        """
        score = 0
        notes = []
        
        # SIZE PREFERENCE CHECK
        preferred_size = adopter.get('preferred_size')
        dog_size = dog.get('size_category')
        
        if preferred_size and dog_size == preferred_size:
            score += 5
            notes.append(f"✅ Preferred size ({dog_size})")
        elif preferred_size:  # Has preference but doesn't match
            score += 2  # Light penalty
        
        # AGE PREFERENCE CHECK
        preferred_age = adopter.get('preferred_age')
        dog_age_cat = dog.get('age_category')
        
        if preferred_age and dog_age_cat == preferred_age:
            score += 5
            notes.append(f"✅ Preferred age range ({dog_age_cat})")
        elif preferred_age:  # Has preference but doesn't match
            score += 2  # Light penalty
        
        return score, notes
    
    def _get_match_level(self, percentage):
        """
        CONVERT NUMERICAL PERCENTAGE TO QUALITATIVE RATING
        
        Args:
            percentage (float): Compatibility percentage (0-100)
            
        Returns:
            str: Textual match level with emoji
        """
        if percentage >= 90:
            return "⭐ PERFECT MATCH"
        elif percentage >= 75:
            return "👍 HIGH COMPATIBILITY"
        elif percentage >= 60:
            return "🤔 MODERATE COMPATIBILITY"
        elif percentage >= 40:
            return "⚠️ LOW COMPATIBILITY"
        else:
            return "❌ NOT RECOMMENDED"

class AdoptionFormUI:
    """
    ADOPTION FORM USER INTERFACE COMPONENT
    
    RESPONSIBILITIES:
    1. Render comprehensive adoption form
    2. Collect adopter information and preferences
    3. Validate form inputs
    4. Create structured adopter profile
    
    DESIGN: Streamlit form with multiple sections and validation
    UX PRINCIPLES: Progressive disclosure, clear labeling, instant feedback
    """
    
    @staticmethod
    def render_form():
        """
        RENDER COMPLETE ADOPTION FORM WITH ALL SECTIONS
        
        FORM STRUCTURE:
        1. Personal Information (required)
        2. Housing Situation (required)
        3. Family Composition
        4. Dog Experience (required)
        5. Lifestyle (required)
        6. Dog Preferences (optional)
        
        Returns:
            dict or None: Structured adopter profile if form submitted and valid
        """
        st.title("📋 Adoption Form")
        st.markdown("Fill out the form below to find dogs compatible with your lifestyle.")
        
        # CREATE STREAMLIT FORM WITH PERSISTENT DATA
        with st.form("adoption_form", clear_on_submit=False):
            # SECTION 1: PERSONAL INFORMATION
            st.subheader("👤 Personal Information")
            col1, col2 = st.columns(2)  # Two-column layout for better organization
            
            with col1:
                name = st.text_input("Full name*", placeholder="Your full name")
                email = st.text_input("Email*", placeholder="your.email@example.com")
            
            with col2:
                phone = st.text_input("Phone*", placeholder="(555) 123-4567")
                city_state = st.text_input("City/State*", placeholder="New York, NY")
            
            # SECTION 2: HOUSING SITUATION
            st.subheader("🏠 Housing Situation")
            housing_type = st.selectbox(
                "Housing type*",
                ["Apartment", "House with Yard", "House without Yard", "Farm/Rural Area"],
                help="Select the type of housing where the dog will live"
            )
            
            house_size = st.select_slider(
                "Approximate housing size",
                options=["Small (<50m²)", "Medium (50-100m²)", "Large (>100m²)"],
                value="Medium (50-100m²)"
            )
            
            # SECTION 3: FAMILY COMPOSITION
            st.subheader("👨‍👩‍👧‍👦 Family Composition")
            col1, col2 = st.columns(2)
            
            with col1:
                has_children = st.checkbox("Do you have children at home?")
                if has_children:
                    children_ages = st.multiselect(
                        "Children ages",
                        ["0-3 years", "4-6 years", "7-12 years", "13-17 years"]
                    )
                else:
                    children_ages = []
            
            with col2:
                has_other_dogs = st.checkbox("Do you have other dogs?")
                if has_other_dogs:
                    num_dogs = st.number_input("How many?", 1, 10, 1)
                
                has_cats = st.checkbox("Do you have cats?")
            
            # SECTION 4: EXPERIENCE
            st.subheader("🎓 Dog Experience")
            experience_level = st.selectbox(
                "Experience level*",
                [
                    "No experience (first dog)",
                    "Little experience (had a dog long time ago)",
                    "Moderate experience (had a dog recently)",
                    "Lots of experience (handled dogs with behavioral issues)",
                    "Professional (trainer/veterinarian)"
                ]
            )
            
            # SECTION 5: LIFESTYLE
            st.subheader("🏃‍♂️ Lifestyle")
            activity_level = st.select_slider(
                "Your activity level*",
                options=["Sedentary", "Moderate", "Active", "Very active"],
                value="Moderate"
            )
            
            daily_time_available = st.slider(
                "Daily hours available for the dog*",
                min_value=1,
                max_value=8,
                value=3,
                help="Includes walks, playtime, and care"
            )
            
            # SECTION 6: PREFERENCES
            st.subheader("❤️ Dog Preferences")
            col1, col2 = st.columns(2)
            
            with col1:
                preferred_size = st.selectbox(
                    "Preferred size (optional)",
                    ["No preference", "Small", "Medium", "Large"]
                )
            
            with col2:
                preferred_age = st.selectbox(
                    "Preferred age (optional)",
                    ["No preference", "Puppy", "Young", "Adult", "Senior"]
                )
            
            accept_special_needs = st.checkbox(
                "Accept dogs with special needs",
                help="Dogs needing special medical care or with physical limitations"
            )
            
            # SUBMIT BUTTON
            submitted = st.form_submit_button(
                "🔍 Find Compatible Dogs",
                use_container_width=True
            )
            
            # FORM SUBMISSION HANDLING
            if submitted:
                # VALIDATION: Check required fields
                if not all([name, email, phone, city_state]):
                    st.error("Please fill all required fields (*)")
                    return None
                
                # CREATE STRUCTURED ADOPTER PROFILE
                adopter_profile = {
                    'personal_info': {
                        'name': name,
                        'email': email,
                        'phone': phone,
                        'location': city_state
                    },
                    'housing': {
                        'type': housing_type,
                        'size': house_size
                    },
                    'family': {
                        'has_children': has_children,
                        'children_ages': children_ages,
                        'has_other_dogs': has_other_dogs,
                        'num_other_dogs': num_dogs if has_other_dogs else 0,
                        'has_cats': has_cats
                    },
                    'experience_level': experience_level.split(" (")[0],  # Remove parenthetical
                    'lifestyle': {
                        'activity_level': activity_level,
                        'daily_time_available': daily_time_available
                    },
                    'preferences': {
                        'preferred_size': None if preferred_size == "No preference" else preferred_size,
                        'preferred_age': None if preferred_age == "No preference" else preferred_age,
                        'accept_special_needs': accept_special_needs
                    },
                    'submission_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                return adopter_profile
        
        return None  # Form not submitted

class ShelterDashboard:
    """
    SHELTER MANAGEMENT DASHBOARD - ANALYTICS AND INSIGHTS
    
    PURPOSE: Provide shelter staff with data-driven insights
    FEATURES:
    1. Key Performance Indicators (KPIs)
    2. Visual analytics charts
    3. Priority dog listings
    4. Shelter performance metrics
    5. Data export functionality
    
    AUDIENCE: Shelter managers, adoption counselors, volunteers
    """
    
    def __init__(self, dogs_data):
        """
        INITIALIZE DASHBOARD WITH SHELTER DATA
        
        Args:
            dogs_data (pd.DataFrame): Processed shelter dog data
        """
        self.dogs_data = dogs_data
    
    def render(self):
        """
        RENDER COMPLETE DASHBOARD WITH ALL COMPONENTS
        
        DASHBOARD LAYOUT:
        1. KPI cards (top metrics)
        2. Visualization charts (middle)
        3. Data tables (bottom)
        4. Export options (footer)
        """
        st.title("📊 Shelter Management Dashboard")
        
        # DATA VALIDATION
        if self.dogs_data.empty:
            st.warning("No data available to show in the dashboard.")
            return
        
        # RENDER ALL DASHBOARD COMPONENTS
        self._render_kpis()      # Top metrics
        self._render_charts()    # Middle visualizations
        self._render_tables()    # Bottom data tables
        self._render_export()    # Footer export options
    
    def _render_kpis(self):
        """
        RENDER KEY PERFORMANCE INDICATOR CARDS
        
        KPIs SHOWN:
        1. Total dogs available
        2. High priority dogs
        3. Average difficulty level
        4. Average days in shelter
        """
        st.subheader("📈 Key Performance Indicators")
        
        # FOUR-COLUMN LAYOUT FOR KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_dogs = len(self.dogs_data)
            st.metric("Total Dogs", total_dogs)
        
        with col2:
            high_priority = len(self.dogs_data[self.dogs_data['adoption_priority'] >= 8])
            # Delta shows percentage of total that are high priority
            st.metric("High Priority", high_priority, delta=f"{high_priority/total_dogs*100:.1f}%")
        
        with col3:
            avg_difficulty = self.dogs_data['difficulty_level'].mean()
            st.metric("Average Difficulty", f"{avg_difficulty:.1f}/5")
        
        with col4:
            avg_days = self.dogs_data['days_in_shelter'].mean()
            st.metric("Days in Shelter (average)", f"{avg_days:.0f}")
    
    def _render_charts(self):
        """
        RENDER DATA VISUALIZATION CHARTS
        
        CHARTS:
        1. Size distribution (pie chart)
        2. Age distribution (bar chart)
        3. Priority vs Difficulty (scatter plot)
        """
        st.subheader("📊 Visual Analysis")
        
        col1, col2 = st.columns(2)  # Two-column chart layout
        
        with col1:
            # CHART 1: SIZE DISTRIBUTION (PIE)
            size_dist = self.dogs_data['size_category'].value_counts()
            fig1 = px.pie(
                values=size_dist.values,
                names=size_dist.index,
                title="Distribution by Size",
                hole=0.4,  # Donut chart for better aesthetics
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # CHART 2: AGE DISTRIBUTION (BAR)
            age_dist = self.dogs_data['age_category'].value_counts()
            fig2 = px.bar(
                x=age_dist.index,
                y=age_dist.values,
                title="Distribution by Age Category",
                color=age_dist.values,  # Color by value
                color_continuous_scale='viridis'  # Colorblind-friendly palette
            )
            fig2.update_layout(xaxis_title="Age Category", yaxis_title="Number of Dogs")
            st.plotly_chart(fig2, use_container_width=True)
        
        # CHART 3: PRIORITY VS DIFFICULTY ANALYSIS
        st.subheader("🎯 Priority Analysis")
        
        fig3 = px.scatter(
            self.dogs_data,
            x='difficulty_level',
            y='adoption_priority',
            color='size_category',      # Color points by size
            size='days_in_shelter',     # Size points by days in shelter
            hover_data=['DOG_NAME', 'breed', 'age'],  # Show on hover
            title="Priority vs Difficulty",
            labels={
                'difficulty_level': 'Difficulty Level',
                'adoption_priority': 'Adoption Priority'
            }
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    def _render_tables(self):
        """
        RENDER DATA TABLES FOR DETAILED ANALYSIS
        
        TABLES:
        1. High priority dogs (expandable)
        2. Shelter distribution statistics (expandable)
        """
        st.subheader("📋 Detailed Data")
        
        # TABLE 1: HIGH PRIORITY DOGS
        high_priority_dogs = self.dogs_data[
            self.dogs_data['adoption_priority'] >= 8
        ].sort_values('adoption_priority', ascending=False)
        
        if not high_priority_dogs.empty:
            with st.expander("🚨 High Priority Dogs for Adoption", expanded=True):
                display_cols = ['DOG_NAME', 'breed', 'age', 'size_category', 
                              'adoption_priority', 'shelter_name']
                st.dataframe(
                    high_priority_dogs[display_cols].head(10),  # Show top 10
                    use_container_width=True,
                    hide_index=True
                )
        
        # TABLE 2: SHELTER DISTRIBUTION STATISTICS
        with st.expander("🏢 Distribution by Shelter"):
            shelter_stats = self.dogs_data.groupby('shelter_name').agg({
                'DOG_NAME': 'count',
                'adoption_priority': 'mean',
                'days_in_shelter': 'mean'
            }).round(1)
            
            shelter_stats.columns = ['Number of Dogs', 'Average Priority', 'Average Days']
            shelter_stats = shelter_stats.sort_values('Number of Dogs', ascending=False)
            
            st.dataframe(shelter_stats, use_container_width=True)
    
    def _render_export(self):
        """
        RENDER DATA EXPORT OPTIONS
        
        EXPORT FORMATS:
        1. Complete shelter data (CSV)
        2. Priority dogs only (CSV)
        """
        st.subheader("📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Complete Data", use_container_width=True):
                self._download_dataframe(self.dogs_data, "complete_shelter_data")
        
        with col2:
            if st.button("📋 Export Priority Dogs", use_container_width=True):
                high_priority = self.dogs_data[self.dogs_data['adoption_priority'] >= 7]
                self._download_dataframe(high_priority, "priority_dogs")
    
    def _download_dataframe(self, df, filename):
        """
        CREATE DOWNLOAD LINK FOR DATAFRAME AS CSV
        
        TECHNIQUE: Base64 encoding for HTML download links
        ALTERNATIVE: Could use Streamlit's native download_button
        
        Args:
            df (pd.DataFrame): Data to export
            filename (str): Base filename for download
        """
        csv = df.to_csv(index=False)  # Convert dataframe to CSV string
        b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
        # Create HTML download link with current date
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}_{datetime.now().strftime("%Y%m%d")}.csv">Click to download</a>'
        st.markdown(href, unsafe_allow_html=True)

class ResultsDisplay:
    """
    RESULTS DISPLAY COMPONENT - PRESENTS MATCHING RESULTS
    
    RESPONSIBILITIES:
    1. Visualize match scores
    2. Display dog cards with information
    3. Provide export options
    4. Generate reports
    
    DESIGN: Card-based layout for easy scanning
    UX: Progressive information disclosure with expandable details
    """
    
    @staticmethod
    def render(matches):
        """
        RENDER COMPLETE RESULTS DISPLAY
        
        COMPONENTS:
        1. Title and summary statistics
        2. Score visualization chart
        3. Individual dog cards
        4. Export options
        
        Args:
            matches (list): List of match dictionaries from matching engine
        """
        if not matches:
            st.info("No compatible dogs found. Try adjusting your preferences.")
            return
        
        # HEADER WITH SUMMARY
        st.title("🏆 Your Recommended Dogs")
        st.markdown(f"We found **{len(matches)}** dogs compatible with your profile!")
        
        # RENDER ALL COMPONENTS
        ResultsDisplay._render_score_chart(matches)   # Visualization
        ResultsDisplay._render_dog_cards(matches)     # Individual dogs
        ResultsDisplay._render_export_options(matches) # Export
    
    @staticmethod
    def _render_score_chart(matches):
        """
        CREATE BAR CHART OF TOP MATCH SCORES
        
        Args:
            matches (list): Match results
        """
        # PREPARE DATA FOR CHARTING
        scores_df = pd.DataFrame([{
            'Dog': m['dog_name'],
            'Compatibility (%)': m['match_percentage'],
            'Level': m['match_level'].split()[0]  # First word of match level
        } for m in matches])
        
        # CREATE BAR CHART
        fig = px.bar(
            scores_df.head(10),  # Top 10 dogs
            x='Dog',
            y='Compatibility (%)',
            color='Level',  # Color bars by match level
            title="Top 10 Dogs by Compatibility",
            color_discrete_sequence=['#FF6B6B', '#FFD166', '#06D6A0', '#118AB2'],  # Custom palette
            text='Compatibility (%)'  # Show values on bars
        )
        
        # FORMATTING
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Compatibility (%)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_dog_cards(matches):
        """
        CREATE INDIVIDUAL DOG CARDS FOR TOP MATCHES
        
        CARD LAYOUT: Three-column grid
        CARD CONTENT: Metrics, compatibility notes, shelter info
        
        Args:
            matches (list): Match results
        """
        st.subheader("❤️ Your Best Matches")
        
        # SHOW TOP 8 DOGS (configurable)
        for i, match in enumerate(matches[:8], 1):
            with st.container():  # Container for spacing
                col1, col2, col3 = st.columns([1, 2, 1])  # Three-column card
                
                with col1:
                    # MEDAL EMOJIS FOR TOP 3
                    medal = ""
                    if i == 1:
                        medal = "🥇 "
                    elif i == 2:
                        medal = "🥈 "
                    elif i == 3:
                        medal = "🥉 "
                    
                    # METRIC DISPLAY
                    st.metric(
                        label=f"{medal}#{i} {match['dog_name']}",
                        value=f"{match['match_percentage']:.1f}%",
                        delta=match['match_level']
                    )
                    
                    # BASIC INFORMATION
                    st.write(f"**Breed:** {match['dog_breed']}")
                    st.write(f"**Age:** {match['dog_age']}")
                    st.write(f"**Size:** {match['dog_size']}")
                
                with col2:
                    # COMPATIBILITY NOTES
                    st.write("**Compatibility Points:**")
                    for note in match['compatibility_notes']:
                        st.write(f"• {note}")
                    
                    # SHELTER INFORMATION
                    st.write(f"**Shelter:** {match['shelter_name']}")
                
                with col3:
                    # ACTION BUTTONS
                    if st.button(f"📋 View details", key=f"detail_{match['dog_id']}", use_container_width=True):
                        st.session_state.selected_dog = match['dog_id']
                    
                    # CONTACT INFORMATION
                    st.write("**Interested?**")
                    st.info(f"Contact the shelter {match['shelter_name']}")
                
                st.markdown("---")  # Separator between cards
    
    @staticmethod
    def _render_export_options(matches):
        """
        PROVIDE EXPORT OPTIONS FOR MATCH RESULTS
        
        EXPORT FORMATS:
        1. Complete list (CSV)
        2. Text report (TXT)
        """
        st.subheader("💾 Export Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Complete List", use_container_width=True):
                df = pd.DataFrame(matches)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Click to download CSV",
                    data=csv,
                    file_name=f"adoption_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🖨️ Generate Report", use_container_width=True):
                report = ResultsDisplay._generate_report(matches)
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name=f"adoption_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
    
    @staticmethod
    def _generate_report(matches):
        """
        GENERATE TEXT REPORT OF RECOMMENDATIONS
        
        REPORT INCLUDES:
        1. Header with date and summary
        2. Individual dog entries with details
        3. Compatibility notes
        
        Args:
            matches (list): Match results
            
        Returns:
            str: Formatted text report
        """
        report = f"ADOPTION RECOMMENDATIONS REPORT\n"
        report += f"Date: {datetime.now().strftime('%m/%d/%Y %H:%M')}\n"
        report += f"Total dogs recommended: {len(matches)}\n\n"
        
        # ADD TOP 10 DOGS TO REPORT
        for i, match in enumerate(matches[:10], 1):
            report += f"{i}. {match['dog_name']} - {match['dog_breed']}\n"
            report += f"   Compatibility: {match['match_percentage']:.1f}% ({match['match_level']})\n"
            report += f"   Age: {match['dog_age']} | Size: {match['dog_size']}\n"
            report += f"   Shelter: {match['shelter_name']}\n"
            report += f"   Address: {match['shelter_address']}\n"
            
            # ADD COMPATIBILITY NOTES
            if match['compatibility_notes']:
                report += "   Positive points:\n"
                for note in match['compatibility_notes'][:3]:  # Top 3 notes
                    report += f"   - {note}\n"
            
            report += "\n"  # Blank line between entries
        
        return report

# ============================================================================
# SHELTER ADOPTION PAGES - PAGE FUNCTIONS
# ============================================================================

def show_home_page_adoption():
    """
    HOME PAGE FOR SHELTER ADOPTION SYSTEM
    
    PURPOSE: Introduction and overview of the adoption system
    CONTENT: Benefits, how it works, quick statistics
    """
    st.title("🐕 Welcome to the Intelligent Adoption System")
    
    col1, col2 = st.columns([2, 1])  # Main content + sidebar image
    
    with col1:
        st.markdown("""
        ## Find your perfect canine companion! 🐾
        
        **Our system uses intelligent algorithms to connect you with the ideal dog** 
        based on your lifestyle, experience, and needs.
        
        ### ✨ Benefits:
        - ✅ **Increased successful adoptions**
        - ✅ **Reduced returns**
        - ✅ **Optimized shelter resources**
        - ✅ **Perfect match for every family**
        
        ### 📋 How it works:
        1. **Fill out our detailed form**
        2. **Our algorithm calculates compatibility**
        3. **Receive personalized recommendations**
        4. **Meet your new friend** 🐶
        
        **👉 Click on 'Adoption Form' in the menu to get started!**
        """)
    
    with col2:
        # DECORATIVE IMAGE
        st.image(
            "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400&auto=format&fit=crop",
            caption="Your new best friend is waiting!"
        )
    
    # QUICK STATISTICS SECTION
    if 'dogs_data' in st.session_state and not st.session_state.dogs_data.empty:
        st.markdown("---")
        st.subheader("📊 Current Statistics")
        
        dogs_data = st.session_state.dogs_data
        total_dogs = len(dogs_data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dogs Available", total_dogs)
        
        with col2:
            seniors = len(dogs_data[dogs_data['age_category'] == 'Senior'])
            st.metric("Senior Dogs", seniors)
        
        with col3:
            high_priority = len(dogs_data[dogs_data['adoption_priority'] >= 8])
            st.metric("High Priority", high_priority)

def show_form_page_adoption():
    """
    ADOPTION FORM PAGE
    
    PURPOSE: Collect adopter information for matching
    WORKFLOW: Form → Matching → Results
    """
    adopter_profile = AdoptionFormUI.render_form()
    
    if adopter_profile and 'matching_engine' in st.session_state:
        # STORE PROFILE IN SESSION STATE
        st.session_state.adopter_profile = adopter_profile
        
        # CALCULATE MATCHES WITH PROGRESS INDICATOR
        with st.spinner("Calculating compatibility with all dogs..."):
            matches = st.session_state.matching_engine.calculate_matches(adopter_profile)
            st.session_state.matches = matches
        
        # SUCCESS FEEDBACK
        st.success(f"✅ We found {len(matches)} compatible dogs!")
        st.balloons()  # Celebration animation
        
        # NAVIGATION TO RESULTS
        if st.button("🏆 View My Results", use_container_width=True):
            st.session_state.show_results = True
            st.rerun()

def show_results_page_adoption():
    """
    RESULTS PAGE FOR ADOPTION MATCHES
    
    PURPOSE: Display matching results to adopter
    CONDITIONAL: Only shows if matches exist
    """
    if not st.session_state.matches:
        st.info("ℹ️ No results available. Please fill out the form first.")
        
        # NAVIGATION BACK TO FORM
        if st.button("📋 Go to Form", use_container_width=True):
            st.session_state.show_results = False
            st.rerun()
    else:
        ResultsDisplay.render(st.session_state.matches)

def show_dashboard_page_adoption():
    """
    SHELTER DASHBOARD PAGE
    
    PURPOSE: Analytics and management tools for shelter staff
    ACCESS: Typically for shelter administrators
    """
    if 'dogs_data' in st.session_state and not st.session_state.dogs_data.empty:
        dashboard = ShelterDashboard(st.session_state.dogs_data)
        dashboard.render()
    else:
        st.warning("⚠️ Data not loaded. Please check if the CSV file is in the folder.")

def show_dogs_page_adoption():
    """
    ALL DOGS BROWSING PAGE
    
    PURPOSE: Browse all available dogs with filters
    FEATURES: Filtering, pagination, detailed views
    """
    st.title("🐕 All Available Dogs")
    
    # DATA VALIDATION
    if 'dogs_data' not in st.session_state or st.session_state.dogs_data.empty:
        st.warning("Data not available.")
        return
    
    dogs_data = st.session_state.dogs_data
    
    # FILTERS IN SIDEBAR
    st.sidebar.header("🔍 Advanced Filters")
    
    # SIZE FILTER
    sizes = ['All'] + sorted(dogs_data['size_category'].unique().tolist())
    selected_size = st.sidebar.selectbox("Size", sizes)
    
    # AGE FILTER
    ages = ['All'] + sorted(dogs_data['age_category'].unique().tolist())
    selected_age = st.sidebar.selectbox("Age Category", ages)
    
    # DIFFICULTY FILTER
    difficulties = ['All'] + sorted(dogs_data['difficulty_level'].unique().tolist())
    selected_difficulty = st.sidebar.selectbox("Difficulty Level", difficulties)
    
    # APPLY FILTERS
    filtered_dogs = dogs_data.copy()
    
    if selected_size != 'All':
        filtered_dogs = filtered_dogs[filtered_dogs['size_category'] == selected_size]
    
    if selected_age != 'All':
        filtered_dogs = filtered_dogs[filtered_dogs['age_category'] == selected_age]
    
    if selected_difficulty != 'All':
        filtered_difficulty = int(selected_difficulty) if selected_difficulty != 'All' else selected_difficulty
        filtered_dogs = filtered_dogs[filtered_dogs['difficulty_level'] == filtered_difficulty]
    
    # RESULTS SUMMARY
    st.write(f"**Showing {len(filtered_dogs)} of {len(dogs_data)} dogs**")
    
    # PAGINATION SYSTEM
    items_per_page = 12
    total_pages = max(1, len(filtered_dogs) // items_per_page + (1 if len(filtered_dogs) % items_per_page > 0 else 0))
    
    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    
    # CALCULATE PAGE SLICE
    start_idx = (page_number - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_dogs))
    
    # DISPLAY DOGS IN THREE-COLUMN GRID
    cols = st.columns(3)
    
    for idx in range(start_idx, end_idx):
        dog = filtered_dogs.iloc[idx]
        col_idx = (idx - start_idx) % 3  # Distribute across columns
        
        with cols[col_idx]:
            with st.expander(f"🐕 {dog['DOG_NAME']}", expanded=False):
                # BASIC INFORMATION
                st.write(f"**Breed:** {dog['breed']}")
                st.write(f"**Age:** {dog['age']} ({dog['age_category']})")
                st.write(f"**Size:** {dog['size_category']}")
                st.write(f"**Gender:** {dog['gender']}")
                
                # COMPATIBILITY TAGS
                tags = []
                if dog.get('child_compatibility') == 'yes':
                    tags.append("👶 Children")
                if dog.get('dog_compatibility') == 'yes':
                    tags.append("🐕 Other dogs")
                
                if tags:
                    st.write("**Compatible with:** " + " | ".join(tags))
                
                # DIFFICULTY INDICATOR
                difficulty = dog.get('difficulty_level', 3)
                if difficulty >= 4:
                    st.error(f"🎓 Experience needed ({difficulty}/5)")
                elif difficulty <= 2:
                    st.success(f"✨ Beginner ok ({difficulty}/5)")
                else:
                    st.info(f"⚡ Moderate ({difficulty}/5)")
                
                # PRIORITY INDICATOR
                priority = dog.get('adoption_priority', 5)
                if priority >= 8:
                    st.warning(f"🚨 High priority ({priority}/10)")
                
                # SHELTER INFORMATION
                st.write(f"**Shelter:** {dog['shelter_name']}")
                
                # DESCRIPTION VIEWER
                if st.button("📄 View full description", key=f"desc_{dog['dog_id']}"):
                    st.write("**Description:**")
                    # Truncate long descriptions
                    st.write(dog['description'][:500] + "..." if len(str(dog['description'])) > 500 else dog['description'])

# ============================================================================
# SIMPLIFIED STREAMLIT APP - MAIN APPLICATION CONTROLLER
# ============================================================================

def main():
    """
    MAIN APPLICATION CONTROLLER - COORDINATES BOTH SYSTEMS
    
    ARCHITECTURE:
    1. Initialize session state for both systems
    2. Load shelter data on startup
    3. Handle navigation between systems and pages
    4. Route to appropriate page functions
    
    DESIGN PATTERN: Front Controller pattern
    """
    st.title("🐾 Intelligent Dog Adoption System")
    st.markdown("---")
    
    # ========================================================================
    # SESSION STATE INITIALIZATION
    # ========================================================================
    
    # BREED ANALYSIS SYSTEM SESSION STATE
    if 'breeds_data' not in st.session_state:
        st.session_state.breeds_data = None  # Store loaded breed data
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False  # Track breed data loading status
    
    # SHELTER ADOPTION SYSTEM SESSION STATE
    if 'matches' not in st.session_state:
        st.session_state.matches = []  # Store matching results
    if 'adopter_profile' not in st.session_state:
        st.session_state.adopter_profile = None  # Store adopter profile
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False  # Track results display state
    
    # ========================================================================
    # DATA LOADING ON STARTUP
    # ========================================================================
    
    # LOAD SHELTER DATA ONCE AT STARTUP (Efficiency)
    if 'dogs_data' not in st.session_state:
        with st.spinner("Loading shelter dog data..."):
            dogs_data = DogDataProcessor.load_and_process_data()
            if not dogs_data.empty:
                st.session_state.dogs_data = dogs_data
                # Initialize matching engine with loaded data
                st.session_state.matching_engine = AdoptionMatchingEngine(dogs_data)
            else:
                st.session_state.dogs_data = pd.DataFrame()  # Empty on failure
    
    # ========================================================================
    # NAVIGATION SYSTEM
    # ========================================================================
    
    # SIDEBAR NAVIGATION
    st.sidebar.title("Navigation")
    
    # MAIN SYSTEM SELECTION (Top-level navigation)
    app_mode = st.sidebar.selectbox(
        "Choose System",
        ["Breed Analysis System", "Shelter Adoption System"]
    )
    
    # ========================================================================
    # BREED ANALYSIS SYSTEM ROUTING
    # ========================================================================
    
    if app_mode == "Breed Analysis System":
        # Breed analysis sub-navigation
        breed_mode = st.sidebar.selectbox(
            "Breed Analysis",
            ["Home", "Breed Analysis", "Find Breed Matches", "About"]
        )
        
        # Route to appropriate breed analysis page
        if breed_mode == "Home":
            render_home_page()
        elif breed_mode == "Breed Analysis":
            render_breed_analysis()
        elif breed_mode == "Find Breed Matches":
            render_find_matches()
        elif breed_mode == "About":
            render_about_page()
    
    # ========================================================================
    # SHELTER ADOPTION SYSTEM ROUTING
    # ========================================================================
    
    else:  # Shelter Adoption System
        # Shelter adoption sub-navigation
        adoption_mode = st.sidebar.selectbox(
            "Shelter Adoption",
            ["Home", "Adoption Form", "Results", "Shelter Dashboard", "All Dogs"]
        )
        
        # Route to appropriate shelter adoption page
        if adoption_mode == "Home":
            show_home_page_adoption()
        elif adoption_mode == "Adoption Form":
            show_form_page_adoption()
        elif adoption_mode == "Results":
            show_results_page_adoption()
        elif adoption_mode == "Shelter Dashboard":
            show_dashboard_page_adoption()
        elif adoption_mode == "All Dogs":
            show_dogs_page_adoption()

# ============================================================================
# ORIGINAL BREED ANALYSIS FUNCTIONS - KEEPING FOR COMPATIBILITY
# ============================================================================
# Note: These functions are maintained from the original system
# They provide breed analysis functionality separate from shelter adoption

def render_home_page():
    """Renders the home page of the breed analysis application."""
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
                caption="Find Your Perfect Match", use_container_width=True)
        
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
    """Renders the dog matching section for breed analysis."""
    st.header("🎯 Find Your Perfect Breed Match")
    
    if st.session_state.breeds_data is None or st.session_state.breeds_data.empty:
        st.warning("⚠️ Please load breed data first in the 'Breed Analysis' section.")
        return
    
    breeds_data = st.session_state.breeds_data
    
    st.subheader("Create Your Breed Preference Profile")
    
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
        
        submitted = st.form_submit_button("Find My Breed Matches 🐾", use_container_width=True)
        
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
        
        **Version:** 3.0.0 (Complete System with Breed Analysis & Shelter Matching)
        
        ### Purpose
        
        This comprehensive application is designed to help prospective dog owners 
        make informed decisions about dog adoption through two integrated systems:
        
        1. **Breed Analysis System** - Understand different breeds and their requirements
        2. **Shelter Adoption System** - Find specific dogs available for adoption
        
        ### How It Works
        
        **Breed Analysis System:**
        1. Load breed data from CSV files with automatic encoding detection
        2. Clean, encode, and normalize data for analysis
        3. View statistics, distributions, and correlations
        4. Use machine learning to group similar breeds
        5. Get personalized breed recommendations based on your profile
        
        **Shelter Adoption System:**
        1. Load shelter dog data with detailed descriptions
        2. Fill out comprehensive adoption form
        3. Get matched with specific available dogs
        4. View shelter dashboard with analytics
        5. Browse all available dogs with filters
        
        ### Technical Features
        
        - **Robust Data Loading**: Handles various CSV formats and encodings
        - **Machine Learning**: K-Means, Agglomerative, and DBSCAN clustering
        - **Dimensionality Reduction**: PCA and t-SNE for visualization
        - **Intelligent Matching**: Advanced algorithms for breed and dog matching
        - **Interactive Visualizations**: Plotly charts with hover information
        - **Responsive Design**: Works on desktop and mobile devices
        
        ### Data Requirements
        
        The system works with two types of CSV files:
        1. **Breed Data**: Includes breed names, size, exercise needs, grooming, etc.
        2. **Shelter Data**: Includes dog names, breeds, ages, descriptions, shelter info
        
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
        
        1. Start with Breed Analysis to understand your preferences
        2. Use Shelter Adoption to find specific available dogs
        3. Be honest in your profiles for better matches
        4. Explore both systems for comprehensive adoption planning
        
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
# ENTRY POINT - APPLICATION STARTUP
# ============================================================================

if __name__ == "__main__":
    """
    APPLICATION ENTRY POINT
    
    When executed directly (not imported), run the Streamlit application.
    This is the standard pattern for Streamlit applications.
    """
    main()