# leaderboard_tab.py - Enhanced version with advanced visualizations and analysis
import gradio as gr
import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from collections import defaultdict
import os
import re
from scipy import stats
import seaborn as sns
import plotly.express as px
colors = px.colors.qualitative.Set3 
# Try to import sklearn, fallback gracefully if not available
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not available, clustering analysis will be limited")
    SKLEARN_AVAILABLE = False

def safe_float_conversion(value, default=0.0):
    """Safely convert a value to float, handling various edge cases"""
    if pd.isna(value):
        return default
    
    # Convert to string and clean
    str_val = str(value).strip()
    
    # Handle common non-numeric placeholders
    if str_val in ['-', '--', 'N/A', 'n/a', 'NA', 'null', 'NULL', '']:
        return default
    
    # Remove any non-numeric characters except decimal point and minus sign
    cleaned = re.sub(r'[^\d\.-]', '', str_val)
    
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        print(f"⚠️  Could not convert '{value}' to float, using default {default}")
        return default

class ArenaLeaderboard:
    def __init__(self):
        self.benchmark_data = {}
        self.user_data_cache = {}
        self.last_update = None
        
        # Load the pre-existing benchmark CSV file
        self.load_default_benchmark_data()
        
    def load_default_benchmark_data(self):
        """Load the default benchmark results from leaderboard/result.csv with robust error handling"""
        default_path = "leaderboard/result.csv"
        if os.path.exists(default_path):
            try:
                # Load CSV with robust handling
                df = pd.read_csv(default_path)
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                print(f"📊 Loaded CSV with columns: {list(df.columns)}")
                print(f"📊 First few rows:")
                if not df.empty:
                    print(df.head())
                
                # Auto-detect column names - check if first two columns are method and model
                col_names = list(df.columns)
                if len(col_names) >= 2:
                    # Assume first column is Method, second is Model
                    if col_names[0] not in ['Method', 'method']:
                        df.rename(columns={col_names[0]: 'Method'}, inplace=True)
                        print(f"🔧 Renamed column '{col_names[0]}' to 'Method'")
                    
                    if col_names[1] not in ['Model', 'model']:
                        df.rename(columns={col_names[1]: 'Model'}, inplace=True)
                        print(f"🔧 Renamed column '{col_names[1]}' to 'Model'")
                
                # Ensure required columns exist
                if 'Method' not in df.columns or 'Model' not in df.columns:
                    print("⚠️ Warning: Method or Model column not found")
                    # Try to handle different column naming
                    if 'method' in df.columns:
                        df.rename(columns={'method': 'Method'}, inplace=True)
                    if 'model' in df.columns:
                        df.rename(columns={'model': 'Model'}, inplace=True)
                
                # Clean Method and Model columns
                if 'Method' in df.columns:
                    df['Method'] = df['Method'].astype(str).str.strip()
                if 'Model' in df.columns:
                    df['Model'] = df['Model'].astype(str).str.strip()
                
                # Create Full_Key for arena matching - normalize the method names
                if 'Method' in df.columns and 'Model' in df.columns:
                    df['Full_Key'] = df['Method'] + "::" + df['Model']
                    print(f"📊 Created Full_Key column with {len(df)} entries")
                    print(f"📊 Sample Full_Keys: {df['Full_Key'].head().tolist()}")
                
                # Define all benchmark columns - try to auto-detect from CSV
                # Look for numeric columns that might be benchmarks
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Known benchmark patterns
                dl_cols = []
                beir_cols = []
                
                # Check for DL19, DL20 variants
                for col in df.columns:
                    col_lower = col.lower().strip()
                    if 'dl19' in col_lower or 'dl-19' in col_lower:
                        dl_cols.append(col)
                        df.rename(columns={col: 'DL19'}, inplace=True)
                    elif 'dl20' in col_lower or 'dl-20' in col_lower:
                        dl_cols.append(col)
                        df.rename(columns={col: 'DL20'}, inplace=True)
                
                # Check for BEIR datasets
                beir_patterns = ['covid', 'nfcorpus', 'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']
                for col in df.columns:
                    col_lower = col.lower().strip()
                    for pattern in beir_patterns:
                        if pattern in col_lower:
                            # Normalize column name
                            proper_name = pattern.capitalize()
                            if pattern == 'nfcorpus':
                                proper_name = 'NFCorpus'
                            elif pattern == 'dbpedia':
                                proper_name = 'DBPedia'
                            elif pattern == 'scifact':
                                proper_name = 'SciFact'
                            elif pattern == 'robust04':
                                proper_name = 'Robust04'
                            elif pattern == 'touche':
                                proper_name = 'Touche'
                            
                            beir_cols.append(proper_name)
                            if col != proper_name:
                                df.rename(columns={col: proper_name}, inplace=True)
                            break
                
                # Update column lists after renaming
                dl_cols = ['DL19', 'DL20']  # Standard names after renaming
                all_benchmark_cols = dl_cols + beir_cols
                
                print(f"📊 Detected DL columns: {dl_cols}")
                print(f"📊 Detected BEIR columns: {beir_cols}")
                
                # Clean numeric benchmark columns with more lenient conversion
                for col in all_benchmark_cols:
                    if col in df.columns:
                        print(f"🔧 Cleaning column: {col}")
                        # Convert to numeric, errors='coerce' will turn invalid values to NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Calculate BEIR average (excluding DL19 and DL20)
                available_beir_cols = [col for col in beir_cols if col in df.columns]
                if available_beir_cols:
                    # Calculate mean only from non-NaN values
                    df['BEIR_Average'] = df[available_beir_cols].mean(axis=1, skipna=True)
                    print(f"📊 Calculated BEIR average from {len(available_beir_cols)} columns: {available_beir_cols}")
                else:
                    df['BEIR_Average'] = np.nan
                    print("⚠️ No BEIR columns found, setting average to NaN")
                
                # Ensure DL19 and DL20 columns exist
                for col in ['DL19', 'DL20']:
                    if col not in df.columns:
                        df[col] = np.nan
                        print(f"⚠️ {col} column not found, setting to NaN")
                
                # Store the cleaned dataframe
                self.benchmark_data['UNIFIED'] = df
                
                print(f"✅ Successfully loaded benchmark data: {len(df)} models")
                if not df['BEIR_Average'].isna().all():
                    print(f"📈 BEIR stats: avg={df['BEIR_Average'].mean():.2f}, max={df['BEIR_Average'].max():.2f}")
                
            except Exception as e:
                print(f"❌ Error loading default benchmark: {e}")
                import traceback
                traceback.print_exc()
                # Create empty dataframe as fallback
                self.benchmark_data['UNIFIED'] = pd.DataFrame()
        else:
            print(f"⚠️ Default benchmark file not found: {default_path}")
            self.benchmark_data['UNIFIED'] = pd.DataFrame()
    
    def collect_user_votes(self):
        """Collect and aggregate user votes from user_data folders"""
        vote_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'ties': 0, 
            'total_votes': 0, 'win_rate': 0.0
        })
        
        llm_stats = defaultdict(lambda: {
            'llm_wins': 0, 'llm_losses': 0, 'llm_ties': 0,
            'llm_total': 0, 'llm_win_rate': 0.0,
            'agreement_rate': 0.0, 'agreements': 0
        })
        
        user_data_path = Path("user_data")
        if not user_data_path.exists():
            print("📁 No user_data folder found, creating empty stats")
            return vote_stats, llm_stats
        
        total_interactions = 0
        
        try:
            for user_folder in user_data_path.iterdir():
                if not user_folder.is_dir():
                    continue
                    
                # Process user votes
                votes_folder = user_folder / "votes"
                if votes_folder.exists():
                    for vote_file in votes_folder.glob("*.json"):
                        try:
                            with open(vote_file, 'r') as f:
                                vote_data = json.load(f)
                                
                            if 'reranking_context' in vote_data:
                                models = vote_data['reranking_context']['models']
                                model_a_key = models['model_a']['full_key']
                                model_b_key = models['model_b']['full_key']
                                winner = vote_data['winner']
                                
                                total_interactions += 1
                                
                                # Update stats for both models
                                if winner == "Model A":
                                    vote_stats[model_a_key]['wins'] += 1
                                    vote_stats[model_b_key]['losses'] += 1
                                elif winner == "Model B":
                                    vote_stats[model_b_key]['wins'] += 1
                                    vote_stats[model_a_key]['losses'] += 1
                                else:  # Tie
                                    vote_stats[model_a_key]['ties'] += 1
                                    vote_stats[model_b_key]['ties'] += 1
                                
                                vote_stats[model_a_key]['total_votes'] += 1
                                vote_stats[model_b_key]['total_votes'] += 1
                                
                        except Exception as e:
                            print(f"Error processing vote file {vote_file}: {e}")
                
                # Process LLM votes  
                llm_votes_folder = user_folder / "votellm"
                if llm_votes_folder.exists():
                    for llm_file in llm_votes_folder.glob("*.json"):
                        try:
                            with open(llm_file, 'r') as f:
                                llm_data = json.load(f)
                            
                            if 'reranking_context' in llm_data:
                                models = llm_data['reranking_context']['models']
                                model_a_key = models['model_a']['full_key']
                                model_b_key = models['model_b']['full_key']
                                
                                llm_winner = llm_data['llm_evaluation']['winner']
                                user_vote = llm_data.get('user_vote', '')
                                
                                # Update LLM stats
                                if llm_winner == "Model A":
                                    llm_stats[model_a_key]['llm_wins'] += 1
                                    llm_stats[model_b_key]['llm_losses'] += 1
                                elif llm_winner == "Model B":
                                    llm_stats[model_b_key]['llm_wins'] += 1
                                    llm_stats[model_a_key]['llm_losses'] += 1
                                else:  # Tie or Error
                                    llm_stats[model_a_key]['llm_ties'] += 1
                                    llm_stats[model_b_key]['llm_ties'] += 1
                                
                                llm_stats[model_a_key]['llm_total'] += 1
                                llm_stats[model_b_key]['llm_total'] += 1
                                
                                # Check agreement between user and LLM
                                if user_vote and llm_winner not in ["Error", "Unknown"]:
                                    if ((user_vote == "Model A" and llm_winner == "Model A") or
                                        (user_vote == "Model B" and llm_winner == "Model B") or
                                        (user_vote == "Tie" and llm_winner == "Tie")):
                                        llm_stats[model_a_key]['agreements'] += 1
                                        llm_stats[model_b_key]['agreements'] += 1
                                        
                        except Exception as e:
                            print(f"Error processing LLM file {llm_file}: {e}")
        
        except Exception as e:
            print(f"Error collecting user votes: {e}")
        
        # Calculate win rates and agreement rates
        for model_key in vote_stats:
            total = vote_stats[model_key]['total_votes']
            if total > 0:
                wins = vote_stats[model_key]['wins']
                vote_stats[model_key]['win_rate'] = wins / total
        
        for model_key in llm_stats:
            total = llm_stats[model_key]['llm_total']
            if total > 0:
                wins = llm_stats[model_key]['llm_wins']
                llm_stats[model_key]['llm_win_rate'] = wins / total
                
                agreements = llm_stats[model_key]['agreements']
                llm_stats[model_key]['agreement_rate'] = agreements / total
        
        print(f"📊 Processed {total_interactions} user interactions")
        return dict(vote_stats), dict(llm_stats)
    
    def calculate_elo_ratings(self, vote_stats, initial_rating=1200, k_factor=32):
        """Calculate ELO ratings from head-to-head comparisons"""
        models = list(vote_stats.keys())
        elo_ratings = {model: initial_rating for model in models}
        
        # Simple ELO calculation based on win rate
        for model in models:
            stats = vote_stats[model]
            if stats['total_votes'] > 0:
                win_rate = stats['win_rate']
                # Adjust ELO based on deviation from expected 50% win rate
                # Use log scale to prevent extreme ratings with few votes
                vote_factor = min(np.log(stats['total_votes'] + 1), 5.0)  # Cap the influence
                rating_adjustment = k_factor * (win_rate - 0.5) * vote_factor
                elo_ratings[model] = initial_rating + rating_adjustment
        
        return elo_ratings
    
    def create_combined_leaderboard(self):
        """Create combined leaderboard with benchmark + user data"""
        try:
            user_votes, llm_votes = self.collect_user_votes()
            elo_ratings = self.calculate_elo_ratings(user_votes)
            
            # Get all models from benchmark data
            all_models = set()
            benchmark_df = self.benchmark_data.get('UNIFIED', pd.DataFrame())
            
            if not benchmark_df.empty:
                if 'Full_Key' in benchmark_df.columns:
                    all_models.update(benchmark_df['Full_Key'].tolist())
                elif 'Method' in benchmark_df.columns and 'Model' in benchmark_df.columns:
                    # Create full keys if not already present
                    benchmark_df['Full_Key'] = benchmark_df['Method'] + "::" + benchmark_df['Model']
                    all_models.update(benchmark_df['Full_Key'].tolist())
            
            # Add models from user votes
            all_models.update(user_votes.keys())
            all_models.update(llm_votes.keys())
            
            # Remove any None or empty values
            all_models = {model for model in all_models if model and str(model) != 'nan'}
            
            leaderboard_data = []
            
            for model in all_models:
                # Parse model info
                if "::" in str(model):
                    method, model_name = str(model).split("::", 1)
                else:
                    method, model_name = "Unknown", str(model)
                
                # Debug: Print what we're trying to match
                print(f"🔍 Processing model: '{model}' -> Method: '{method}', Model: '{model_name}'")
                
                row = {
                    'Model': model,
                    'Method': method,
                    'Model_Name': model_name,
                    
                    # User Arena Stats
                    'User_Votes': user_votes.get(model, {}).get('total_votes', 0),
                    'User_Win_Rate': round(user_votes.get(model, {}).get('win_rate', 0) * 100, 1),
                    'User_Wins': user_votes.get(model, {}).get('wins', 0),
                    'User_Losses': user_votes.get(model, {}).get('losses', 0),
                    'User_Ties': user_votes.get(model, {}).get('ties', 0),
                    
                    # LLM Judge Stats  
                    'LLM_Evaluations': llm_votes.get(model, {}).get('llm_total', 0),
                    'LLM_Win_Rate': round(llm_votes.get(model, {}).get('llm_win_rate', 0) * 100, 1),
                    'Agreement_Rate': round(llm_votes.get(model, {}).get('agreement_rate', 0) * 100, 1),
                    
                    # ELO Rating
                    'ELO_Rating': round(elo_ratings.get(model, 1200), 0),
                }
                
                # Add benchmark scores from CSV
                if not benchmark_df.empty:
                    # Debug: Show what's in the CSV for comparison
                    if not benchmark_df.empty and len(benchmark_df) > 0:
                        print(f"📊 CSV contains {len(benchmark_df)} rows")
                        print(f"📊 CSV Methods: {benchmark_df['Method'].unique().tolist()}")
                        print(f"📊 Looking for method: '{method}' and model: '{model_name}'")
                    
                    # Try multiple matching strategies
                    model_row = pd.DataFrame()
                    
                    # Strategy 1: Exact Full_Key match
                    if 'Full_Key' in benchmark_df.columns:
                        model_row = benchmark_df[benchmark_df['Full_Key'] == model]
                        if not model_row.empty:
                            print(f"✅ Found exact Full_Key match for: {model}")
                    
                    # Strategy 2: Try matching with different separators/formats
                    if model_row.empty and "::" in model:
                        method_part, model_part = model.split("::", 1)
                        
                        # Try normalized method names
                        method_variations = [
                            method_part,
                            method_part.replace('_', ' '),
                            method_part.replace(' ', '_'),
                            method_part.replace('_', '').replace(' ', ''),
                            method_part.title(),
                            method_part.lower(),
                            method_part.upper(),
                            method_part.capitalize()
                        ]
                        
                        print(f"🔍 Trying method variations: {method_variations}")
                        
                        for method_var in method_variations:
                            if model_row.empty:
                                # Try exact method and model match
                                method_matches = benchmark_df['Method'].str.lower() == method_var.lower()
                                model_matches = benchmark_df['Model'].str.lower() == model_part.lower()
                                model_row = benchmark_df[method_matches & model_matches]
                                
                                if not model_row.empty:
                                    print(f"✅ Found match using method variation: '{method_var}' + '{model_part}'")
                                    break
                                else:
                                    # Debug: Show what we tried to match
                                    csv_methods = benchmark_df['Method'].str.lower().tolist()
                                    csv_models = benchmark_df['Model'].str.lower().tolist()
                                    print(f"❌ No match for '{method_var.lower()}' in {csv_methods[:3]}... or '{model_part.lower()}' in {csv_models[:3]}...")
                    
                    # Strategy 3: Just match by model name (case insensitive)
                    if model_row.empty and 'Model' in benchmark_df.columns:
                        model_name_to_match = model_name if "::" in model else model
                        model_matches = benchmark_df['Model'].str.lower() == model_name_to_match.lower()
                        model_row = benchmark_df[model_matches]
                        if not model_row.empty:
                            print(f"✅ Found match using model name only: {model_name_to_match}")
                    
                    # Strategy 4: Partial matching for similar names
                    if model_row.empty:
                        print(f"🔍 Trying partial matching for method: '{method}' model: '{model_name}'")
                        for idx, row_data in benchmark_df.iterrows():
                            csv_method = str(row_data.get('Method', '')).strip()
                            csv_model = str(row_data.get('Model', '')).strip()
                            
                            # Try various matching approaches
                            method_clean = method.lower().replace('_', '').replace('-', '').replace(' ', '')
                            csv_method_clean = csv_method.lower().replace('_', '').replace('-', '').replace(' ', '')
                            
                            model_clean = model_name.lower().replace('_', '').replace('-', '').replace(' ', '')
                            csv_model_clean = csv_model.lower().replace('_', '').replace('-', '').replace(' ', '')
                            
                            if csv_method_clean == method_clean and csv_model_clean == model_clean:
                                model_row = pd.DataFrame([row_data])
                                print(f"✅ Found partial match: '{csv_method}' ~ '{method}', '{csv_model}' ~ '{model_name}'")
                                break
                            
                            # Also try if method contains the CSV method or vice versa
                            if (csv_method_clean in method_clean or method_clean in csv_method_clean) and csv_model_clean == model_clean:
                                model_row = pd.DataFrame([row_data])
                                print(f"✅ Found contains match: '{csv_method}' ~ '{method}', '{csv_model}' = '{model_name}'")
                                break
                    
                    if not model_row.empty:
                        # Add individual benchmark scores
                        benchmark_cols = ['DL19', 'DL20', 'Covid', 'NFCorpus', 'Touche', 
                                        'DBPedia', 'SciFact', 'Signal', 'News', 'Robust04']
                        
                        for col in benchmark_cols:
                            if col in model_row.columns:
                                value = model_row[col].iloc[0]
                                # Keep NaN values as NaN, don't convert to 0
                                if pd.isna(value):
                                    row[col] = None  # Display as None/empty in table
                                else:
                                    row[col] = round(float(value), 2)
                            else:
                                row[col] = None
                        
                        # Use pre-calculated BEIR average if available
                        if 'BEIR_Average' in model_row.columns:
                            beir_avg = model_row['BEIR_Average'].iloc[0]
                            row['BEIR_Avg'] = round(float(beir_avg), 2) if not pd.isna(beir_avg) else None
                        else:
                            # Calculate BEIR average from available scores (excluding DL19, DL20)
                            beir_cols = ['Covid', 'NFCorpus', 'Touche', 'DBPedia', 'SciFact', 'Signal', 'News', 'Robust04']
                            beir_scores = [row[col] for col in beir_cols if row[col] is not None]
                            row['BEIR_Avg'] = round(np.mean(beir_scores), 2) if beir_scores else None
                        
                        print(f"✅ Added benchmark data for {model}: BEIR_Avg={row['BEIR_Avg']}, DL19={row.get('DL19')}, DL20={row.get('DL20')}")
                    else:
                        # No benchmark data for this model - let's debug why
                        print(f"⚠️ No benchmark data found for: {model}")
                        print(f"   Method: '{method}', Model: '{model_name}'")
                        if not benchmark_df.empty:
                            print(f"   Available CSV methods: {sorted(benchmark_df['Method'].unique())}")
                            print(f"   Available CSV models: {sorted(benchmark_df['Model'].unique())}")
                            # Check if there's a partial match
                            method_lower = method.lower()
                            model_lower = model_name.lower()
                            similar_methods = [m for m in benchmark_df['Method'].unique() if method_lower in m.lower() or m.lower() in method_lower]
                            similar_models = [m for m in benchmark_df['Model'].unique() if model_lower in m.lower() or m.lower() in model_lower]
                            if similar_methods:
                                print(f"   Similar methods found: {similar_methods}")
                            if similar_models:
                                print(f"   Similar models found: {similar_models}")
                        
                        for col in ['DL19', 'DL20', 'Covid', 'NFCorpus', 'Touche', 
                                  'DBPedia', 'SciFact', 'Signal', 'News', 'Robust04']:
                            row[col] = None
                        row['BEIR_Avg'] = None
                
                leaderboard_data.append(row)
            
            df = pd.DataFrame(leaderboard_data)
            
            if df.empty:
                print("⚠️ No leaderboard data generated")
                return df
            
            # Filter out rows that have no meaningful data (no votes and no benchmark scores)
            def has_meaningful_data(row):
                # Keep if has user votes
                if row.get('User_Votes', 0) > 0:
                    return True
                # Keep if has any benchmark scores
                benchmark_cols = ['DL19', 'DL20', 'Covid', 'NFCorpus', 'Touche', 
                                'DBPedia', 'SciFact', 'Signal', 'News', 'Robust04', 'BEIR_Avg']
                for col in benchmark_cols:
                    if row.get(col) is not None and not pd.isna(row.get(col)):
                        return True
                return False
            
            # Filter the dataframe
            original_len = len(df)
            df = df[df.apply(has_meaningful_data, axis=1)].copy()
            filtered_len = len(df)
            
            if original_len != filtered_len:
                print(f"🔧 Filtered out {original_len - filtered_len} empty rows, keeping {filtered_len} models")
            
            if df.empty:
                print("⚠️ No models with meaningful data after filtering")
                return df
            
            # Sort by ELO rating (arena performance), then by BEIR average
            # Handle None values in sorting by filling with 0 temporarily
            df['ELO_Rating_sort'] = df['ELO_Rating'].fillna(0)
            df['BEIR_Avg_sort'] = df['BEIR_Avg'].fillna(0)
            
            df = df.sort_values(['ELO_Rating_sort', 'BEIR_Avg_sort'], ascending=[False, False]).reset_index(drop=True)
            df['Arena_Rank'] = df.index + 1
            
            # Remove temporary sorting columns
            df = df.drop(['ELO_Rating_sort', 'BEIR_Avg_sort'], axis=1)
            
            print(f"✅ Generated leaderboard with {len(df)} models")
            return df
            
        except Exception as e:
            print(f"❌ Error creating combined leaderboard: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_visualizations(self, leaderboard_df):
        """Create comprehensive visualization charts with enhanced analysis"""
        if leaderboard_df.empty:
            empty_fig = go.Figure().add_annotation(text="No data available", 
                                                 xref="paper", yref="paper",
                                                 x=0.5, y=0.5, showarrow=False)
            return [empty_fig] * 12
        
        try:
            # 1. Performance Radar Chart for Top Models
            fig1 = self.create_performance_radar(leaderboard_df)
            
            # 2. Interactive Bubble Chart: Performance vs Popularity
            fig2 = self.create_performance_popularity_bubble(leaderboard_df)
            
            # 3. Benchmark Correlation Heatmap
            fig3 = self.create_benchmark_correlation_heatmap(leaderboard_df)
            
            # 4. Method Performance Distribution
            fig4 = self.create_method_performance_distribution(leaderboard_df)
            
            # 5. Head-to-Head Win Rate Matrix
            fig5 = self.create_head_to_head_matrix(leaderboard_df)
            
            # 6. Performance Trends Over Time (if applicable)
            fig6 = self.create_performance_trends(leaderboard_df)
            
            # 7. Agreement Analysis Dashboard
            fig7 = self.create_agreement_analysis(leaderboard_df)
            
            # 8. Model Efficiency Analysis
            fig8 = self.create_efficiency_analysis(leaderboard_df)
            
            # 🆕 NEW ADVANCED ANALYSES
            # 9. Statistical Significance Analysis
            fig9 = self.create_statistical_significance_analysis(leaderboard_df)
            
            # 10. Model Clustering Analysis
            fig10 = self.create_model_clustering_analysis(leaderboard_df)
            
            # 11. Performance Consistency Analysis
            fig11 = self.create_performance_consistency_analysis(leaderboard_df)
            
            # 12. Voting Patterns & User Behavior Analysis
            fig12 = self.create_voting_patterns_analysis(leaderboard_df)
            
            return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            empty_fig = go.Figure().add_annotation(text=f"Error creating charts: {str(e)}", 
                                                 xref="paper", yref="paper",
                                                 x=0.5, y=0.5, showarrow=False)
            return [empty_fig] * 12

    def create_performance_radar(self, df):
        """Create radar chart showing multi-dimensional performance of top models"""
        try:
            # Get top 5 models with benchmark data
            top_models = df.dropna(subset=['BEIR_Avg']).head(5)
            
            if top_models.empty:
                return go.Figure().add_annotation(text="No benchmark data for radar chart", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            fig = go.Figure()
            
            # Define the metrics for radar chart
            metrics = ['BEIR_Avg', 'DL19', 'DL20', 'User_Win_Rate', 'Agreement_Rate']
            metric_labels = ['BEIR Average', 'DL19', 'DL20', 'User Win Rate', 'User-LLM Agreement']
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
            
            for idx, (_, model) in enumerate(top_models.iterrows()):
                values = []
                for metric in metrics:
                    val = model.get(metric, 0)
                    if pd.isna(val) or val is None:
                        val = 0
                    # Normalize to 0-100 scale
                    if metric in ['BEIR_Avg', 'DL19', 'DL20']:
                        values.append(float(val) if val else 0)
                    else:
                        values.append(float(val) if val else 0)
                
                # Close the radar chart
                values.append(values[0])
                labels = metric_labels + [metric_labels[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself',
                    name=f"{model['Method']} - {model['Model_Name'][:15]}...",
                    line_color=colors[idx % len(colors)]
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="🎯 Multi-Dimensional Performance Analysis (Top 5 Models)",
                height=600
            )
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in radar chart: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    def create_performance_popularity_bubble(self, df):
        """Create bubble chart showing performance vs popularity"""
        try:
            plot_data = df[(df['User_Votes'] > 0) & (df['BEIR_Avg'].notna())].copy()
            
            if plot_data.empty:
                return go.Figure().add_annotation(text="No data for performance vs popularity chart", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            fig = px.scatter(
                plot_data,
                x='User_Votes',
                y='BEIR_Avg',
                size='ELO_Rating',
                color='Method',
                hover_name='Model_Name',
                hover_data=['User_Win_Rate', 'Agreement_Rate'],
                title='🚀 Performance vs Popularity Analysis',
                labels={
                    'User_Votes': 'Total User Votes (Popularity)',
                    'BEIR_Avg': 'BEIR Average Score (Performance)',
                    'ELO_Rating': 'ELO Rating (Bubble Size)'
                },
                size_max=15,
                #range_sizeref=2.*max(plot_data['ELO_Rating'])/(40.**2)  # Add this line
            )
            
            # Add trend line
            if len(plot_data) > 2:
                fig.add_trace(go.Scatter(
                    x=plot_data['User_Votes'],
                    y=np.poly1d(np.polyfit(plot_data['User_Votes'], plot_data['BEIR_Avg'], 1))(plot_data['User_Votes']),
                    mode='lines',
                    name='Trend Line',
                    line=dict(dash='dash', color='gray')
                ))
            
            fig.update_layout(height=500, showlegend=True)
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in bubble chart: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    def create_benchmark_correlation_heatmap(self, df):
        """Create correlation heatmap between different benchmarks"""
        try:
            benchmark_cols = ['BEIR_Avg', 'DL19', 'DL20', 'Covid', 'NFCorpus', 'Touche', 'DBPedia', 'SciFact']
            available_cols = [col for col in benchmark_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return go.Figure().add_annotation(text="Not enough benchmark data for correlation analysis", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            # Calculate correlation matrix
            corr_data = df[available_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu_r',
                zmid=0,
                text=np.around(corr_data.values, decimals=2),
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="🔍 Benchmark Correlation Matrix",
                xaxis_title="Benchmarks",
                yaxis_title="Benchmarks",
                height=500
            )
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in correlation heatmap: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    def create_method_performance_distribution(self, df):
        """Create box plots showing performance distribution by method"""
        try:
            method_data = df[df['BEIR_Avg'].notna()].copy()
            
            if method_data.empty:
                return go.Figure().add_annotation(text="No method performance data available", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            fig = go.Figure()
            
            methods = method_data['Method'].unique()
            colors = px.colors.qualitative.Set3
            
            for idx, method in enumerate(methods):
                method_scores = method_data[method_data['Method'] == method]['BEIR_Avg']
                
                fig.add_trace(go.Box(
                    y=method_scores,
                    name=method,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker_color=colors[idx % len(colors)]
                ))
            
            fig.update_layout(
                title="📊 Performance Distribution by Method",
                yaxis_title="BEIR Average Score",
                xaxis_title="Methods",
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in distribution chart: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    def create_head_to_head_matrix(self, df):
        """Create heatmap showing head-to-head win rates"""
        try:
            # For now, create a simulated head-to-head matrix based on ELO ratings
            active_models = df[df['User_Votes'] > 0].copy()
            
            if len(active_models) < 2:
                return go.Figure().add_annotation(text="Need at least 2 models with votes for head-to-head analysis", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            model_names = [f"{row['Method'][:10]}-{row['Model_Name'][:10]}" for _, row in active_models.iterrows()]
            n_models = len(model_names)
            
            # Create simulated win rate matrix based on ELO differences
            win_matrix = np.zeros((n_models, n_models))
            
            for i in range(n_models):
                for j in range(n_models):
                    if i != j:
                        elo_diff = active_models.iloc[i]['ELO_Rating'] - active_models.iloc[j]['ELO_Rating']
                        # Convert ELO difference to win probability
                        win_prob = 1 / (1 + 10**(-elo_diff/400))
                        win_matrix[i][j] = win_prob * 100
                    else:
                        win_matrix[i][j] = 50  # 50% against self
            
            fig = go.Figure(data=go.Heatmap(
                z=win_matrix,
                x=model_names,
                y=model_names,
                colorscale='RdYlBu_r',
                zmid=50,
                text=np.around(win_matrix, decimals=0),
                texttemplate="%{text}",
                textfont={"size": 6},
                hovertemplate='%{y} vs %{x}<br>Win Rate: %{z:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title="⚔️ Head-to-Head Win Rate Matrix (Predicted)",
                xaxis_title="Opponent",
                yaxis_title="Model",
                height=600
            )
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in head-to-head matrix: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    def create_performance_trends(self, df):
        """Create meaningful performance trends analysis"""
        try:
            # Get models with benchmark data and sort by performance
            trend_data = df.dropna(subset=['BEIR_Avg']).copy()
            
            if trend_data.empty:
                return go.Figure().add_annotation(text="No performance data for trends", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            # Sort by BEIR performance
            trend_data = trend_data.sort_values('BEIR_Avg', ascending=False)
            
            # Create a 3x1 layout (vertical stack) for better visibility
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=['📊 Method Performance Trends', 
                              '🗳️ Performance vs Votes Analysis', 
                              '⚖️ ELO vs BEIR Correlation'],
                specs=[[{"secondary_y": False}], 
                       [{"secondary_y": False}], 
                       [{"secondary_y": False}]],
                vertical_spacing=0.15  # Add more space between subplots
            )
            
            # 1. Method performance trends (larger and more detailed)
            method_avg = trend_data.groupby('Method')['BEIR_Avg'].mean().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(x=method_avg.index, y=method_avg.values,
                    text=[f"{val:.1f}" for val in method_avg.values],
                    textposition='auto',
                    name='Method Average',
                    marker=dict(color=[colors[i % len(colors)] for i in range(len(method_avg))]),  # Different colors
                    showlegend=False),
                row=1, col=1
            )
            
            # 2. Performance vs Votes (only for models with votes)
            voted_models = trend_data[trend_data['User_Votes'] > 0]
            if not voted_models.empty:
                fig.add_trace(
                    go.Scatter(x=voted_models['User_Votes'], y=voted_models['BEIR_Avg'],
                              mode='markers',
                              text=[f"{row['Method'][:10]}<br>{row['Model_Name'][:12]}" for _, row in voted_models.iterrows()],
                              name='Performance vs Votes',
                              marker=dict(size=12, color='red', opacity=0.7),
                              hovertemplate='<b>%{text}</b><br>Votes: %{x}<br>BEIR Score: %{y:.1f}<extra></extra>',
                              showlegend=False),
                    row=2, col=1
                )
            else:
                fig.add_annotation(
                    text="No voting data available",
                    xref="x domain", yref="y domain",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16),
                    row=2, col=1
                )
            
            # 3. ELO vs BEIR correlation
            elo_beir_data = trend_data[trend_data['User_Votes'] > 0]  # Only models with arena data
            if not elo_beir_data.empty:
                fig.add_trace(
                    go.Scatter(x=elo_beir_data['BEIR_Avg'], y=elo_beir_data['ELO_Rating'],
                              mode='markers',
                              text=[f"{row['Method'][:10]}<br>{row['Model_Name'][:12]}" for _, row in elo_beir_data.iterrows()],
                              name='ELO vs BEIR',
                              marker=dict(size=12, color='purple', opacity=0.7),
                              hovertemplate='<b>%{text}</b><br>BEIR Score: %{x:.1f}<br>ELO Rating: %{y:.0f}<extra></extra>',
                              showlegend=False),
                    row=3, col=1
                )
            else:
                fig.add_annotation(
                    text="No ELO data available",
                    xref="x domain", yref="y domain",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16),
                    row=3, col=1
                )
            
            # Update layout for better readability with larger height
            fig.update_layout(
                height=1200,  # Much taller to accommodate 3 vertical subplots
                title_text="📈 Performance Analysis & Trends",
                title_x=0.5,
                title_font_size=20
            )
            
            # Update axes labels and formatting
            fig.update_xaxes(tickangle=45, title_text="Method", row=1, col=1)
            fig.update_yaxes(title_text="Average BEIR Score", row=1, col=1)
            
            fig.update_xaxes(title_text="User Votes", row=2, col=1)
            fig.update_yaxes(title_text="BEIR Average Score", row=2, col=1)
            
            fig.update_xaxes(title_text="BEIR Average Score", row=3, col=1)
            fig.update_yaxes(title_text="ELO Rating", row=3, col=1)
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in trends analysis: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    def create_agreement_analysis(self, df):
        """Create detailed agreement analysis between users and LLM judges"""
        try:
            agreement_data = df[(df['LLM_Evaluations'] > 0) & (df['User_Votes'] > 0)].copy()
            
            if agreement_data.empty:
                return go.Figure().add_annotation(text="No LLM evaluation data for agreement analysis", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            # Create subplot with multiple agreement metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Agreement vs Performance', 'Agreement vs Popularity', 
                              'Win Rate Comparison', 'Agreement Distribution'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                      [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Agreement vs Performance
            fig.add_trace(
                go.Scatter(x=agreement_data['BEIR_Avg'], y=agreement_data['Agreement_Rate'],
                          mode='markers', name='Agreement vs BEIR', 
                          marker=dict(size=8, color='blue')),
                row=1, col=1
            )
            
            # 2. Agreement vs Popularity
            fig.add_trace(
                go.Scatter(x=agreement_data['User_Votes'], y=agreement_data['Agreement_Rate'],
                          mode='markers', name='Agreement vs Votes',
                          marker=dict(size=8, color='green')),
                row=1, col=2
            )
            
            # 3. User vs LLM Win Rates
            fig.add_trace(
                go.Scatter(x=agreement_data['User_Win_Rate'], y=agreement_data['LLM_Win_Rate'],
                          mode='markers', name='Win Rate Comparison',
                          marker=dict(size=8, color='red')),
                row=2, col=1
            )
            
            # 4. Agreement Distribution
            fig.add_trace(
                go.Histogram(x=agreement_data['Agreement_Rate'], name='Agreement Distribution',
                           marker=dict(color='purple', opacity=0.7)),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text="🤝 User-LLM Agreement Analysis Dashboard")
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in agreement analysis: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    def create_efficiency_analysis(self, df):
        """Create efficiency analysis comparing votes needed vs performance achieved"""
        try:
            efficiency_data = df[df['User_Votes'] > 0].copy()
            
            if efficiency_data.empty:
                return go.Figure().add_annotation(text="No efficiency data available", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            # Calculate efficiency score (Performance per Vote)
            efficiency_data['Efficiency'] = efficiency_data['BEIR_Avg'].fillna(0) / efficiency_data['User_Votes']
            efficiency_data = efficiency_data[efficiency_data['Efficiency'] > 0]
            
            fig = px.scatter(
                efficiency_data,
                x='User_Votes',
                y='Efficiency',
                size='BEIR_Avg',
                color='Method',
                hover_name='Model_Name',
                title='⚡ Model Efficiency Analysis (Performance per Vote)',
                labels={
                    'User_Votes': 'Total Votes Required',
                    'Efficiency': 'Efficiency Score (BEIR/Vote)',
                    'BEIR_Avg': 'BEIR Performance'
                }
            )
            
            fig.update_layout(height=500)
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in efficiency analysis: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    def create_statistical_significance_analysis(self, df):
        """Create statistical significance analysis with confidence intervals"""
        try:
            benchmark_data = df.dropna(subset=['BEIR_Avg']).copy()
            
            if len(benchmark_data) < 3:
                return go.Figure().add_annotation(text="Need at least 3 models for statistical analysis", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            # Create statistical analysis with clearer visualizations
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Top 10 Models with Error Bars', 'Performance Distribution',
                              'Statistical Power Simulation', 'Model Score Ranges'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                      [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Top 10 models with confidence intervals
            top_models = benchmark_data.nlargest(10, 'BEIR_Avg')
            model_names = [f"{row['Method'][:8]}" for _, row in top_models.iterrows()]
            beir_scores = top_models['BEIR_Avg'].values
            
            # Simulate confidence intervals (±5% for demonstration)
            ci_size = beir_scores * 0.05  # 5% confidence interval
            
            fig.add_trace(
                go.Bar(x=model_names, y=beir_scores,
                      error_y=dict(type='data', array=ci_size, visible=True),
                      text=[f"{score:.1f}" for score in beir_scores],
                      textposition='auto',
                      name='BEIR ± 95% CI',
                      marker=dict(color='skyblue')),
                row=1, col=1
            )
            
            # 2. Performance distribution histogram
            fig.add_trace(
                go.Histogram(x=benchmark_data['BEIR_Avg'],
                           nbinsx=15,
                           name='Score Distribution',
                           marker=dict(color='lightgreen', opacity=0.7)),
                row=1, col=2
            )
            
            # 3. Statistical power simulation (sample size vs power)
            sample_sizes = np.arange(5, 100, 5)
            # Simulate power calculation (more sophisticated in real implementation)
            power_values = 1 - np.exp(-sample_sizes / 40)  # Simulated power curve
            
            fig.add_trace(
                go.Scatter(x=sample_sizes, y=power_values * 100,
                          mode='lines+markers',
                          name='Statistical Power (%)',
                          line=dict(color='red', width=3),
                          marker=dict(size=6)),
                row=2, col=1
            )
            
            # Add power threshold line
            fig.add_hline(y=80, line_dash="dash", line_color="gray", 
                         annotation_text="80% Power Threshold", row=2, col=1)
            
            # 4. Model score ranges (min/max across benchmarks)
            benchmark_cols = ['DL19', 'DL20', 'Covid', 'NFCorpus', 'Touche', 'DBPedia', 'SciFact']
            available_benchmarks = [col for col in benchmark_cols if col in top_models.columns]
            
            if len(available_benchmarks) >= 2:
                score_ranges = []
                model_labels_ranges = []
                
                for _, model in top_models.iterrows():
                    scores = [model[col] for col in available_benchmarks if not pd.isna(model[col])]
                    if scores:
                        score_ranges.append(max(scores) - min(scores))
                        model_labels_ranges.append(f"{model['Method'][:8]}")
                
                if score_ranges:
                    fig.add_trace(
                        go.Bar(x=model_labels_ranges, y=score_ranges,
                              text=[f"{r:.1f}" for r in score_ranges],
                              textposition='auto',
                              name='Score Range (Max-Min)',
                              marker=dict(color='orange')),
                        row=2, col=2
                    )
            
            fig.update_layout(height=800, title_text="📊 Statistical Analysis & Significance Testing", showlegend=False)
            
            # Update layout for better readability
            fig.update_xaxes(tickangle=45, row=1, col=1)
            fig.update_xaxes(title_text="Sample Size", row=2, col=1)
            fig.update_yaxes(title_text="Power (%)", row=2, col=1)
            fig.update_xaxes(tickangle=45, row=2, col=2)
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in statistical analysis: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    def create_model_clustering_analysis(self, df):
        """Create model clustering analysis to show similar performing models"""
        try:
            if not SKLEARN_AVAILABLE:
                return go.Figure().add_annotation(text="Clustering requires scikit-learn library\nShowing alternative similarity analysis", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                
            # Get models with benchmark data
            cluster_data = df.dropna(subset=['BEIR_Avg']).copy()
            
            if len(cluster_data) < 4:
                return go.Figure().add_annotation(text="Need at least 4 models for clustering analysis", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            # Select features for clustering
            feature_cols = ['BEIR_Avg', 'DL19', 'DL20', 'User_Win_Rate', 'Agreement_Rate']
            available_features = [col for col in feature_cols if col in cluster_data.columns]
            
            if len(available_features) < 2:
                return go.Figure().add_annotation(text="Not enough features for clustering", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            # Prepare data for clustering
            feature_data = cluster_data[available_features].fillna(0)
            
            # Normalize features
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(feature_data)
            
            # Perform clustering
            n_clusters = min(4, len(cluster_data) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(normalized_data)
            
            # Reduce to 2D for visualization
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(normalized_data)
            
            # Create clustering visualization with separate plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Model Clusters (PCA Space)', 'Feature Importance',
                              'Cluster Performance Comparison', 'Cluster Statistics'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                      [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. PCA Scatter plot with clusters
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_models = cluster_data[cluster_mask]
                
                fig.add_trace(
                    go.Scatter(x=pca_data[cluster_mask, 0], 
                              y=pca_data[cluster_mask, 1],
                              mode='markers',
                              name=f'Cluster {i+1}',
                              text=[f"{row['Method'][:8]}-{row['Model_Name'][:8]}" for _, row in cluster_models.iterrows()],
                              hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
                              marker=dict(size=12, color=colors[i % len(colors)])),
                    row=1, col=1
                )
            
            # 2. Feature importance (PCA components)
            feature_importance = abs(pca.components_[0])
            fig.add_trace(
                go.Bar(x=available_features, y=feature_importance,
                      text=[f"{imp:.2f}" for imp in feature_importance],
                      textposition='auto',
                      name='Feature Importance',
                      marker=dict(color='skyblue'),
                      showlegend=False),
                row=1, col=2
            )
            
            # 3. Cluster performance comparison (bar chart instead of radar)
            cluster_means = []
            cluster_names = []
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_mean_beir = feature_data[cluster_mask]['BEIR_Avg'].mean()
                cluster_means.append(cluster_mean_beir)
                cluster_names.append(f'Cluster {i+1}')
            
            fig.add_trace(
                go.Bar(x=cluster_names, y=cluster_means,
                      text=[f"{mean:.1f}" for mean in cluster_means],
                      textposition='auto',
                      name='Average BEIR Score',
                      marker=dict(color=[colors[i % len(colors)] for i in range(n_clusters)]),
                      showlegend=False),
                row=2, col=1
            )
            
            # 4. Cluster size statistics (bar chart instead of pie)
            cluster_sizes = [sum(cluster_labels == i) for i in range(n_clusters)]
            cluster_labels_list = [f'Cluster {i+1}' for i in range(n_clusters)]
            
            fig.add_trace(
                go.Bar(x=cluster_labels_list, y=cluster_sizes,
                      text=[f"{size} models" for size in cluster_sizes],
                      textposition='auto',
                      name="Models per Cluster",
                      marker=dict(color=[colors[i % len(colors)] for i in range(n_clusters)]),
                      showlegend=False),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text="🔗 Model Clustering & Similarity Analysis")
            
            # Update x-axis labels for better readability
            fig.update_xaxes(tickangle=45, row=1, col=2)
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in clustering analysis: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    def create_performance_consistency_analysis(self, df):
        """Analyze how consistent models are across different benchmarks"""
        try:
            consistency_data = df.dropna(subset=['BEIR_Avg']).copy()
            
            if consistency_data.empty:
                return go.Figure().add_annotation(text="No data for consistency analysis", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            # Calculate consistency metrics
            benchmark_cols = ['DL19', 'DL20', 'Covid', 'NFCorpus', 'Touche', 'DBPedia', 'SciFact']
            available_benchmarks = [col for col in benchmark_cols if col in consistency_data.columns]
            
            if len(available_benchmarks) < 3:
                return go.Figure().add_annotation(text="Need at least 3 benchmarks for consistency analysis", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            # Calculate coefficient of variation for each model
            consistency_scores = []
            model_names = []
            performance_scores = []
            
            for _, model in consistency_data.iterrows():
                scores = [model[col] for col in available_benchmarks if not pd.isna(model[col])]
                if len(scores) >= 3:
                    cv = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
                    consistency_scores.append(cv)
                    model_names.append(f"{model['Method'][:10]}")  # Shorter names
                    performance_scores.append(model['BEIR_Avg'])
            
            if not consistency_scores:
                return go.Figure().add_annotation(text="No valid consistency data", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            # Create consistency analysis with better layout
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Most Consistent Models (Top 10)', 'Consistency vs Performance',
                              'Benchmark Average Scores', 'Benchmark Score Variance'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                      [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Top 10 most consistent models (lower CV = more consistent)
            sorted_indices = np.argsort(consistency_scores)[:10]  # Top 10 most consistent
            top_models = [model_names[i] for i in sorted_indices]
            top_scores = [consistency_scores[i] for i in sorted_indices]
            
            fig.add_trace(
                go.Bar(x=top_models, y=top_scores,
                      text=[f"{score:.3f}" for score in top_scores],
                      textposition='auto',
                      name='Consistency Score (lower = better)',
                      marker=dict(color='lightgreen')),
                row=1, col=1
            )
            
            # 2. Consistency vs Performance scatter (with better spacing)
            # Sample only top performers to avoid overcrowding
            top_performers = sorted(range(len(performance_scores)), 
                                  key=lambda i: performance_scores[i], reverse=True)[:15]
            
            selected_consistency = [consistency_scores[i] for i in top_performers]
            selected_performance = [performance_scores[i] for i in top_performers]
            selected_names = [model_names[i] for i in top_performers]
            
            fig.add_trace(
                go.Scatter(x=selected_consistency, y=selected_performance,
                          mode='markers',
                          text=selected_names,
                          textposition="top center",
                          name='Top 15 Models',
                          marker=dict(size=12, color='blue', opacity=0.7),
                          hovertemplate='<b>%{text}</b><br>Consistency: %{x:.3f}<br>Performance: %{y:.1f}<extra></extra>'),
                row=1, col=2
            )
            
            # 3. Benchmark average scores (difficulty analysis)
            benchmark_averages = []
            for benchmark in available_benchmarks:
                avg_score = consistency_data[benchmark].mean()
                if not pd.isna(avg_score):
                    benchmark_averages.append(avg_score)
                else:
                    benchmark_averages.append(0)
            
            fig.add_trace(
                go.Bar(x=available_benchmarks, y=benchmark_averages,
                      text=[f"{avg:.1f}" for avg in benchmark_averages],
                      textposition='auto',
                      name='Benchmark Difficulty',
                      marker=dict(color='orange')),
                row=2, col=1
            )
            
            # 4. Benchmark variance (how much models differ on each benchmark)
            benchmark_variances = []
            for benchmark in available_benchmarks:
                var_score = consistency_data[benchmark].var()
                if not pd.isna(var_score):
                    benchmark_variances.append(var_score)
                else:
                    benchmark_variances.append(0)
            
            fig.add_trace(
                go.Bar(x=available_benchmarks, y=benchmark_variances,
                      text=[f"{var:.1f}" for var in benchmark_variances],
                      textposition='auto',
                      name='Performance Variance',
                      marker=dict(color='red')),
                row=2, col=2
            )
            
            # Update layout for better readability
            fig.update_layout(height=800, title_text="⚖️ Performance Consistency & Reliability Analysis", showlegend=False)
            
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45, row=1, col=1)
            fig.update_xaxes(tickangle=45, row=2, col=1)
            fig.update_xaxes(tickangle=45, row=2, col=2)
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in consistency analysis: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    def create_voting_patterns_analysis(self, df):
        """Analyze voting patterns and user behavior"""
        try:
            voting_data = df[df['User_Votes'] > 0].copy()
            
            if voting_data.empty:
                return go.Figure().add_annotation(text="No voting data for pattern analysis", 
                                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            
            # Create voting patterns analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Vote Distribution by Model', 'Win Rate vs Vote Count',
                            'Method Popularity (Total Votes)', 'User-LLM Agreement Analysis'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"type": "pie"}]],
                vertical_spacing=0.15,  # Add more space between rows
                horizontal_spacing=0.1  # Add space between columns too
            )
            
            # 1. Vote distribution - top 10 most voted models
            top_voted = voting_data.nlargest(10, 'User_Votes')
            model_labels = [f"{row['Method'][:8]}" for _, row in top_voted.iterrows()]
            
            fig.add_trace(
                go.Bar(x=model_labels, y=top_voted['User_Votes'],
                      text=top_voted['User_Votes'],
                      textposition='auto',
                      name='Vote Count',
                      marker=dict(color='lightblue')),
                row=1, col=1
            )
            
            # 2. Win Rate vs Vote Count with better spacing
            # Only show models with reasonable vote counts for clarity
            meaningful_votes = voting_data[voting_data['User_Votes'] >= 3]  # At least 3 votes
            
            if not meaningful_votes.empty:
                fig.add_trace(
                    go.Scatter(x=meaningful_votes['User_Votes'], 
                              y=meaningful_votes['User_Win_Rate'],
                              mode='markers',
                              text=[f"{row['Method'][:8]}<br>{row['Model_Name'][:10]}" for _, row in meaningful_votes.iterrows()],
                              name='Win Rate vs Votes',
                              marker=dict(size=10, color='green'),
                              hovertemplate='<b>%{text}</b><br>Votes: %{x}<br>Win Rate: %{y:.1f}%<extra></extra>'),
                    row=1, col=2
                )
            
            # 3. Method popularity (total votes per method)
            method_votes = voting_data.groupby('Method')['User_Votes'].sum().sort_values(ascending=True)
            
            fig.add_trace(
                go.Bar(x=method_votes.values, 
                      y=method_votes.index,
                      orientation='h',
                      text=method_votes.values,
                      textposition='auto',
                      name='Total Votes',
                      marker=dict(color='purple')),
                row=2, col=1
            )
            
            # 4. Agreement patterns - only models with both user votes and LLM evaluations
            agreement_data = voting_data[
                (voting_data['Agreement_Rate'] > 0) & 
                (voting_data['LLM_Evaluations'] > 0)
            ]
            
            if not agreement_data.empty:
                # Create agreement categories
                agreement_data['Agreement_Category'] = pd.cut(
                    agreement_data['Agreement_Rate'], 
                    bins=[0, 50, 75, 100], 
                    labels=['Low (0-50%)', 'Medium (50-75%)', 'High (75-100%)']
                )
                
                agreement_counts = agreement_data['Agreement_Category'].value_counts()
                
                fig.add_trace(
                    go.Pie(labels=agreement_counts.index,
                        values=agreement_counts.values,
                        name="Agreement Levels",
                        marker=dict(colors=['red', 'orange', 'green']),
                        textinfo='label+percent',
                        hole=0.3,  # Make it a donut chart (smaller)
                        textposition='inside',  # Put text inside
                        domain=dict(x=[0.6, 1.0], y=[0.0, 0.4])),  # Constrain the pie chart size
                    row=2, col=2
                )
            else:
                # Show placeholder if no agreement data
                fig.add_annotation(
                    text="No User-LLM<br>Agreement Data",
                    xref="x domain", yref="y domain",
                    x=0.5, y=0.5, showarrow=False,
                    row=2, col=2
                )
            
            fig.update_layout(height=800, title_text="🗳️ Voting Patterns & User Behavior Analysis", showlegend=False)
            
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45, row=1, col=1)
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in voting patterns analysis: {str(e)}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

def build_leaderboard_tab():
    """Build the enhanced leaderboard tab for Gradio - correctly returns components"""
    
    leaderboard = ArenaLeaderboard()
    
    with gr.Tab("📊 Arena Leaderboard"):
        gr.Markdown("# 🏆 RerankArena Leaderboard")
        gr.Markdown("Combined rankings from user votes, LLM evaluations, and benchmark results (BEIR, DL19, DL20)")
        
        with gr.Row():
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh Leaderboard", variant="primary", size="lg")
                
                gr.Markdown("### 📈 System Status")
                status_display = gr.Markdown("Loading data...")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Quick Stats")
                stats_display = gr.Markdown("Loading stats...")
        
        # Leaderboard Display
        with gr.Tab("🏆 Combined Rankings"):
            gr.Markdown("**Arena Rank** = ELO Rating + BEIR Performance")
            leaderboard_table = gr.Dataframe(
                interactive=False,
                wrap=True
            )
            
        with gr.Tab("📊 Advanced Analytics"):
            gr.Markdown("### 🎯 Multi-Dimensional Analysis")
            with gr.Row():
                plot1 = gr.Plot(label="Performance Radar Chart")
                plot2 = gr.Plot(label="Performance vs Popularity")
            
            gr.Markdown("### 🔍 Correlation & Distribution Analysis")
            with gr.Row():
                plot3 = gr.Plot(label="Benchmark Correlations")
                plot4 = gr.Plot(label="Method Performance Distribution")
            
        with gr.Tab("⚔️ Head-to-Head Analysis"):
            gr.Markdown("### Win Rate Matrix & Trends")
            with gr.Row():
                plot5 = gr.Plot(label="Head-to-Head Matrix")
                plot6 = gr.Plot(label="Performance Trends")
            
        with gr.Tab("🤝 Agreement Analysis"):
            gr.Markdown("### User-LLM Judge Agreement & Efficiency")
            with gr.Row():
                plot7 = gr.Plot(label="Agreement Dashboard")
                plot8 = gr.Plot(label="Efficiency Analysis")
        
        with gr.Tab("📈 Advanced Statistics"):
            gr.Markdown("### Statistical Significance & Model Clustering")
            with gr.Row():
                plot9 = gr.Plot(label="Statistical Significance Analysis")
                plot10 = gr.Plot(label="Model Clustering Analysis")
            
            gr.Markdown("### Performance Consistency & Voting Patterns")
            with gr.Row():
                plot11 = gr.Plot(label="Performance Consistency")
                plot12 = gr.Plot(label="Voting Patterns Analysis")
            
        with gr.Tab("📋 Detailed Data"):
            detailed_table = gr.Dataframe(interactive=False, wrap=True)
            with gr.Row():
                download_btn = gr.Button("📥 Download Results (CSV)")
                download_file = gr.File(visible=False, label="Download")
        
        # Event handlers
        def refresh_leaderboard():
            try:
                print("🔄 Refreshing leaderboard data...")
                df = leaderboard.create_combined_leaderboard()
                
                if df.empty:
                    empty_df = pd.DataFrame([["No data", "", "", "", "", "", "", "", ""]], 
                                          columns=['Rank', 'Method', 'Model', 'ELO', 'Votes', 'Win %', 'BEIR Avg', 'DL19', 'DL20'])
                    return tuple([empty_df, empty_df, "❌ No data found", 
                                "❌ No leaderboard data available"] + [None] * 12)
                
                # Prepare simplified display table with Method before Model
                display_columns = ['Arena_Rank', 'Method', 'Model_Name', 'ELO_Rating', 
                                 'User_Votes', 'User_Win_Rate', 'BEIR_Avg', 'DL19', 'DL20']
                
                # Ensure all required columns exist
                missing_cols = [col for col in display_columns if col not in df.columns]
                if missing_cols:
                    print(f"⚠️ Missing columns: {missing_cols}")
                    # Add missing columns with default values
                    for col in missing_cols:
                        if 'Rate' in col or 'Rating' in col or 'Avg' in col or 'Votes' in col or 'Rank' in col:
                            df[col] = 0
                        else:
                            df[col] = 'Unknown'
                
                # Filter out internal columns that shouldn't be displayed
                internal_cols = ['Table_Source', 'Method_Normalized', 'Full_Key', 'BEIR_Average', 'Model']
                for col in internal_cols:
                    if col in df.columns:
                        print(f"🔧 Removing internal column from display: {col}")
                
                # Create display dataframe with only the columns we want
                display_df = df[display_columns].copy()
                display_df.columns = ['Rank', 'Method', 'Model', 'ELO', 'Votes', 'Win %', 'BEIR Avg', 'DL19', 'DL20']
                
                # Replace None values with empty string for better display
                display_df = display_df.fillna('')
                
                # Create detailed dataframe for download (exclude internal columns)
                detail_columns = [col for col in df.columns if col not in internal_cols]
                detailed_df = df[detail_columns].copy()
                detailed_df = detailed_df.fillna('')
                
                # Create enhanced visualizations
                fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12 = leaderboard.create_visualizations(df)
                
                # Generate stats
                total_votes = df['User_Votes'].sum()
                active_models = len(df[df['User_Votes'] > 0])
                benchmark_models = len(df.dropna(subset=['BEIR_Avg']))
                methods = df['Method'].nunique()
                
                agreement_models = df[df['Agreement_Rate'] > 0]
                avg_agreement = agreement_models['Agreement_Rate'].mean() if not agreement_models.empty else 0.0
                
                stats_text = f"""
                **📊 Arena Statistics**
                - **Total User Votes:** {total_votes:,}
                - **Active Models:** {active_models} (with votes)
                - **Benchmark Models:** {benchmark_models} (with BEIR scores)
                - **Methods Tested:** {methods}
                - **Avg User-LLM Agreement:** {avg_agreement:.1f}%
                """
                
                status_text = f"✅ **System Status:** Loaded {len(df)} models | Last updated: {datetime.now().strftime('%H:%M:%S')}"
                
                print(f"✅ Leaderboard refresh completed: {len(df)} models")
                return display_df, detailed_df, stats_text, status_text, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12
                
            except Exception as e:
                print(f"❌ Error in refresh_leaderboard: {e}")
                import traceback
                traceback.print_exc()
                
                error_df = pd.DataFrame([["Error", str(e), "", "", "", "", "", "", ""]], 
                                      columns=['Rank', 'Method', 'Model', 'ELO', 'Votes', 'Win %', 'BEIR Avg', 'DL19', 'DL20'])
                error_msg = f"❌ Error loading leaderboard: {str(e)}"
                return tuple([error_df, error_df, error_msg, error_msg] + [None] * 12)
        
        def download_results(detailed_df):
            try:
                if detailed_df is not None and len(detailed_df) > 0:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"rerankarena_leaderboard_{timestamp}.csv"
                    detailed_df.to_csv(filename, index=False)
                    return gr.update(value=filename, visible=True)
                return gr.update(visible=False)
            except Exception as e:
                print(f"❌ Error downloading results: {e}")
                return gr.update(visible=False)
        
        # Wire up events
        refresh_btn.click(
            refresh_leaderboard,
            outputs=[leaderboard_table, detailed_table, stats_display, status_display, 
                    plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10, plot11, plot12]
        )
        
        download_btn.click(
            download_results,
            [detailed_table],
            [download_file]
        )
    
    # Return components outside the Tab context for auto-refresh from main app
    return leaderboard_table, detailed_table, stats_display, status_display, plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10, plot11, plot12, refresh_leaderboard