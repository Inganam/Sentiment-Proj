import pandas as pd
import numpy as np
from datetime import datetime
import re
import io

class DataProcessor:
    def __init__(self):
        pass
    
    def validate_csv_file(self, file_data):
        """Validate uploaded CSV file"""
        try:
            df = pd.read_csv(io.StringIO(file_data))
            
            if df.empty:
                raise ValueError("CSV file is empty")
            
            if len(df.columns) == 0:
                raise ValueError("CSV file has no columns")
            
            # Check for text columns
            text_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains text (not just numbers/dates)
                    sample_values = df[col].dropna().head(10)
                    if any(isinstance(val, str) and len(str(val).strip()) > 10 for val in sample_values):
                        text_columns.append(col)
            
            if not text_columns:
                raise ValueError("No suitable text columns found in CSV file")
            
            return {
                'valid': True,
                'dataframe': df,
                'text_columns': text_columns,
                'total_rows': len(df),
                'total_columns': len(df.columns)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def preprocess_text_column(self, df, column_name):
        """Preprocess text column for analysis"""
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataframe")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Convert to string and handle missing values
        processed_df[column_name] = processed_df[column_name].astype(str)
        processed_df[column_name] = processed_df[column_name].replace('nan', '')
        
        # Remove empty or very short texts
        processed_df = processed_df[processed_df[column_name].str.len() > 5]
        
        # Add text length column
        processed_df['text_length'] = processed_df[column_name].str.len()
        
        # Add word count column
        processed_df['word_count'] = processed_df[column_name].str.split().str.len()
        
        return processed_df
    
    def split_long_texts(self, df, column_name, max_length=5000):
        """Split texts that are too long into smaller chunks"""
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataframe")
        
        expanded_rows = []
        
        for idx, row in df.iterrows():
            text = str(row[column_name])
            
            if len(text) <= max_length:
                expanded_rows.append(row.to_dict())
            else:
                # Split into chunks
                chunks = self._split_text_into_chunks(text, max_length)
                for i, chunk in enumerate(chunks):
                    new_row = row.to_dict()
                    new_row[column_name] = chunk
                    new_row['original_index'] = idx
                    new_row['chunk_number'] = i + 1
                    new_row['total_chunks'] = len(chunks)
                    expanded_rows.append(new_row)
        
        return pd.DataFrame(expanded_rows)
    
    def _split_text_into_chunks(self, text, max_length):
        """Split text into chunks at sentence boundaries when possible"""
        if len(text) <= max_length:
            return [text]
        
        # Try to split at sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed max_length
            if len(current_chunk) + len(sentence) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk) + len(word) + 1 > max_length:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                            word_chunk = word
                        else:
                            word_chunk += " " + word if word_chunk else word
                    if word_chunk:
                        current_chunk = word_chunk
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def aggregate_results(self, results_list):
        """Aggregate analysis results into a structured format"""
        if not results_list:
            return pd.DataFrame()
        
        # Convert results to DataFrame
        df = pd.DataFrame(results_list)
        
        # Ensure timestamp is properly formatted
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add derived columns
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        # Calculate sentiment scores
        if 'detailed_scores' in df.columns:
            try:
                score_df = pd.json_normalize(df['detailed_scores'].tolist())
                df = pd.concat([df, score_df], axis=1)
            except Exception:
                pass  # Skip if normalization fails
        
        return df
    
    def filter_results(self, df, filters):
        """Apply filters to results DataFrame"""
        filtered_df = df.copy()
        
        # Sentiment filter
        if 'sentiment' in filters and filters['sentiment']:
            filtered_df = filtered_df[filtered_df['sentiment'].isin(filters['sentiment'])]
        
        # Confidence filter
        if 'min_confidence' in filters:
            filtered_df = filtered_df[filtered_df['confidence'] >= filters['min_confidence']]
        
        if 'max_confidence' in filters:
            filtered_df = filtered_df[filtered_df['confidence'] <= filters['max_confidence']]
        
        # Date range filter
        if 'start_date' in filters and 'end_date' in filters:
            if 'timestamp' in filtered_df.columns:
                start_date = pd.to_datetime(filters['start_date'])
                end_date = pd.to_datetime(filters['end_date'])
                filtered_df = filtered_df[
                    (filtered_df['timestamp'] >= start_date) &
                    (filtered_df['timestamp'] <= end_date)
                ]
        
        # Text length filter
        if 'min_text_length' in filters:
            filtered_df = filtered_df[filtered_df['text_length'] >= filters['min_text_length']]
        
        if 'max_text_length' in filters:
            filtered_df = filtered_df[filtered_df['text_length'] <= filters['max_text_length']]
        
        # Source filter
        if 'source' in filters and filters['source']:
            filtered_df = filtered_df[filtered_df['source'].isin(filters['source'])]
        
        return filtered_df
    
    def get_summary_statistics(self, df):
        """Calculate summary statistics for the results"""
        if df.empty:
            return {}
        
        stats = {
            'total_texts': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'average_confidence': df['confidence'].mean(),
            'confidence_std': df['confidence'].std(),
            'average_text_length': df['text_length'].mean() if 'text_length' in df.columns else 0,
            'average_word_count': df['word_count'].mean() if 'word_count' in df.columns else 0,
        }
        
        # Confidence quartiles
        stats['confidence_quartiles'] = {
            'q25': df['confidence'].quantile(0.25),
            'q50': df['confidence'].quantile(0.5),
            'q75': df['confidence'].quantile(0.75)
        }
        
        # Analysis method distribution
        if 'analysis_method' in df.columns:
            stats['analysis_methods'] = df['analysis_method'].value_counts().to_dict()
        
        # Time-based statistics
        if 'timestamp' in df.columns:
            stats['date_range'] = {
                'earliest': df['timestamp'].min(),
                'latest': df['timestamp'].max()
            }
            
            # Group by date
            df['date'] = df['timestamp'].dt.date
            daily_counts = df.groupby('date').size()
            stats['daily_analysis_count'] = {
                'average': daily_counts.mean(),
                'max': daily_counts.max(),
                'min': daily_counts.min()
            }
        
        return stats
    
    def prepare_export_data(self, df):
        """Prepare data for export by flattening nested structures"""
        export_df = df.copy()
        
        # Flatten detailed_scores if present
        if 'detailed_scores' in export_df.columns:
            score_df = pd.json_normalize(export_df['detailed_scores'])
            score_df.columns = [f'score_{col}' for col in score_df.columns]
            export_df = pd.concat([export_df.drop('detailed_scores', axis=1), score_df], axis=1)
        
        # Convert keywords list to string
        if 'keywords' in export_df.columns:
            export_df['keywords'] = export_df['keywords'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
        
        # Convert sentiment_phrases list to string
        if 'sentiment_phrases' in export_df.columns:
            export_df['sentiment_phrases'] = export_df['sentiment_phrases'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
        
        # Format timestamp
        if 'timestamp' in export_df.columns:
            export_df['timestamp'] = export_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return export_df
