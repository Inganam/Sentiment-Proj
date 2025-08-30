import pandas as pd
import json
import io
from datetime import datetime
import csv
from typing import List, Dict, Any

class ExportManager:
    def __init__(self):
        pass
    
    def export_to_csv(self, results_data):
        """Export analysis results to CSV format"""
        try:
            # Convert results to DataFrame if it's a list
            if isinstance(results_data, list):
                df = pd.DataFrame(results_data)
            elif isinstance(results_data, pd.DataFrame):
                df = results_data.copy()
            else:
                raise ValueError("Data must be a list or DataFrame")
            
            if df.empty:
                raise ValueError("No data to export")
            
            # Prepare data for export
            export_df = self._prepare_csv_data(df)
            
            # Convert to CSV string
            output = io.StringIO()
            export_df.to_csv(output, index=False, quoting=csv.QUOTE_NONNUMERIC)
            csv_data = output.getvalue()
            output.close()
            
            return csv_data
            
        except Exception as e:
            raise Exception(f"CSV export failed: {str(e)}")
    
    def export_to_json(self, results_data):
        """Export analysis results to JSON format"""
        try:
            # Convert results to list if it's a DataFrame
            if isinstance(results_data, pd.DataFrame):
                data_list = results_data.to_dict('records')
            elif isinstance(results_data, list):
                data_list = results_data.copy()
            else:
                raise ValueError("Data must be a list or DataFrame")
            
            if not data_list:
                raise ValueError("No data to export")
            
            # Prepare data for JSON export
            export_data = self._prepare_json_data(data_list)
            
            # Create complete export structure
            export_structure = {
                'export_metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_records': len(export_data),
                    'export_format': 'JSON',
                    'version': '1.0'
                },
                'summary_statistics': self._calculate_summary_stats(export_data),
                'analysis_results': export_data
            }
            
            # Convert to JSON string
            json_data = json.dumps(export_structure, indent=2, default=str)
            
            return json_data
            
        except Exception as e:
            raise Exception(f"JSON export failed: {str(e)}")
    
    def export_to_excel(self, results_data, include_charts=False):
        """Export analysis results to Excel format with multiple sheets"""
        try:
            # Convert results to DataFrame if it's a list
            if isinstance(results_data, list):
                df = pd.DataFrame(results_data)
            elif isinstance(results_data, pd.DataFrame):
                df = results_data.copy()
            else:
                raise ValueError("Data must be a list or DataFrame")
            
            if df.empty:
                raise ValueError("No data to export")
            
            # Create Excel file in memory
            output = io.BytesIO()
            
            # Note: Excel export requires openpyxl package
            try:
                import openpyxl
            except ImportError:
                raise ImportError("Excel export requires openpyxl package")
                
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main results sheet
                export_df = self._prepare_csv_data(df)
                export_df.to_excel(writer, sheet_name='Analysis_Results', index=False)
                
                # Summary statistics sheet
                summary_stats = self._calculate_summary_stats_df(df)
                summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=True)
                
                # Sentiment distribution sheet
                sentiment_dist = df['sentiment'].value_counts().reset_index()
                sentiment_dist.columns = ['Sentiment', 'Count']
                sentiment_dist.to_excel(writer, sheet_name='Sentiment_Distribution', index=False)
                
                # Keywords sheet (if available)
                if 'keywords' in df.columns:
                    keywords_df = self._create_keywords_summary(df)
                    if not keywords_df.empty:
                        keywords_df.to_excel(writer, sheet_name='Keywords_Summary', index=False)
            
            excel_data = output.getvalue()
            output.close()
            
            return excel_data
            
        except Exception as e:
            raise Exception(f"Excel export failed: {str(e)}")
    
    def _prepare_csv_data(self, df):
        """Prepare data for CSV export by flattening nested structures"""
        export_df = df.copy()
        
        # Flatten detailed_scores if present
        if 'detailed_scores' in export_df.columns:
            for idx, scores in export_df['detailed_scores'].items():
                if isinstance(scores, dict):
                    for sentiment, score in scores.items():
                        export_df.loc[idx, f'score_{sentiment}'] = score
            export_df = export_df.drop('detailed_scores', axis=1)
        
        # Convert lists to strings
        list_columns = ['keywords', 'sentiment_phrases']
        for col in list_columns:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x) if x else ''
                )
        
        # Format timestamp
        if 'timestamp' in export_df.columns:
            export_df['timestamp'] = pd.to_datetime(export_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Clean text columns
        text_columns = ['text', 'cleaned_text', 'explanation', 'text_preview']
        for col in text_columns:
            if col in export_df.columns:
                export_df[col] = export_df[col].astype(str).str.replace('\n', ' ').str.replace('\r', '')
        
        # Reorder columns for better readability
        preferred_order = [
            'timestamp', 'sentiment', 'confidence', 'text_preview', 'text',
            'keywords', 'explanation', 'analysis_method', 'source'
        ]
        
        # Add score columns
        score_columns = [col for col in export_df.columns if col.startswith('score_')]
        preferred_order.extend(score_columns)
        
        # Add remaining columns
        remaining_columns = [col for col in export_df.columns if col not in preferred_order]
        preferred_order.extend(remaining_columns)
        
        # Filter to only include existing columns
        final_order = [col for col in preferred_order if col in export_df.columns]
        export_df = export_df[final_order]
        
        return export_df
    
    def _prepare_json_data(self, data_list):
        """Prepare data for JSON export"""
        import numpy as np
        
        export_data = []
        
        for item in data_list:
            export_item = {}
            
            for key, value in item.items():
                try:
                    # Handle None first
                    if value is None:
                        export_item[key] = None
                    # Handle numpy arrays first to avoid truth value error
                    elif isinstance(value, np.ndarray):
                        if value.size == 0:
                            export_item[key] = []
                        else:
                            export_item[key] = value.tolist()
                    # Handle numpy types
                    elif isinstance(value, (np.integer, np.floating, np.bool_)):
                        export_item[key] = value.item()
                    # Handle NaN values (check after numpy handling)
                    elif pd.isna(value):
                        export_item[key] = None
                    # Handle datetime objects
                    elif isinstance(value, datetime):
                        export_item[key] = value.isoformat()
                    # Handle pandas Timestamp (check type explicitly)
                    elif str(type(value).__name__) == 'Timestamp':
                        export_item[key] = value.isoformat()
                    # Handle lists and dictionaries
                    elif isinstance(value, (list, dict)):
                        export_item[key] = self._sanitize_for_json(value)
                    # Handle other types
                    else:
                        # Convert to string if not JSON serializable
                        try:
                            json.dumps(value)
                            export_item[key] = value
                        except (TypeError, ValueError):
                            export_item[key] = str(value)
                except Exception as e:
                    # Fallback to string representation
                    export_item[key] = str(value) if value is not None else None
            
            export_data.append(export_item)
        
        return export_data
    
    def _calculate_summary_stats(self, data_list):
        """Calculate summary statistics for JSON export"""
        if not data_list:
            return {}
        
        df = pd.DataFrame(data_list)
        
        stats = {
            'total_analyses': len(df),
            'sentiment_distribution': {},
            'confidence_statistics': {},
            'analysis_methods': {},
            'sources': {}
        }
        
        # Sentiment distribution
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            stats['sentiment_distribution'] = {
                'counts': sentiment_counts.to_dict(),
                'percentages': (sentiment_counts / len(df) * 100).round(2).to_dict()
            }
        
        # Confidence statistics
        if 'confidence' in df.columns:
            stats['confidence_statistics'] = {
                'mean': float(df['confidence'].mean()),
                'std': float(df['confidence'].std()),
                'min': float(df['confidence'].min()),
                'max': float(df['confidence'].max()),
                'median': float(df['confidence'].median()),
                'q25': float(df['confidence'].quantile(0.25)),
                'q75': float(df['confidence'].quantile(0.75))
            }
        
        # Analysis methods
        if 'analysis_method' in df.columns:
            method_counts = df['analysis_method'].value_counts()
            stats['analysis_methods'] = method_counts.to_dict()
        
        # Sources
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            stats['sources'] = source_counts.to_dict()
        
        # Time range
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            stats['time_range'] = {
                'earliest': timestamps.min().isoformat(),
                'latest': timestamps.max().isoformat(),
                'duration_days': (timestamps.max() - timestamps.min()).days
            }
        
        return stats
    
    def _sanitize_for_json(self, obj):
        """Recursively sanitize objects for JSON serialization"""
        import numpy as np
        
        try:
            if obj is None:
                return None
            elif isinstance(obj, dict):
                return {k: self._sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._sanitize_for_json(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                if obj.size == 0:
                    return []
                else:
                    return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif pd.isna(obj):
                return None
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif str(type(obj).__name__) == 'Timestamp':
                return obj.isoformat()
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        except Exception:
            return str(obj) if obj is not None else None
    
    def _calculate_summary_stats_df(self, df):
        """Calculate summary statistics as a DataFrame for Excel export"""
        stats_data = []
        
        # Basic statistics
        stats_data.append(['Total Analyses', len(df)])
        
        # Sentiment distribution
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(df)) * 100
                stats_data.append([f'Sentiment - {sentiment.title()}', f'{count} ({percentage:.1f}%)'])
        
        # Confidence statistics
        if 'confidence' in df.columns:
            stats_data.append(['Average Confidence', f'{df["confidence"].mean():.3f}'])
            stats_data.append(['Confidence Std Dev', f'{df["confidence"].std():.3f}'])
            stats_data.append(['Min Confidence', f'{df["confidence"].min():.3f}'])
            stats_data.append(['Max Confidence', f'{df["confidence"].max():.3f}'])
        
        # Analysis methods
        if 'analysis_method' in df.columns:
            method_counts = df['analysis_method'].value_counts()
            for method, count in method_counts.items():
                stats_data.append([f'Method - {method}', count])
        
        # Text statistics
        if 'text_length' in df.columns:
            stats_data.append(['Average Text Length', f'{df["text_length"].mean():.0f} characters'])
        
        if 'word_count' in df.columns:
            stats_data.append(['Average Word Count', f'{df["word_count"].mean():.0f} words'])
        
        stats_df = pd.DataFrame(stats_data)
        if len(stats_df.columns) >= 2:
            stats_df.columns = ['Metric', 'Value']
        return stats_df
    
    def _create_keywords_summary(self, df):
        """Create a summary of keywords for Excel export"""
        if 'keywords' not in df.columns:
            return pd.DataFrame()
        
        all_keywords = []
        for keywords in df['keywords']:
            if isinstance(keywords, list):
                all_keywords.extend(keywords)
            elif isinstance(keywords, str) and keywords:
                all_keywords.extend([kw.strip() for kw in keywords.split(',')])
        
        if not all_keywords:
            return pd.DataFrame()
        
        # Count keyword frequency
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        
        # Create DataFrame
        keywords_df = pd.DataFrame([
            {'Keyword': keyword, 'Frequency': count, 'Percentage': (count / len(all_keywords)) * 100}
            for keyword, count in keyword_counts.most_common(50)
        ])
        
        return keywords_df
    
    def create_export_report(self, results_data, include_summary=True):
        """Create a comprehensive export report with multiple formats"""
        try:
            exports = {}
            
            # CSV export
            exports['csv'] = self.export_to_csv(results_data)
            
            # JSON export
            exports['json'] = self.export_to_json(results_data)
            
            # Excel export (if openpyxl is available)
            try:
                exports['excel'] = self.export_to_excel(results_data)
            except ImportError:
                exports['excel'] = None
            
            if include_summary:
                # Create summary report
                if isinstance(results_data, list):
                    df = pd.DataFrame(results_data)
                else:
                    df = results_data
                
                summary = self._calculate_summary_stats(df.to_dict('records') if isinstance(df, pd.DataFrame) else results_data)
                exports['summary'] = summary
            
            return exports
            
        except Exception as e:
            raise Exception(f"Export report creation failed: {str(e)}")
