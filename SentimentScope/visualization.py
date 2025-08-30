import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime, timedelta

class SentimentVisualizer:
    def __init__(self):
        self.color_scheme = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#4682B4'    # Steel Blue
        }
    
    def create_sentiment_distribution(self, df):
        """Create a pie chart showing sentiment distribution"""
        if df.empty:
            return self._create_empty_chart("No data available")
        
        sentiment_counts = df['sentiment'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,
            marker_colors=[self.color_scheme.get(sentiment, '#999999') for sentiment in sentiment_counts.index],
            textinfo='label+percent+value',
            textfont_size=12,
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Sentiment Distribution",
            title_x=0.5,
            font_size=12,
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_confidence_distribution(self, df):
        """Create a histogram showing confidence score distribution"""
        if df.empty:
            return self._create_empty_chart("No data available")
        
        fig = px.histogram(
            df, 
            x='confidence',
            color='sentiment',
            nbins=20,
            title="Confidence Score Distribution",
            color_discrete_map=self.color_scheme
        )
        
        fig.update_layout(
            xaxis_title="Confidence Score",
            yaxis_title="Count",
            title_x=0.5,
            height=400,
            bargap=0.1
        )
        
        # Add average confidence line
        avg_confidence = df['confidence'].mean()
        fig.add_vline(
            x=avg_confidence,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {avg_confidence:.2f}"
        )
        
        return fig
    
    def create_sentiment_timeline(self, df):
        """Create a timeline showing sentiment trends over time"""
        if df.empty or 'timestamp' not in df.columns:
            return self._create_empty_chart("No temporal data available")
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by time periods and sentiment
        df['date'] = df['timestamp'].dt.date
        timeline_data = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
        
        fig = px.line(
            timeline_data,
            x='date',
            y='count',
            color='sentiment',
            title="Sentiment Trends Over Time",
            color_discrete_map=self.color_scheme,
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Analyses",
            title_x=0.5,
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_source_comparison(self, df):
        """Create a bar chart comparing sentiment across different sources"""
        if df.empty or 'source' not in df.columns:
            return self._create_empty_chart("No source data available")
        
        # Create crosstab of source and sentiment
        source_sentiment = pd.crosstab(df['source'], df['sentiment'])
        
        fig = go.Figure()
        
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in source_sentiment.columns:
                fig.add_trace(go.Bar(
                    name=sentiment.title(),
                    x=source_sentiment.index,
                    y=source_sentiment[sentiment],
                    marker_color=self.color_scheme.get(sentiment, '#999999'),
                    hovertemplate=f'<b>{sentiment.title()}</b><br>Source: %{{x}}<br>Count: %{{y}}<extra></extra>'
                ))
        
        fig.update_layout(
            title="Sentiment Distribution by Source",
            title_x=0.5,
            xaxis_title="Source",
            yaxis_title="Count",
            barmode='stack',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_keyword_analysis(self, df):
        """Create a bar chart of most common keywords"""
        if df.empty or 'keywords' not in df.columns:
            return self._create_empty_chart("No keyword data available")
        
        # Extract all keywords
        all_keywords = []
        for keywords in df['keywords']:
            if isinstance(keywords, list):
                all_keywords.extend(keywords)
            elif isinstance(keywords, str) and keywords:
                all_keywords.extend(keywords.split(', '))
        
        if not all_keywords:
            return self._create_empty_chart("No keywords extracted")
        
        # Count keyword frequency
        keyword_counts = Counter(all_keywords)
        top_keywords = dict(keyword_counts.most_common(15))
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(top_keywords.values()),
                y=list(top_keywords.keys()),
                orientation='h',
                marker_color='steelblue',
                hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Most Frequent Keywords",
            title_x=0.5,
            xaxis_title="Frequency",
            yaxis_title="Keywords",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_confidence_vs_sentiment(self, df):
        """Create a scatter plot of confidence vs sentiment scores"""
        if df.empty:
            return self._create_empty_chart("No data available")
        
        # Check if detailed scores are available
        if 'detailed_scores' not in df.columns:
            return self._create_empty_chart("No detailed score data available")
        
        # Extract detailed scores
        plot_data = []
        for idx, row in df.iterrows():
            scores = row['detailed_scores']
            if isinstance(scores, dict):
                for sentiment, score in scores.items():
                    plot_data.append({
                        'confidence': row['confidence'],
                        'sentiment_score': score,
                        'sentiment_type': sentiment,
                        'overall_sentiment': row['sentiment']
                    })
        
        if not plot_data:
            return self._create_empty_chart("No valid score data found")
        
        plot_df = pd.DataFrame(plot_data)
        
        fig = px.scatter(
            plot_df,
            x='confidence',
            y='sentiment_score',
            color='sentiment_type',
            title="Confidence vs Sentiment Scores",
            color_discrete_map=self.color_scheme,
            hover_data=['overall_sentiment']
        )
        
        fig.update_layout(
            xaxis_title="Overall Confidence",
            yaxis_title="Sentiment Score",
            title_x=0.5,
            height=400
        )
        
        return fig
    
    def create_text_length_analysis(self, df):
        """Create visualization showing relationship between text length and sentiment"""
        if df.empty or 'text_length' not in df.columns:
            return self._create_empty_chart("No text length data available")
        
        fig = px.box(
            df,
            x='sentiment',
            y='text_length',
            color='sentiment',
            title="Text Length Distribution by Sentiment",
            color_discrete_map=self.color_scheme
        )
        
        fig.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="Text Length (characters)",
            title_x=0.5,
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_comparative_analysis(self, df1, df2, label1="Dataset 1", label2="Dataset 2"):
        """Create comparative analysis between two datasets"""
        if df1.empty and df2.empty:
            return self._create_empty_chart("No data available for comparison")
        
        # Calculate sentiment distributions
        dist1 = df1['sentiment'].value_counts() if not df1.empty else pd.Series(dtype='int64')
        dist2 = df2['sentiment'].value_counts() if not df2.empty else pd.Series(dtype='int64')
        
        # Combine data for plotting
        comparison_data = []
        for sentiment in ['positive', 'negative', 'neutral']:
            comparison_data.append({
                'sentiment': sentiment,
                'dataset': label1,
                'count': dist1.get(sentiment, 0),
                'percentage': (dist1.get(sentiment, 0) / len(df1) * 100) if not df1.empty and len(df1) > 0 else 0.0
            })
            comparison_data.append({
                'sentiment': sentiment,
                'dataset': label2,
                'count': dist2.get(sentiment, 0),
                'percentage': (dist2.get(sentiment, 0) / len(df2) * 100) if not df2.empty and len(df2) > 0 else 0.0
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Count Comparison', 'Percentage Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Count comparison
        for dataset in [label1, label2]:
            data = comparison_df[comparison_df['dataset'] == dataset]
            fig.add_trace(
                go.Bar(
                    name=f"{dataset} (Count)",
                    x=data['sentiment'],
                    y=data['count'],
                    legendgroup=dataset,
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Percentage comparison
        for dataset in [label1, label2]:
            data = comparison_df[comparison_df['dataset'] == dataset]
            fig.add_trace(
                go.Bar(
                    name=f"{dataset} (Percent)",
                    x=data['sentiment'],
                    y=data['percentage'],
                    legendgroup=dataset,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title="Sentiment Distribution Comparison",
            title_x=0.5,
            height=400,
            barmode='group'
        )
        
        fig.update_xaxes(title_text="Sentiment", row=1, col=1)
        fig.update_xaxes(title_text="Sentiment", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Percentage", row=1, col=2)
        
        return fig
    
    def create_dashboard_summary(self, df):
        """Create a comprehensive dashboard with multiple visualizations"""
        if df.empty:
            return self._create_empty_chart("No data available")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sentiment Distribution',
                'Confidence Distribution',
                'Sentiment Over Time',
                'Text Length vs Sentiment'
            ),
            specs=[
                [{"type": "pie"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "box"}]
            ]
        )
        
        # Sentiment distribution (pie chart)
        sentiment_counts = df['sentiment'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                name="Sentiment",
                marker_colors=[self.color_scheme.get(s, '#999') for s in sentiment_counts.index]
            ),
            row=1, col=1
        )
        
        # Confidence histogram
        fig.add_trace(
            go.Histogram(
                x=df['confidence'],
                name="Confidence",
                marker_color='steelblue',
                nbinsx=20
            ),
            row=1, col=2
        )
        
        # Sentiment timeline (if timestamp available)
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            timeline_data = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
            
            for sentiment in df['sentiment'].unique():
                sentiment_data = timeline_data[timeline_data['sentiment'] == sentiment]
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_data['date'],
                        y=sentiment_data['count'],
                        mode='lines+markers',
                        name=f"{sentiment.title()} Timeline",
                        line_color=self.color_scheme.get(sentiment, '#999')
                    ),
                    row=2, col=1
                )
        
        # Text length box plot
        if 'text_length' in df.columns:
            for sentiment in df['sentiment'].unique():
                sentiment_data = df[df['sentiment'] == sentiment]
                fig.add_trace(
                    go.Box(
                        y=sentiment_data['text_length'],
                        name=f"{sentiment.title()} Length",
                        marker_color=self.color_scheme.get(sentiment, '#999')
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="Sentiment Analysis Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _create_empty_chart(self, message="No data available"):
        """Create an empty chart with a message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color="gray"),
            showarrow=False
        )
        
        fig.update_layout(
            title="Visualization",
            title_x=0.5,
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        
        return fig
