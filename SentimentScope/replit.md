# Sentiment Analysis Dashboard

## Overview

A Streamlit-based web application for analyzing sentiment in text data. The application provides both basic TextBlob sentiment analysis and advanced OpenAI-powered sentiment analysis capabilities. Users can upload CSV files containing text data, analyze sentiment across multiple text columns, visualize results through interactive charts, and export analysis results in various formats.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

**Frontend Architecture**
- Streamlit framework for the web interface
- Plotly for interactive data visualizations
- Session state management for maintaining analysis results and history
- Component-based architecture with cached resource initialization

**Backend Architecture**
- Modular design with separate classes for different functionalities:
  - `SentimentAnalyzer`: Core sentiment analysis using TextBlob and OpenAI API
  - `DataProcessor`: CSV file validation and text preprocessing
  - `SentimentVisualizer`: Interactive chart generation using Plotly
  - `ExportManager`: Data export functionality (CSV/JSON formats)

**Data Processing Pipeline**
- CSV file upload and validation
- Text column identification and preprocessing
- Batch sentiment analysis with progress tracking
- Results aggregation and statistical analysis

**Analysis Engine**
- Dual-mode sentiment analysis:
  - Basic: TextBlob-based polarity and subjectivity scoring
  - Advanced: OpenAI API integration for enhanced accuracy
- Keyword extraction using NLTK
- Text cleaning and preprocessing with regex patterns

**Visualization System**
- Interactive dashboard with multiple chart types:
  - Pie charts for sentiment distribution
  - Histograms for confidence score distribution
  - Time-series analysis capabilities
- Custom color schemes for consistent branding
- Empty state handling for graceful error display

## External Dependencies

**Core Libraries**
- Streamlit: Web application framework
- Pandas: Data manipulation and analysis
- Plotly: Interactive visualization library
- NumPy: Numerical computing

**Natural Language Processing**
- TextBlob: Basic sentiment analysis and text processing
- NLTK: Advanced text processing, tokenization, and POS tagging
- OpenAI API: Enhanced sentiment analysis capabilities

**Data Processing**
- IO libraries for file handling
- JSON for data serialization
- CSV for structured data export
- Regex for text cleaning and preprocessing

**Runtime Requirements**
- NLTK data packages: punkt, stopwords, averaged_perceptron_tagger
- Environment variable: OPENAI_API_KEY (optional, for advanced analysis)