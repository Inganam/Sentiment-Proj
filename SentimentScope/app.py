import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import json
import time

from sentiment_analyzer import SentimentAnalyzer
from data_processor import DataProcessor
from visualization import SentimentVisualizer
from export_utils import ExportManager

# Page configuration with accessibility improvements
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Sentiment Analysis Dashboard\nAdvanced AI-powered sentiment analysis with professional NLP integration."
    }
)

# Initialize components
@st.cache_resource
def initialize_analyzer():
    return SentimentAnalyzer()

@st.cache_resource
def initialize_data_processor():
    return DataProcessor()

@st.cache_resource
def initialize_visualizer():
    return SentimentVisualizer()

@st.cache_resource
def initialize_export_manager():
    return ExportManager()

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def main():
    analyzer = initialize_analyzer()
    data_processor = initialize_data_processor()
    visualizer = initialize_visualizer()
    export_manager = initialize_export_manager()

    # Header
    st.title("📊 Interactive Sentiment Analysis Dashboard")
    st.markdown("Analyze emotional tone in text data with advanced NLP and visualization capabilities")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API status indicator
        analyzer_status = "🟢 OpenAI Connected" if analyzer.openai_client else "🟡 Basic Mode Only"
        st.info(f"**Status:** {analyzer_status}")
        
        # Analysis settings
        st.subheader("Analysis Settings")
        use_advanced_analysis = st.checkbox(
            "Use Advanced AI Analysis (OpenAI)", 
            value=True,
            help="Enable for more accurate sentiment analysis with detailed explanations. Falls back to basic analysis if API is unavailable."
        )
        extract_keywords = st.checkbox(
            "Extract Key Phrases", 
            value=True,
            help="Extract important keywords and phrases that drive sentiment"
        )
        batch_size = st.slider(
            "Batch Processing Size", 
            min_value=1, max_value=50, value=10,
            help="Number of texts to process in each batch. Smaller batches provide more frequent progress updates."
        )
        
        # Export options with help text
        st.subheader("Export Options")
        export_format = st.selectbox(
            "Export Format", 
            ["CSV", "JSON", "Both"],
            help="CSV: Spreadsheet format, JSON: Structured data with metadata"
        )
        
        # Help section
        with st.expander("❔ Need Help?"):
            st.markdown("""
            **Getting Started:**
            1. 📝 Enter text or upload a file in the "Text Analysis" tab
            2. 🚀 For multiple texts, use "Batch Processing" with a CSV file
            3. 📈 View charts and trends in "Visualizations"
            4. 💾 Export your results in "Results & Export"
            
            **Tips for Best Results:**
            - Use clear, complete sentences
            - For CSV uploads, ensure one text per row
            - Advanced AI analysis provides more detailed insights
            - Basic mode works without internet/API limits
            """)
        
        # Clear history
        if st.button("🗑️ Clear Analysis History"):
            st.session_state.analysis_results = []
            st.session_state.analysis_history = []
            st.rerun()

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Analysis", "📊 Batch Processing", "📈 Visualizations", "📋 Results & Export"])

    with tab1:
        st.header("Single Text Analysis")
        
        # Text input method selection
        input_method = st.radio("Choose input method:", ["Direct Text Entry", "Text File Upload"])
        
        text_to_analyze = ""
        
        if input_method == "Direct Text Entry":
            text_to_analyze = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Type or paste your text here..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file",
                type=['txt'],
                help="Upload a .txt file containing the text to analyze"
            )
            if uploaded_file:
                text_to_analyze = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", value=text_to_analyze, height=150, disabled=True)
        
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if text_to_analyze.strip():
                with st.spinner("Analyzing sentiment..."):
                    try:
                        # Perform analysis with progress feedback
                        status_placeholder = st.empty()
                        status_placeholder.info("🔄 Processing text...")
                        
                        result = analyzer.analyze_text(
                            text_to_analyze,
                            use_advanced=use_advanced_analysis,
                            extract_keywords=extract_keywords
                        )
                        
                        # Add timestamp and source
                        result['timestamp'] = datetime.now()
                        result['source'] = 'Direct Input'
                        result['text_preview'] = text_to_analyze[:100] + "..." if len(text_to_analyze) > 100 else text_to_analyze
                        
                        # Store results
                        st.session_state.analysis_results.append(result)
                        st.session_state.analysis_history.append(result)
                        
                        status_placeholder.success("✅ Analysis completed successfully!")
                        time.sleep(1)  # Brief success message display
                        status_placeholder.empty()
                        
                        # Display results
                        display_single_result(result)
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "quota" in error_msg.lower():
                            st.error("🚫 **OpenAI API Limit Reached**\n\nYour OpenAI API quota has been exceeded. The system will use basic sentiment analysis instead. \n\n💡 **Solutions:**\n- Check your OpenAI billing and usage limits\n- Wait for your quota to reset\n- Basic analysis will still provide sentiment results")
                            # Try basic analysis as fallback
                            try:
                                result = analyzer.analyze_text(
                                    text_to_analyze,
                                    use_advanced=False,
                                    extract_keywords=extract_keywords
                                )
                                result['timestamp'] = datetime.now()
                                result['source'] = 'Direct Input'
                                result['text_preview'] = text_to_analyze[:100] + "..." if len(text_to_analyze) > 100 else text_to_analyze
                                st.session_state.analysis_results.append(result)
                                st.session_state.analysis_history.append(result)
                                st.info("✅ Completed using basic sentiment analysis")
                                display_single_result(result)
                            except Exception as fallback_error:
                                st.error(f"Both advanced and basic analysis failed: {str(fallback_error)}")
                        elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                            st.error("🔑 **API Authentication Issue**\n\nThere's a problem with your OpenAI API key. Please check your API key configuration.")
                        else:
                            st.error(f"❌ **Analysis Error:** {error_msg}\n\n💡 Try using basic analysis mode or check your internet connection.")
            else:
                st.warning("⚠️ Please enter some text to analyze.")

    with tab2:
        st.header("Batch Processing")
        
        # File upload for batch processing
        batch_file = st.file_uploader(
            "Upload CSV file for batch processing",
            type=['csv'],
            help="Upload a CSV file with a 'text' column containing texts to analyze"
        )
        
        if batch_file:
            try:
                df = pd.read_csv(batch_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head(), width='stretch')
                
                # Column selection
                text_column = st.selectbox(
                    "Select the column containing text to analyze:",
                    df.columns.tolist()
                )
                
                if st.button("🚀 Start Batch Analysis", type="primary"):
                    if text_column in df.columns:
                        # Enhanced batch processing with comprehensive error handling
                        progress_container = st.container()
                        
                        with progress_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            error_summary = st.empty()
                            
                            batch_results = []
                            failed_analyses = []
                            total_rows = len(df)
                            api_errors = 0
                            processing_start = datetime.now()
                            
                            # Process in configurable batches
                            for batch_start in range(0, total_rows, batch_size):
                                batch_end = min(batch_start + batch_size, total_rows)
                                batch_df = df.iloc[batch_start:batch_end]
                                
                                status_text.info(f"📦 Processing batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end} of {total_rows})")
                                
                                for idx, row in batch_df.iterrows():
                                    try:
                                        text = str(row[text_column])
                                        if text.strip() and text.lower() not in ['nan', 'none', '']:
                                            # Attempt analysis with fallback
                                            try:
                                                result = analyzer.analyze_text(
                                                    text,
                                                    use_advanced=use_advanced_analysis,
                                                    extract_keywords=extract_keywords
                                                )
                                            except Exception as api_error:
                                                if "quota" in str(api_error).lower() or "429" in str(api_error):
                                                    api_errors += 1
                                                    # Fall back to basic analysis
                                                    result = analyzer.analyze_text(
                                                        text,
                                                        use_advanced=False,
                                                        extract_keywords=False
                                                    )
                                                    result['fallback_reason'] = 'API quota exceeded'
                                                else:
                                                    raise api_error
                                            
                                            result['timestamp'] = datetime.now()
                                            result['source'] = 'Batch Upload'
                                            result['row_index'] = idx
                                            result['text_preview'] = text[:100] + "..." if len(text) > 100 else text
                                            batch_results.append(result)
                                            
                                            # Preserve original data
                                            for col in df.columns:
                                                if col != text_column:
                                                    result[f'original_{col}'] = row[col]
                                        
                                        # Update progress
                                        current_row = len(batch_results) + len(failed_analyses)
                                        progress = current_row / total_rows
                                        progress_bar.progress(min(progress, 1.0))
                                        
                                        # Estimate remaining time
                                        if current_row > 0:
                                            elapsed = (datetime.now() - processing_start).total_seconds()
                                            rate = current_row / elapsed
                                            remaining = (total_rows - current_row) / rate if rate > 0 else 0
                                            status_text.text(f"⏱️ Processing: {current_row}/{total_rows} | Est. {remaining:.0f}s remaining")
                                        
                                    except Exception as e:
                                        error_info = {
                                            'row_index': idx,
                                            'error': str(e),
                                            'text_preview': str(row[text_column])[:50] + "..." if len(str(row[text_column])) > 50 else str(row[text_column])
                                        }
                                        failed_analyses.append(error_info)
                                
                                # Brief pause between batches to respect rate limits
                                if batch_end < total_rows:
                                    time.sleep(0.5)
                            
                            # Final results
                            st.session_state.analysis_results.extend(batch_results)
                            st.session_state.analysis_history.extend(batch_results)
                            
                            progress_bar.progress(1.0)
                            processing_time = (datetime.now() - processing_start).total_seconds()
                            
                            # Results summary
                            if batch_results:
                                st.success(f"✅ **Batch Processing Complete!**\n\n"
                                         f"📊 **Results:** {len(batch_results)} texts analyzed successfully\n"
                                         f"⏱️ **Time:** {processing_time:.1f} seconds\n"
                                         f"🚀 **Rate:** {len(batch_results)/processing_time:.1f} texts/second")
                                
                                if api_errors > 0:
                                    st.warning(f"⚠️ {api_errors} texts used basic analysis due to API limits")
                                    
                                if failed_analyses:
                                    st.error(f"❌ {len(failed_analyses)} texts failed to analyze")
                                    with st.expander("View failed analyses"):
                                        for failure in failed_analyses[:10]:  # Show first 10
                                            st.write(f"**Row {failure['row_index']}:** {failure['error']}")
                                            st.write(f"Text: {failure['text_preview']}")
                                            st.divider()
                            else:
                                st.error("❌ No texts were successfully analyzed. Please check your data and try again.")
                            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    with tab3:
        st.header("Sentiment Visualizations")
        
        if st.session_state.analysis_results:
            # Create visualizations
            results_df = pd.DataFrame(st.session_state.analysis_results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                fig_dist = visualizer.create_sentiment_distribution(results_df)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Confidence scores
                fig_conf = visualizer.create_confidence_distribution(results_df)
                st.plotly_chart(fig_conf, use_container_width=True)
            
            with col2:
                # Sentiment over time
                if len(results_df) > 1:
                    fig_time = visualizer.create_sentiment_timeline(results_df)
                    st.plotly_chart(fig_time, use_container_width=True)
                
                # Source comparison
                if 'source' in results_df.columns:
                    fig_source = visualizer.create_source_comparison(results_df)
                    st.plotly_chart(fig_source, use_container_width=True)
            
            # Keyword analysis
            if extract_keywords and any(r.get('keywords') for r in st.session_state.analysis_results):
                st.subheader("📝 Keyword Analysis")
                keyword_fig = visualizer.create_keyword_analysis(results_df)
                st.plotly_chart(keyword_fig, use_container_width=True)
                
        else:
            st.info("No analysis results available. Please analyze some text first.")

    with tab4:
        st.header("Results & Export")
        
        if st.session_state.analysis_results:
            # Display results summary
            results_df = pd.DataFrame(st.session_state.analysis_results)
            
            st.subheader("📊 Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyses", len(results_df))
            with col2:
                positive_count = len(results_df[results_df['sentiment'] == 'positive'])
                st.metric("Positive", positive_count)
            with col3:
                negative_count = len(results_df[results_df['sentiment'] == 'negative'])
                st.metric("Negative", negative_count)
            with col4:
                neutral_count = len(results_df[results_df['sentiment'] == 'neutral'])
                st.metric("Neutral", neutral_count)
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            
            # Create display dataframe
            display_df = results_df.copy()
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Select columns to display
            display_columns = ['text_preview', 'sentiment', 'confidence', 'source', 'timestamp']
            if extract_keywords:
                display_columns.insert(-2, 'keywords')
            
            display_df = display_df[display_columns]
            st.dataframe(display_df, width='stretch')
            
            # Enhanced export functionality with better UX
            st.subheader("📥 Export Results")
            
            # Export instructions
            st.info("📊 **Export Guide:** CSV for spreadsheets, JSON for detailed analysis with metadata")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Export CSV", use_container_width=True):
                    try:
                        with st.spinner("Preparing CSV export..."):
                            csv_data = export_manager.export_to_csv(results_df)
                            st.download_button(
                                label="💾 Download CSV File",
                                data=csv_data,
                                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            st.success("✅ CSV ready for download!")
                    except Exception as e:
                        st.error(f"CSV export failed: {str(e)}")
            
            with col2:
                if st.button("📋 Export JSON", use_container_width=True):
                    try:
                        with st.spinner("Preparing JSON export..."):
                            json_data = export_manager.export_to_json(st.session_state.analysis_results)
                            st.download_button(
                                label="💾 Download JSON File",
                                data=json_data,
                                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                            st.success("✅ JSON ready for download!")
                    except Exception as e:
                        st.error(f"❌ JSON export error: {str(e)}")
                        st.info("💡 Try refreshing the page or contact support if this persists")
            
            with col3:
                # Quick export info
                st.markdown("""
                **📊 Export Contains:**
                - All analysis results
                - Confidence scores
                - Timestamps
                - Method details
                - Keywords & phrases
                """)
                    
        else:
            st.info("No results to display. Please analyze some text first.")

def display_single_result(result):
    """Display results for a single text analysis with enhanced accessibility"""
    st.subheader("📊 Analysis Results")
    
    # Main sentiment display with accessibility features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment = result['sentiment']
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        color_map = {"positive": "#2E8B57", "negative": "#DC143C", "neutral": "#4682B4"}
        
        st.markdown(f"<div style='text-align: center; padding: 20px; background-color: {color_map.get(sentiment, '#E0E0E0')}20; border-radius: 10px; border-left: 5px solid {color_map.get(sentiment, '#E0E0E0')};'>"
                   f"<h3 style='color: {color_map.get(sentiment, '#333333')};'>{emoji.get(sentiment, '❓')} {sentiment.title()}</h3>"
                   f"<p style='color: #666666; margin: 0;'>Overall Sentiment</p></div>", unsafe_allow_html=True)
    
    with col2:
        confidence = result['confidence']
        confidence_color = "#28a745" if confidence > 0.8 else "#ffc107" if confidence > 0.6 else "#dc3545"
        confidence_label = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
        
        st.markdown(f"<div style='text-align: center; padding: 20px; background-color: {confidence_color}20; border-radius: 10px; border-left: 5px solid {confidence_color};'>"
                   f"<h3 style='color: {confidence_color};'>{confidence:.1%}</h3>"
                   f"<p style='color: #666666; margin: 0;'>Confidence ({confidence_label})</p></div>", unsafe_allow_html=True)
    
    with col3:
        method = result.get('analysis_method', 'Unknown')
        method_emoji = "🤖" if "OpenAI" in method else "📊"
        method_color = "#7B68EE" if "OpenAI" in method else "#4682B4"
        
        st.markdown(f"<div style='text-align: center; padding: 20px; background-color: {method_color}20; border-radius: 10px; border-left: 5px solid {method_color};'>"
                   f"<h3 style='color: {method_color};'>{method_emoji}</h3>"
                   f"<p style='color: #666666; margin: 0; font-size: 14px;'>{method}</p></div>", unsafe_allow_html=True)
    
    # Analysis explanation with improved readability
    if 'explanation' in result and result['explanation']:
        st.markdown("""<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8;'>
                       <h4 style='color: #17a2b8; margin-top: 0;'>💡 Analysis Explanation</h4>
                       <p style='margin-bottom: 0; line-height: 1.6;'>{}</p>
                    </div>""".format(result['explanation']), unsafe_allow_html=True)
    
    # Fallback notice if applicable
    if 'fallback_reason' in result:
        st.warning(f"⚠️ Note: Used basic analysis due to: {result['fallback_reason']}")
    
    # Enhanced keywords display
    if 'keywords' in result and result['keywords']:
        st.markdown("### 🔑 Key Phrases")
        
        # Create keyword tags
        keywords_html = "<div style='margin: 10px 0;'>"
        for keyword in result['keywords'][:8]:  # Limit to 8 keywords for readability
            keywords_html += f"<span style='display: inline-block; background-color: #e3f2fd; color: #1976d2; padding: 5px 10px; margin: 2px; border-radius: 15px; font-size: 14px;'>{keyword}</span> "
        keywords_html += "</div>"
        st.markdown(keywords_html, unsafe_allow_html=True)
    
    # Enhanced sentiment phrases if available
    if 'sentiment_phrases' in result and result['sentiment_phrases']:
        st.markdown("### 🎨 Sentiment-Driving Phrases")
        phrases_html = "<div style='margin: 10px 0;'>"
        phrase_color = color_map.get(sentiment, '#4682B4')
        for phrase in result['sentiment_phrases'][:5]:
            phrases_html += f"<span style='display: inline-block; background-color: {phrase_color}20; color: {phrase_color}; padding: 5px 10px; margin: 2px; border-radius: 15px; font-size: 14px; border: 1px solid {phrase_color}40;'>{phrase}</span> "
        phrases_html += "</div>"
        st.markdown(phrases_html, unsafe_allow_html=True)
    
    # Detailed breakdown with improved visualization
    if 'detailed_scores' in result:
        st.markdown("### 📈 Detailed Sentiment Breakdown")
        
        # Create a more accessible chart
        scores_data = result['detailed_scores']
        categories = list(scores_data.keys())
        values = list(scores_data.values())
        colors = [color_map.get(cat, '#4682B4') for cat in categories]
        
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f'{v:.1%}' for v in values],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Score: %{y:.2%}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Sentiment Score Breakdown",
            xaxis_title="Sentiment Type",
            yaxis_title="Confidence Score",
            yaxis_tickformat='.1%',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
