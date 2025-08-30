import os
import json
import time
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import re
from openai import OpenAI
from datetime import datetime, timedelta

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

class SentimentAnalyzer:
    def __init__(self):
        self.openai_client = None
        self.setup_openai()
        self.stop_words = set(stopwords.words('english'))
        
        # Rate limiting for API calls
        self.last_api_call = None
        self.api_call_count = 0
        self.rate_limit_reset = None
        self.min_delay_between_calls = 1.0  # 1 second minimum delay
        self.max_retries = 3
        self.backoff_factor = 2.0
    
    def setup_openai(self):
        """Initialize OpenAI client if API key is available"""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
    
    def analyze_text(self, text, use_advanced=True, extract_keywords=True):
        """
        Analyze sentiment of given text
        
        Args:
            text (str): Text to analyze
            use_advanced (bool): Whether to use OpenAI for advanced analysis
            extract_keywords (bool): Whether to extract keywords
        
        Returns:
            dict: Analysis results
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        result = {
            'text': text,
            'cleaned_text': cleaned_text
        }
        
        if use_advanced and self.openai_client:
            try:
                # Use OpenAI for advanced analysis
                openai_result = self._analyze_with_openai(cleaned_text)
                result.update(openai_result)
            except Exception as e:
                # Fallback to basic analysis if OpenAI fails
                print(f"OpenAI analysis failed, using basic analysis: {e}")
                basic_result = self._analyze_with_textblob(cleaned_text)
                result.update(basic_result)
        else:
            # Use basic TextBlob analysis
            basic_result = self._analyze_with_textblob(cleaned_text)
            result.update(basic_result)
        
        # Extract keywords if requested
        if extract_keywords:
            keywords = self._extract_keywords(cleaned_text)
            result['keywords'] = keywords
        
        return result
    
    def _clean_text(self, text):
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def _rate_limit_check(self):
        """Check and enforce API rate limiting"""
        current_time = time.time()
        
        # Enforce minimum delay between calls
        if self.last_api_call:
            elapsed = current_time - self.last_api_call
            if elapsed < self.min_delay_between_calls:
                sleep_time = self.min_delay_between_calls - elapsed
                time.sleep(sleep_time)
        
        self.last_api_call = time.time()
        self.api_call_count += 1
    
    def _analyze_with_openai(self, text):
        """Analyze sentiment using OpenAI GPT-5 with rate limiting and error handling"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized - API key may be missing")
        
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self._rate_limit_check()
                
                # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
                # do not change this unless explicitly requested by the user
                response = self.openai_client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert sentiment analyst. Analyze the given text and provide:
                            1. Overall sentiment (positive, negative, or neutral)
                            2. Confidence score (0.0 to 1.0)
                            3. Detailed scores for positive, negative, and neutral (each 0.0 to 1.0, should sum to 1.0)
                            4. Brief explanation of the sentiment reasoning
                            5. Key sentiment-driving phrases (up to 5)
                            
                            Respond with JSON in this exact format:
                            {
                                "sentiment": "positive/negative/neutral",
                                "confidence": 0.85,
                                "detailed_scores": {
                                    "positive": 0.7,
                                    "negative": 0.1,
                                    "neutral": 0.2
                                },
                                "explanation": "Brief explanation of why this sentiment was assigned",
                                "sentiment_phrases": ["phrase1", "phrase2", "phrase3"]
                            }"""
                        },
                        {"role": "user", "content": text}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=500,
                    timeout=30.0  # 30 second timeout
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise Exception("Empty response from OpenAI API")
                
                result = json.loads(content)
                
                # Validate and normalize the response
                sentiment = result.get('sentiment', 'neutral').lower()
                if sentiment not in ['positive', 'negative', 'neutral']:
                    sentiment = 'neutral'
                
                confidence = max(0.0, min(1.0, result.get('confidence', 0.5)))
                
                detailed_scores = result.get('detailed_scores', {})
                # Ensure scores sum to 1.0
                total_score = sum(detailed_scores.values())
                if total_score > 0:
                    detailed_scores = {k: v/total_score for k, v in detailed_scores.items()}
                else:
                    detailed_scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                
                return {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'detailed_scores': detailed_scores,
                    'explanation': result.get('explanation', 'AI-based sentiment analysis'),
                    'sentiment_phrases': result.get('sentiment_phrases', []),
                    'analysis_method': 'OpenAI GPT-5'
                }
                
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON response from OpenAI API: {str(e)}"
                if attempt == self.max_retries - 1:
                    raise Exception(error_msg)
                time.sleep(self.backoff_factor ** attempt)
                continue
                
            except Exception as e:
                error_str = str(e)
                
                # Handle specific API errors
                if "429" in error_str or "quota" in error_str.lower():
                    raise Exception(f"API quota exceeded. Please check your OpenAI billing and usage limits. Error: {error_str}")
                elif "401" in error_str or "unauthorized" in error_str.lower():
                    raise Exception(f"API authentication failed. Please check your OpenAI API key. Error: {error_str}")
                elif "timeout" in error_str.lower():
                    if attempt < self.max_retries - 1:
                        time.sleep(self.backoff_factor ** attempt)
                        continue
                    raise Exception(f"API request timed out after {self.max_retries} attempts")
                elif "rate_limit" in error_str.lower():
                    if attempt < self.max_retries - 1:
                        # Exponential backoff for rate limiting
                        sleep_time = (self.backoff_factor ** attempt) * 60  # Minutes
                        time.sleep(min(sleep_time, 300))  # Max 5 minutes
                        continue
                    raise Exception(f"Rate limit exceeded after {self.max_retries} attempts")
                else:
                    if attempt == self.max_retries - 1:
                        raise Exception(f"OpenAI API error after {self.max_retries} attempts: {error_str}")
                    time.sleep(self.backoff_factor ** attempt)
                    continue
        
        raise Exception(f"Failed to get response from OpenAI after {self.max_retries} attempts")
    
    def _analyze_with_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        try:
            polarity = float(blob.sentiment.polarity)
            subjectivity = float(blob.sentiment.subjectivity)
        except (AttributeError, ValueError):
            # Fallback if TextBlob sentiment fails
            polarity = 0.0
            subjectivity = 0.5
        
        # Convert polarity to sentiment categories
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Calculate confidence based on absolute polarity and subjectivity
        confidence = min(1.0, abs(polarity) + (subjectivity * 0.3))
        
        # Create detailed scores
        if sentiment == 'positive':
            detailed_scores = {
                'positive': 0.5 + (polarity * 0.5),
                'negative': max(0, -polarity * 0.3),
                'neutral': 1 - (0.5 + (polarity * 0.5)) - max(0, -polarity * 0.3)
            }
        elif sentiment == 'negative':
            detailed_scores = {
                'negative': 0.5 + (abs(polarity) * 0.5),
                'positive': max(0, polarity * 0.3),
                'neutral': 1 - (0.5 + (abs(polarity) * 0.5)) - max(0, polarity * 0.3)
            }
        else:
            detailed_scores = {
                'neutral': 0.6,
                'positive': 0.2,
                'negative': 0.2
            }
        
        # Normalize scores to sum to 1.0
        total = sum(detailed_scores.values())
        detailed_scores = {k: v/total for k, v in detailed_scores.items()}
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'detailed_scores': detailed_scores,
            'explanation': f'TextBlob analysis: polarity={polarity:.2f}, subjectivity={subjectivity:.2f}',
            'sentiment_phrases': [],
            'analysis_method': 'TextBlob',
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def _extract_keywords(self, text):
        """Extract important keywords and phrases from text"""
        try:
            # Tokenize and tag parts of speech
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # Extract meaningful words (nouns, adjectives, verbs)
            meaningful_words = []
            for word, pos in pos_tags:
                if (pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB')) and \
                   word not in self.stop_words and len(word) > 2 and word.isalpha():
                    meaningful_words.append(word)
            
            # Get most common words
            word_freq = Counter(meaningful_words)
            keywords = [word for word, freq in word_freq.most_common(10)]
            
            # Extract bigrams (two-word phrases)
            bigrams = []
            for i in range(len(tokens) - 1):
                if tokens[i] not in self.stop_words and tokens[i+1] not in self.stop_words:
                    if len(tokens[i]) > 2 and len(tokens[i+1]) > 2:
                        bigram = f"{tokens[i]} {tokens[i+1]}"
                        bigrams.append(bigram)
            
            # Get most common bigrams
            bigram_freq = Counter(bigrams)
            top_bigrams = [bigram for bigram, freq in bigram_freq.most_common(5)]
            
            # Combine keywords and bigrams
            all_keywords = keywords[:5] + top_bigrams[:3]
            
            return all_keywords[:8]  # Return top 8 keywords/phrases
            
        except Exception as e:
            print(f"Keyword extraction failed: {e}")
            return []
    
    def get_analysis_stats(self, results_list):
        """Get summary statistics from multiple analysis results"""
        if not results_list:
            return {}
        
        sentiments = [r['sentiment'] for r in results_list]
        confidences = [r['confidence'] for r in results_list]
        
        sentiment_counts = Counter(sentiments)
        
        return {
            'total_analyses': len(results_list),
            'sentiment_distribution': dict(sentiment_counts),
            'average_confidence': sum(confidences) / len(confidences),
            'confidence_range': {
                'min': min(confidences),
                'max': max(confidences)
            }
        }
