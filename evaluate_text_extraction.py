import pandas as pd
from difflib import SequenceMatcher
import re
from typing import Dict, Tuple, List, Optional
import numpy as np
import yaml
import logging
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import spacy
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import time
from urllib.parse import urlparse

@dataclass
class AnalysisConfig:
    similarity_threshold: float
    sentence_delimiters: str
    missing_content_thresholds: Dict[str, int]

@dataclass
class InputConfig:
    file: str
    required_columns: List[str]

@dataclass
class OutputConfig:
    file: str
    formats: List[str]

class TextExtractionEvaluator:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.nlp = spacy.load("en_core_web_sm")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=self.config['logging']['level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def extract_text_from_url(self, url: str) -> Optional[str]:
        """Extract text content from URL using newspaper3k."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            self.logger.error(f"Error extracting text from {url}: {e}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts."""
        return SequenceMatcher(None, self.clean_text(text1), self.clean_text(text2)).ratio()

    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text using spaCy."""
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def analyze_missing_content(self, missing_content: List[str], missing_phrases: List[str]) -> Dict:
        """Enhanced analysis of missing content with detailed assessment."""
        if not missing_content and not missing_phrases:
            return {
                'significance': "No significant content missing",
                'severity': "low",
                'missing_key_phrases_count': 0,
                'analysis': "Текст полностью соответствует эталону"
            }
        
        total_missing_length = sum(len(text) for text in missing_content)
        thresholds = self.config['analysis']['missing_content_thresholds']
        
        severity = "low"
        if total_missing_length >= thresholds['significant']:
            severity = "high"
        elif total_missing_length >= thresholds['moderate']:
            severity = "medium"
        
        # Анализ значимости пропущенного контента
        analysis = []
        if missing_phrases:
            analysis.append(f"Пропущено {len(missing_phrases)} ключевых фраз")
        
        if missing_content:
            analysis.append(f"Пропущено {len(missing_content)} предложений")
            
            # Анализ типов пропущенного контента
            content_types = {
                'intro': 0,
                'details': 0,
                'conclusion': 0
            }
            
            for content in missing_content:
                if len(content) < 50:  # Короткие предложения часто являются вводными
                    content_types['intro'] += 1
                elif len(content) > 100:  # Длинные предложения обычно содержат детали
                    content_types['details'] += 1
                else:
                    content_types['conclusion'] += 1
            
            if content_types['intro'] > 0:
                analysis.append(f"Пропущено {content_types['intro']} вводных предложений")
            if content_types['details'] > 0:
                analysis.append(f"Пропущено {content_types['details']} детальных описаний")
            if content_types['conclusion'] > 0:
                analysis.append(f"Пропущено {content_types['conclusion']} заключительных предложений")
        
        # Рекомендации по улучшению
        recommendations = []
        if severity == "high":
            recommendations.append("Необходимо улучшить алгоритм извлечения текста")
        if missing_phrases:
            recommendations.append("Следует обратить внимание на сохранение ключевых фраз")
        if content_types['details'] > 0:
            recommendations.append("Рекомендуется улучшить извлечение детальных описаний")
            
        return {
            'significance': f"{severity.upper()} severity: {len(missing_content)} sentences and {len(missing_phrases)} key phrases missing",
            'severity': severity,
            'missing_key_phrases_count': len(missing_phrases),
            'analysis': " | ".join(analysis),
            'recommendations': " | ".join(recommendations) if recommendations else "Нет рекомендаций"
        }

    def evaluate_extraction(self, extracted_text: str, reference_text: str) -> Dict:
        """Evaluate the quality of text extraction with enhanced analysis."""
        extracted = self.clean_text(extracted_text)
        reference = self.clean_text(reference_text)
        
        # Calculate similarity
        similarity = self.calculate_similarity(extracted, reference)
        
        # Extract key phrases
        reference_phrases = set(self.extract_key_phrases(reference))
        extracted_phrases = set(self.extract_key_phrases(extracted))
        missing_phrases = reference_phrases - extracted_phrases
        
        # Find missing content
        missing_content = []
        if similarity < self.config['analysis']['similarity_threshold']:
            extracted_sentences = set(re.split(self.config['analysis']['sentence_delimiters'], extracted))
            reference_sentences = set(re.split(self.config['analysis']['sentence_delimiters'], reference))
            missing_sentences = reference_sentences - extracted_sentences
            missing_content = [s.strip() for s in missing_sentences if s.strip()]
        
        return {
            'similarity_score': similarity,
            'missing_content': missing_content,
            'missing_key_phrases': list(missing_phrases),
            'completeness_percentage': similarity * 100
        }

    def export_results(self, results_df: pd.DataFrame):
        """Export results in multiple formats."""
        base_path = Path(self.config['output']['file']).stem
        
        for format_type in self.config['output']['formats']:
            if format_type == 'excel':
                results_df.to_excel(f"{base_path}.xlsx", index=False)
            elif format_type == 'csv':
                results_df.to_csv(f"{base_path}.csv", index=False)
            elif format_type == 'json':
                results_df.to_json(f"{base_path}.json", orient='records', indent=2)

    def run_evaluation(self):
        """Run the complete evaluation process."""
        try:
            self.logger.info("Starting text extraction evaluation")
            
            # Read input file
            df = pd.read_excel(self.config['input']['file'])
            
            # Validate required columns
            missing_columns = set(self.config['input']['required_columns']) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            results = []
            for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
                try:
                    url = row['URL']
                    extracted_text = row['lib_text']
                    
                    # Extract reference text from URL
                    reference_text = self.extract_text_from_url(url)
                    if not reference_text:
                        self.logger.warning(f"Could not extract text from {url}, skipping...")
                        continue
                    
                    # Add delay to avoid overwhelming the server
                    time.sleep(1)
                    
                    # Evaluate extraction
                    evaluation = self.evaluate_extraction(extracted_text, reference_text)
                    
                    # Analyze missing content
                    analysis = self.analyze_missing_content(
                        evaluation['missing_content'],
                        evaluation['missing_key_phrases']
                    )
                    
                    results.append({
                        'article_id': index + 1,
                        'url': url,
                        'reference_text': reference_text,
                        'extracted_text': extracted_text,
                        'completeness_percentage': evaluation['completeness_percentage'],
                        'missing_content': '\n'.join(evaluation['missing_content']),
                        'missing_key_phrases': ', '.join(evaluation['missing_key_phrases']),
                        'significance_of_missing': analysis['significance'],
                        'severity': analysis['severity'],
                        'analysis': analysis['analysis'],
                        'recommendations': analysis['recommendations']
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing article {index + 1}: {e}")
                    continue
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Export results
            self.export_results(results_df)
            
            self.logger.info("Evaluation completed successfully")
            
            # Generate summary statistics
            summary = {
                'total_articles': len(results),
                'average_completeness': results_df['completeness_percentage'].mean(),
                'articles_with_high_severity': len(results_df[results_df['severity'] == 'high']),
                'articles_with_medium_severity': len(results_df[results_df['severity'] == 'medium']),
                'articles_with_low_severity': len(results_df[results_df['severity'] == 'low'])
            }
            
            self.logger.info(f"Summary statistics: {json.dumps(summary, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

if __name__ == "__main__":
    evaluator = TextExtractionEvaluator()
    evaluator.run_evaluation() 