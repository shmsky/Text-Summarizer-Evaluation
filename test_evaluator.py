import unittest
from evaluate_text_extraction import TextExtractionEvaluator
import pandas as pd
import os
import tempfile
import yaml

class TestTextExtractionEvaluator(unittest.TestCase):
    def setUp(self):
        # Create temporary config file
        self.config = {
            'input': {
                'file': 'test_input.xlsx',
                'required_columns': ['extracted_text', 'reference_text']
            },
            'output': {
                'file': 'test_output.xlsx',
                'formats': ['excel']
            },
            'analysis': {
                'similarity_threshold': 0.95,
                'sentence_delimiters': '[.!?]+',
                'missing_content_thresholds': {
                    'minor': 50,
                    'moderate': 200,
                    'significant': 200
                }
            },
            'logging': {
                'level': 'INFO',
                'file': 'test.log'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.config, f)
            self.config_path = f.name

        # Create test data
        self.test_data = pd.DataFrame({
            'extracted_text': [
                'This is a test. It has two sentences.',
                'Missing some content here.',
                'Perfect match with reference.'
            ],
            'reference_text': [
                'This is a test. It has two sentences. And one more.',
                'Missing some content here. This is additional content.',
                'Perfect match with reference.'
            ]
        })
        
        # Save test data
        self.test_data.to_excel('test_input.xlsx', index=False)
        
        self.evaluator = TextExtractionEvaluator(self.config_path)

    def tearDown(self):
        # Clean up temporary files
        os.remove(self.config_path)
        os.remove('test_input.xlsx')
        if os.path.exists('test_output.xlsx'):
            os.remove('test_output.xlsx')
        if os.path.exists('test.log'):
            os.remove('test.log')

    def test_clean_text(self):
        text = "  This is a test.  \n  It has spaces.  "
        cleaned = self.evaluator.clean_text(text)
        self.assertEqual(cleaned, "This is a test. It has spaces.")

    def test_calculate_similarity(self):
        text1 = "This is a test"
        text2 = "This is a test"
        similarity = self.evaluator.calculate_similarity(text1, text2)
        self.assertEqual(similarity, 1.0)

    def test_evaluate_extraction(self):
        extracted = "This is a test. It has two sentences."
        reference = "This is a test. It has two sentences. And one more."
        
        result = self.evaluator.evaluate_extraction(extracted, reference)
        
        self.assertIn('similarity_score', result)
        self.assertIn('missing_content', result)
        self.assertIn('completeness_percentage', result)
        self.assertLess(result['similarity_score'], 1.0)

    def test_analyze_missing_content(self):
        missing_content = ["This is missing content."]
        missing_phrases = ["missing content"]
        
        analysis = self.evaluator.analyze_missing_content(missing_content, missing_phrases)
        
        self.assertIn('significance', analysis)
        self.assertIn('severity', analysis)
        self.assertIn('missing_key_phrases_count', analysis)

    def test_run_evaluation(self):
        self.evaluator.run_evaluation()
        
        # Check if output file was created
        self.assertTrue(os.path.exists('test_output.xlsx'))
        
        # Read results
        results = pd.read_excel('test_output.xlsx')
        
        # Verify results structure
        self.assertIn('completeness_percentage', results.columns)
        self.assertIn('severity', results.columns)
        self.assertIn('missing_key_phrases_count', results.columns)

if __name__ == '__main__':
    unittest.main() 