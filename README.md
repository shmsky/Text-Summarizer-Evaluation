# Text Extraction Evaluation Tool

This tool evaluates the performance of a text extraction library by comparing extracted text with reference text. It provides detailed analysis of missing content and generates comprehensive reports.

## Features

- Text similarity analysis
- Key phrase extraction and comparison
- Missing content analysis with severity assessment
- Multiple output formats (Excel, CSV, JSON)
- Detailed logging
- Configurable thresholds and parameters
- Progress tracking
- Summary statistics

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Configuration

The tool uses a YAML configuration file (`config.yaml`) with the following structure:

```yaml
input:
  file: "Test_check.xlsx"
  required_columns:
    - extracted_text
    - reference_text

output:
  file: "evaluation_results.xlsx"
  formats:
    - excel
    - csv
    - json

analysis:
  similarity_threshold: 0.95
  sentence_delimiters: "[.!?]+"
  missing_content_thresholds:
    minor: 50
    moderate: 200
    significant: 200

logging:
  level: "INFO"
  file: "evaluation.log"
```

## Usage

1. Prepare your input Excel file with the required columns:
   - `extracted_text`: Text extracted by the library
   - `reference_text`: Reference/ground truth text

2. Run the evaluation:
```bash
python evaluate_text_extraction.py
```

3. Check the output files:
   - `evaluation_results.xlsx` (or other formats as configured)
   - `evaluation.log` for detailed logs

## Output

The tool generates a detailed report with the following information for each article:
- Article ID
- Reference text
- Extracted text
- Completeness percentage
- Missing content
- Missing key phrases
- Significance assessment
- Severity level
- Missing key phrases count

## Running Tests

To run the test suite:
```bash
python -m unittest test_evaluator.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 