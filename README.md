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
- Fallback text extraction methods
- URL validation and error handling

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Configuration

The tool uses a YAML configuration file (`config.yaml`) with the following structure:

```yaml
input:
  file: "Test_check.xlsx"
  required_columns:
    - URL
    - lib_text

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
   - `URL`: URL of the article
   - `lib_text`: Text extracted by the library

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
- URL
- Reference text (extracted from URL)
- Extracted text (from library)
- Completeness percentage
- Missing content
- Missing key phrases
- Significance assessment
- Severity level
- Detailed analysis
- Recommendations for improvement

## Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```
   Error: spaCy model not found
   Solution: Run `python -m spacy download en_core_web_sm`
   ```

2. **URL access issues**
   ```
   Error: Network error while accessing URL
   Solution: Check URL accessibility and network connection
   ```

3. **Invalid URL format**
   ```
   Error: Invalid URL format
   Solution: Ensure URLs in input file are properly formatted
   ```

4. **Missing columns in input file**
   ```
   Error: Missing required columns
   Solution: Check input file structure matches configuration
   ```

### Logging

The tool generates detailed logs in `evaluation.log`. Check this file for:
- Processing status
- Error messages
- Warning messages
- Summary statistics

## Development

### Running Tests

To run the test suite:
```bash
python -m unittest test_evaluator.py
```

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 