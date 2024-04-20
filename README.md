# PDF Analyzer

The PDF Analyzer is a Python application that facilitates the analysis of PDF documents containing coding materials. It utilizes natural language processing (NLP) models to generate questions and answers based on the content of the PDF. This README provides instructions on setting up the application, running it, and contributing to its development.

## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Special Note](#Specialnote)

## Features
- **Upload PDF**: Users can upload a PDF file containing coding materials.
- **Analysis**: The application processes the PDF document and extracts its content.
- **Question Generation**: Questions are generated for True or False, Multiple Choice Questions (MCQs), and one-word answers based on the PDF content.
- **Answer Generation**: Answers for the generated questions are also provided using NLP models.
- **Export Results**: The generated questions and answers are saved to a CSV file for further use.

## Dependencies
The following dependencies are required to run the PDF Analyzer:
- Streamlit
- PyPDF2
- langchain
- OpenAI GPT-3.5

These dependencies are listed in the `requirements.txt` file and can be installed using `pip`.

## Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/chiru-13/PDF-Analyzer.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pdf-analyzer
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Upgrade ur openAI key in the app.py file

## Usage
1. Run the main application file:
   ```bash
   streamlit run  app.py
   ```
2. Access the application through your web browser.
3. Upload a PDF file containing coding materials.
4. Wait for the analysis to complete and the questions and answers to be generated.
5. Download the generated CSV file containing questions and answers.

## Contributing
Contributions to the PDF Analyzer project are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request. You can contribute by fixing bugs, adding new features, improving documentation, or providing feedback.

----
## Special Note
I initially attempted to use code solely for question and answer extraction. Later, I modified it to adhere to the specific format required for the questions. However, due to the unavailability of an API key, I couldn't extract the new output file. Therefore, I'm submitting the old Q&A CSV.
