import streamlit as st
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain, RefineDocumentsChain, RetrievalQA
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
import csv
import os 

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your api key"

def count_pdf_pages(pdf_path):
    """Count the number of pages in a PDF file."""
    try:
        pdf = PdfReader(pdf_path)
        return len(pdf.pages)
    except Exception as e:
        print("Error:", e)
        return None

def process_pdf(file_path):
    """Process the PDF file and split its content into chunks."""
    loader = PyPDFLoader(file_path)
    data = loader.load()

    page_content = ''.join([page.page_content for page in data])

    # Split content into chunks
    splitter = TokenTextSplitter(model_name='gpt-3.5-turbo', chunk_size=10000, chunk_overlap=200)
    chunks = splitter.split_text(page_content)

    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

def generate_questions_and_answers(documents):
    """Generate questions and answers of different types based on the processed documents."""
    prompt_template = """
    Create questions for True or False, Multiple Choice Questions (MCQs), and one-word answers based on the text below:

    {text}

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = """
    Given the following text, refine the original questions:

    {text}

    QUESTIONS:
    """

    REFINE_PROMPT_QUESTIONS = PromptTemplate(input_variables=["text"], template=refine_template)

    llm_ques_gen_pipeline = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")

    ques_gen_chain = load_summarize_chain(llm=llm_ques_gen_pipeline, chain_type="refine", verbose=True, 
                                          question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)

    questions = ques_gen_chain.run(documents)

    # Split questions into different types
    true_false_questions = []
    mcq_questions = []
    one_word_answers = []

    for question in questions.split("\n"):
        if '?' in question:
            # Extracting the text before the question mark
            question_text = question.split('?')[0] + '?'
            # Generating True or False questions
            true_false_questions.append(question_text + " True")
            true_false_questions.append(question_text + " False")
            # Generating Multiple Choice Questions (MCQs)
            mcq_questions.append(question_text + " A. Option A")
            mcq_questions.append(question_text + " B. Option B")
            mcq_questions.append(question_text + " C. Option C")
            mcq_questions.append(question_text + " D. Option D")
            # Generating one-word answers
            one_word_answers.append(question_text)
    
    return true_false_questions, mcq_questions, one_word_answers

def save_questions_and_answers(questions, answer_generation_chain, output_file):
    """Save questions and answers to a CSV file."""
    with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ['Question', 'Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for question in questions:
            answer = answer_generation_chain.run(question)
            writer.writerow({'Question': question, 'Answer': answer})

def main():
    st.title("PDF Analyzer")

    pdf_file = st.file_uploader("Upload PDF", type=['pdf'])
    if pdf_file:
        filename = pdf_file.name
        st.write("Filename:", filename)
        with open(filename, "wb") as f:
            f.write(pdf_file.getbuffer())

        st.write("Analyzing PDF...")
        documents = process_pdf(filename)
        true_false_questions, mcq_questions, one_word_answers = generate_questions_and_answers(documents)

        # Merge all types of questions
        all_questions = true_false_questions + mcq_questions + one_word_answers

        # Initialize answer generation chain
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(documents, embeddings)
        llm_answer_gen = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
        answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff", retriever=vector_store.as_retriever())

        output_file = f"output/{os.path.splitext(filename)[0]}.csv"
        save_questions_and_answers(all_questions, answer_generation_chain, output_file)
        st.write("Analysis complete. Results saved to:", output_file)

if __name__ == "__main__":
    main()
