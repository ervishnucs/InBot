import os
import random
from thirdai import licensing, neural_db as ndb
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Activate ThirdAI license
thirdai_license_key = os.getenv("THIRDAI_LICENSE_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

if thirdai_license_key:
    licensing.activate(thirdai_license_key)  # Enter your ThirdAI key here

# Initialize NeuralDB
db = ndb.NeuralDB()

# Function to insert insurance-related documents into NeuralDB
def insert_documents():
    insertable_docs = []
    pdf_files = [
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\cash-back-plan-brochuree.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\gold-brochure (2).pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\guaranteed-monthly-income-plan.pdf.coredownload.inline.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\guaranteed-protection-plus-plan-brochure.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-life-fortune-plus-plan-brochure.pdf.coredownload.inline.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-csc-shubhlabh-plan-brochure - Copy - Copy.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-life-guaranteed-benefit-plan-brochure1.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-life-long-guaranteed-income-plan-brochure.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-life-insurance-khata-plan-brochure - Copy - Copy.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-life-micro-bachat-plan-brochure - Copy - Copy.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-life-plan-brochure.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-life-radiance-smart-investment-plan-brochure - Copy - Copy.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-life-saral-bachat-bima-plan-brochure.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-life-saral-jeevan-bima-brochure.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-life-smart-pay-plan-brochure (1) - Copy - Copy.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-life-smart-pay-plan-brochure.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-mahajeevan-plus-plan-individual-product-brochure.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-pos-cash-back-plan-brochure - Copy - Copy.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\indiafirst-simple-benefit-plan-brochure.pdf"
        r"C:\Users\pmlba\OneDrive\Desktop\YZ\project\BOT\policies\single-premium-brochure - Copy - Copy.pdf"
]


    for file in pdf_files:
        pdf_doc = ndb.PDF(file)
        insertable_docs.append(pdf_doc)

    checkpoint_config = ndb.CheckpointConfig(
        checkpoint_dir="C:/data/sample_checkpoint",  # Specify checkpoint directory
        resume_from_checkpoint=False,
        checkpoint_interval=3,
    )
    
    # Insert and train documents into NeuralDB
    source_ids = db.insert(insertable_docs, train=True, checkpoint_config=checkpoint_config)
    return source_ids

# Function to perform search in NeuralDB
def search_neural_db(query):
    search_results = db.search(query, top_k=2)
    return {result.text: round(random.uniform(0.7, 0.9), 2) for result in search_results}

# Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}  
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results

# Function to generate queries using OpenAI's ChatGPT
def generate_queries_chatgpt(original_query):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": "OUTPUT (4 queries):"}
        ]
    )
    generated_queries = response.choices[0].message["content"].strip().split("\n")
    return generated_queries

# Function to generate answers using OpenAI's ChatGPT
def generate_answers(query, references):
    context = "\n\n".join(references[:3])
    prompt = (
        "You are an AI insurance agent for IndiaFirst Life Insurance. Your role is to assist users with their queries by providing accurate and specific answers from the company's documents. "
        "Find the exact answer to the user's query from the provided documents, focusing strictly on the relevant plan details. "
        "Always mention the insurance plan name in your answer. "
        "Respond concisely in 2-3 lines without hallucination. If an exact answer is found in the text, return only that specific answer. "
        "After providing an answer, ask if the user needs further clarification or suggestions. "
        "Where additional context is required, include only relevant information. "
        f"Question: {query} \nContext: {context}\n"
        "Provide quantitative details where possible. "
        "You are an intelligent insurance agent chatbot designed to provide accurate and helpful information about various insurance policies. "
        "You have access to 29 insurance policy documents and are here to assist users with any queries they may have. Ensure to follow these guidelines:"
    )

    messages = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0
    )
    return response.choices[0].message['content']