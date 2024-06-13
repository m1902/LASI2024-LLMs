import openai
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import textwrap

import utilities
from utilities import get_embedding

# Load environment variables from .env file
load_dotenv()

# change the path to the ForumDataset file
forum_file_path = r'C:\Users\YourUser\Documents\LASI2024\ForumDataset.xlsx'

# custom print method, to wrap text after ~80 characters
def _print(text):
    width = 80
    wrapped_text = textwrap.fill(text, width=width)
    print(wrapped_text)

# test and print results of the get_response method
def test_get_api_response(text):
    response = utilities.get_response(text)
    _print(response)

# Embeddings - examples:
# https://platform.openai.com/docs/guides/embeddings/use-cases

def calculate_similarity(text1, text2):
    # get embeddings
    embedding1 = utilities.get_embedding(text1)
    embedding2 = utilities.get_embedding(text2)

    _print(f"\r\n1. {text1}")
    _print(f"\r\n2. {text2}")

    similarity = cosine_similarity([embedding1], [embedding2])
    _print(f"\r\n\nCosine Similarity: {similarity[0][0]}\n")


def test_similarity():

    # due to the architecture of embedding model, we will not observe negative values
    # https://vaibhavgarg1982.medium.com/why-are-cosine-similarities-of-text-embeddings-almost-always-positive-6bd31eaee4d5

    # example, where the cosine similarity close to 1 (very similar)
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A quick black fox jumps over the lazy dog."

    calculate_similarity(text1, text2)

    # example, where the cosine similarity close to 0 (orthogonal)
    text1 = "He loves going on long walks during the sunny days of summer."
    text2 = "Financial markets fluctuated greatly during the economic downturn."

    calculate_similarity(text1, text2)


def test_embeddings(text):
    # first, generate embedding for text using OpenAI's model
    embedding = get_embedding(text)

    # then, print the text and embedding
    _print(text)
    print(embedding)


def test_post_similarity():
    # Texts to be analyzed
    post1 = '''Hi my name is Patricia and I work in the healthcare industry, looking forward to this course!'''

    post2 = '''Hello ,This is Patrick from Ghana. Currently working with ACDI/VOCA as their M&E 
    Specialist on their USAID/ADVANCE Ghana Program. 
    I hope to be Chief of Party in the near future and I will need to understand my accountant. 
    This course will prepare me for the future.'''

    post3 = '''Hi, Jens here, in our German market, in Berlin, we have this problem, too!'''

    calculate_similarity(post1, post2)
    calculate_similarity(post1, post3)


def compare_posts():
    # Load the Excel file, change the path to the xlsx file
    file_path = forum_file_path
    df = pd.read_excel(file_path)

    # Extract the first column, assume the first column is at index 0
    first_column = df['post']

    # Initialize an empty matrix with the same length as the number of elements in the first column
    matrix_size = len(first_column)
    similarity_matrix = np.zeros((matrix_size, matrix_size))

    # Compute the matrix - here using simple absolute difference as an example
    for i in range(matrix_size):
        embedding1 = utilities.get_embedding(first_column[i])
        for j in range(matrix_size):
            embedding2 = utilities.get_embedding(first_column[j])
            similarity = cosine_similarity([embedding1], [embedding2])
            similarity_matrix[i, j] = similarity[0][0]

    # Convert the numpy array to a DataFrame
    correlation_df = pd.DataFrame(similarity_matrix)

    # Save the DataFrame to an Excel file
    output_path = 'similarity_matrix.xlsx'
    correlation_df.to_excel(output_path, index=False)


def start_a_thread():

    thread = utilities.create_thread('vs_put_vector_store_id_here')
    assistant = utilities.get_assistant('asst_BGlydr3Ay_put_assistant_id_here')

    run = utilities.run_assistant(thread, assistant)

def get_thread_citations():

    # you need to create a thread first and copy the id
    thread = utilities.get_thread('thread_p1SBAA3L_put_thread_id_here')

    thread_messages = utilities.get_thread_messages(thread.id)
    for message in thread_messages:

        message_content = message.content[0].text
        _print(message_content.value)
        names_of_cited_papers = ""
        annotations = message_content.annotations
        citations = []
        # Iterate over the annotations and add footnotes
        for index, annotation in enumerate(annotations):
            # Replace the text with a footnote
            message_content.value = message_content.value.replace(annotation.text, f' [{index}]')
            # Gather citations based on annotation attributes
            if (file_citation := getattr(annotation, 'file_citation', None)):
                cited_file = utilities.get_file(file_citation.file_id)
                # cited_file = client.files.retrieve(file_citation.file_id)
                names_of_cited_papers += "\r\n- " + str(index) + " " + cited_file.filename
                citations.append(f'[{index}] {file_citation.quote} from {cited_file.filename}')
            elif (file_path := getattr(annotation, 'file_path', None)):
                cited_file = utilities.get_file(file_path.file_id)
                # cited_file = client.files.retrieve(file_path.file_id)
                names_of_cited_papers += "\n" + cited_file.filename
                citations.append(f'[{index}] Click <here> to download {cited_file.filename}')
                # Note: File download functionality not implemented above for brevity
        # Add footnotes to the end of the message before displaying to user
        message_content.value += '\n' + '\n'.join(citations)
        print(names_of_cited_papers)
        print("\n\n")


def main():
    # OpenAI API key setup
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Check if API key is loaded properly
    if not openai.api_key:
        raise ValueError("API Key is not set. Please check if your .env file contains the OPENAI_API_KEY environment variable.")

    # generate API response
    test_get_api_response("Why are we here?")

    # generate and print embedding vector
    test_embeddings("What is predictive modelling?")

    # calculate similarity for different text pairs
    test_similarity()

    # similarity of forum posts
    test_post_similarity()

    # post similarity
    compare_posts()

    # get documents used for generating response
    # get_thread_citations()

    print("*** DONE ***")

if __name__ == "__main__":
    main()
