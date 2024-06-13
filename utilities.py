from openai import OpenAI

def get_response(text):
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{text}"}
        ]
    )

    return completion.choices[0].message.content

def get_embedding(text):
    client = OpenAI()
    text = text.replace("\n", " ")

    return client.embeddings.create(
        input=[text],
        model="text-embedding-3-small").data[0].embedding

def get_assistants():
    client = OpenAI()
    return client.beta.assistants.list()

def get_assistant(id):
    client = OpenAI()
    return client.beta.assistants.retrieve(id)

def get_thread(id):
    client = OpenAI()
    return client.beta.threads.retrieve(id)

def get_thread_messages(thread_id):
    client = OpenAI()
    return client.beta.threads.messages.list(thread_id)

def get_message_attachments(message_id):
    client = OpenAI()

    messages = client.beta.threads.messages.retrieve(message_id)
    return messages

def get_files():
    client = OpenAI()
    return client.files.list()

def get_file(id):
    client = OpenAI()
    return client.files.retrieve(id)

def get_filename(file):
    return file.filename

def create_thread(vector_store_id):
    client = OpenAI()
    thread = client.beta.threads.create(
        messages=[{
            "role": "user",
            "content": "What are learning outcomes in the Decimal Point game?"}
        ],
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store_id]
            }
        }
    )

    return thread


def run_assistant(thread, assistant):
    client = OpenAI()
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    return run
