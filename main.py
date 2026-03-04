from huggingface_hub import InferenceClient
import os
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import math
import numpy as np
# Takes in a list consisting of file links

def similarity_score(text, prompt, model):
    sentences = [text, prompt]
    embeddings = model.encode(sentences)
    # Use cosine to calculate the similarity score
    score = np.dot(embeddings[0,:], embeddings[1,:]) / (np.linalg.norm(embeddings[0, :]) * np.linalg.norm(embeddings[1, :]))
    return score, text

def rag_retrieval(files_url, batch_size, k_results, prompt):
    # Vector embedding (load the specific embedding model, concatenate both text and prompt and then embed it using the loaded model)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    top_k_chunks = []
    chunks_score = []
    for file_url in files_url:
        # Ensure that the external files are in the same directory of the project to be able to be read
        with open(file_url, 'r') as f:
            file = f.read()
            # Split into words and count each word in the splited list
            splitted_file = file.split()
            word_length = len(splitted_file)
            # If length of file is smaller than batch size than get the whole batch and move on.
            if word_length < batch_size:
                batch_text = file
                # Compute the cosine similarity score
                score, batch_text = similarity_score(batch_text, prompt, model)
                if len(top_k_chunks) < k_results:
                    top_k_chunks.append(batch_text)
                    chunks_score.append(score)
                else:
                    # If minimum score is less than chunk replace the chunk with the minimum score with the new chunk
                    # and its new score
                    if min(chunks_score) < score:
                        min_idx = np.argmin(chunks_score)
                        top_k_chunks[min_idx] = batch_text
                        chunks_score[min_idx] = score

            # Assume that the last batch is not able to be smaller than 20% of the standard batch's size
            if word_length % batch_size < math.floor(batch_size / 5):
                no_partitions = math.floor(word_length / batch_size)
            else:
                no_partitions = math.ceil(word_length / batch_size)

            for i in range(no_partitions):
                if i != no_partitions - 1:
                    batch_text = ' '.join(splitted_file[batch_size*i:batch_size*(i+1)])
                else:
                    batch_text = ' '.join(splitted_file[batch_size*i:word_length])

                # Compute the cosine similarity score
                score, batch_text = similarity_score(batch_text, prompt, model)
                if len(top_k_chunks) < k_results:
                    top_k_chunks.append(batch_text)
                    chunks_score.append(score)
                else:
                    # If minimum score is less than chunk replace the chunk with the minimum score with the new chunk
                    # and its new score
                    if min(chunks_score) < score:
                        min_idx = np.argmin(chunks_score)
                        top_k_chunks[min_idx] = batch_text
                        chunks_score[min_idx] = score

    return '\n'.join(top_k_chunks)

### IMPROVEMENTS ###
# For rag retrieval, the bottleneck is encoding so instead of encoding every batch, create a list of all batches
# from all documents and encode that.
# Note: This may cause memory issues if there are too many documents or if the document is too long so use batches
# e.g. encode every 5 documents or every 75th batch


def diff_systems(client):
    content_list = ["You are a coding tutor", "You are a motivational coach"]
    for i, content in enumerate(content_list):
        messages = [{"role": "system", "content": content},
                    {"role": "user", "content": "What is the main difference between list comprehension and generators?"}]
        response = client.chat_completion(messages=messages)
        print(f"===========MESSAGE {i} ({content})=================")
        reply = response.choices[0].message.content
        print(f"{reply}\n")

def track_token_usage(model, client):
    tokenizer = AutoTokenizer.from_pretrained(model)
    content_list = ["You are a coding tutor but you like giving short and concise answers", "You are a coding tutor but you like explaining concepts in depth"]
    for i, content in enumerate(content_list):
        messages = [{"role": "system", "content": content},
                    {"role": "user", "content": "What is the main difference between list comprehension and generators?"}]
        response = client.chat_completion(messages=messages)
        print(f"===========MESSAGE {i + 1} ({content})=================")
        token_usage = response.usage
        # Split the prompt tokens into system + user tokens
        # Note: tokenizer returns a dictionary with inputs_id being the token_id of each tokenized subword
        system_tokens = len(tokenizer(content).input_ids)
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_tokens = len(tokenizer(msg.get("content")).input_ids)
                break
        print(f"{token_usage}\nSystem Tokens:{system_tokens}\nUser Tokens:{user_tokens}")
        # reply = response.choices[0].message.content
        # print(reply)

    ## NOTE: This count doesn't include hidden tokens such as encoding metadata, msg seperators, special start/end tokens etc..


def chatbot_convo(client, rag_text=""):
    messages = [{"role": "system", "content": "Be nice and provide short but concise answers"}]
    user_input = input("You (Type 'end' to exit): ").strip()
    user_input = user_input + rag_text
    prev_prompt_token = 0
    completion_token = 0
    while user_input.lower() != "end":
        # Account for empty input strings by going back to this if statement if an empty string is sent again (otherwise continue)
        if not user_input:
            user_input = input("Empty Input!\nYou: (Type 'end' to exit): ").strip()
            continue

        messages.append({"role": "user", "content": user_input})
        key_interrupt = False
        try:
            response = client.chat_completion(messages=messages)
            # If user presses Ctrl+C while the chat is generating msg
        except KeyboardInterrupt:
            reply_text = "Keyboard Interrupt, try again or type 'end' to exit."
            # Remove previous message
            messages.pop(-1)
            key_interrupt = True

        # If text generation is not interrupted by user
        if not key_interrupt:
            # If choices is not an empty list (the model returns an output)
            if response.choices:
                reply_text = response.choices[0].message.content
            else:
                reply_text = "No response from API, try again or type 'end' to exit."
                # Remove previous message
                messages.pop(-1)

        print(reply_text)
        prompt_token = response.usage.prompt_tokens
        curr_prompt_token = prompt_token - prev_prompt_token
        completion_token += response.usage.completion_tokens
        print("\n=== Turn Summary ===")
        print(f"Most recent user tokens (+system): {curr_prompt_token}")
        print(f"Most recent model reply tokens: {response.usage.completion_tokens}")
        print(f"Cumulative prompt tokens (system + all users): {prompt_token}")
        print(f"Cumulative model tokens: {completion_token}")
        print(f"Total tokens used so far: {prompt_token + completion_token}")
        print("====================\n")

        messages.append({"role": "assistant", "content": reply_text})
        user_input = input("You: (Type 'end' to exit): ")
        prev_prompt_token = prompt_token
    print("Chat ended. Goodbye!")

if __name__ == "__main__":
    api_token = os.getenv("HF_API_KEY")
    model_name = "openai/gpt-oss-120b"
    client = InferenceClient(model=model_name, api_key=api_token)
    # Note: Rag retrieval is currently manual and is done before the prompt with a selected handful of AI generated
    # business related files. Normally, rag retrieval is done after the prompt is made with vector databases.
    rag_text = rag_retrieval(["txt_data/market_analysis_renewable.txt", "txt_data/employee_productivity_remote.txt", "txt_data/business_proposal_se_asia.txt", "txt_data/financial_overview_q1.txt"], 70, 3, "What are the company's future plans and growth strategies?")
    chatbot_convo(client, rag_text)