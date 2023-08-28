
import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os
import tiktoken
import numpy as np
from collections import defaultdict
from streamlit_extras.buy_me_a_coffee import button

# Initialize an empty list in memory to store the data
if 'jsonl_data_list' not in st.session_state:
    st.session_state.jsonl_data_list = []
# Check for .env and load if present


st.header('Fine-tune OpenAI')
st.text('''This application helps you to prepare , validate and test the responses from OpenAI.''')

# Toggle visibility of the Help section using session state
if 'show_help' not in st.session_state:
    st.session_state.show_help = False

if st.button('Help'):
    st.session_state.show_help = not st.session_state.show_help

if st.session_state.show_help:
    st.write(
        """
        To train the model please use 10 to 100 examples or more.  
        1. **User Prompts**: Enter a question under the label "Enter your question? Human:".
        2. **AI Response**: Provide your ideal AI-generated response.
        3. **Custom System Message**: Add a custom system message or stick with the default message "You are a helpful and friendly assistant.".
        4. **Data Saving**: Upon pressing the "Accept Inputs" button, the provided data gets formatted and appended to an `output.jsonl` file.
        5. **TRAINING_FILE_ID Input**: Users can input their TRAINING_FILE_ID required for fine-tuning. You will receive an email from OpenAI when the model has been trained.
        6. **Send for Fine-Tuning**: A button to send the `output.jsonl` file to OpenAI for fine-tuning.
        7. **Chat Window**: Test the fine-tuned model by sending messages and viewing the model's response.
          
          
        ### Validate your data -  Details from OpenAI
        - **Data Inspection**: The script initially loads the dataset from `output.jsonl` and prints the number of examples and the first example to provide an overview.
        - **Format Error Checks**: The script checks for various formatting issues such as:
        - Incorrect data types
        - Missing message lists
        - Unrecognized message keys
        - Missing content
        - Unrecognized roles in messages
        - Absence of an assistant's message
        - **Token Count**: It calculates the number of tokens for each message and provides distribution statistics such as:
        - Range (Min and Max)
        - Average (Mean)
        - Middle Value (Median)
        - 5th Percentile
        - 95th Percentile

        ##  Understanding OpenAI's Statistics  

        - **Number of Messages per Example Distribution**: Provides statistics about the number of messages in each example.
        - **Total Tokens per Example Distribution**: Indicates the total number of tokens in each example.
        - **Assistant Tokens per Example Distribution**: Pertains to the number of tokens in the assistant's messages within each example.
           For each distribution, the following statistics are provided:
        - **Range**: The smallest and largest values.
        - **Average (Mean)**: The average value.
        - **Middle Value (Median)**: The middle value when sorted.
        - **5th Percentile**: 5% of the data lies below this value.
        - **95th Percentile**: 95% of the data lies below this value.
        """
    )


# Get prompts from the user
system_message_default = 'You are a helpful and friendly assistant.'
system_message = st.text_area('Enter your custom system message:', value=system_message_default)
prompt_text = st.text_area('Enter your question? Human:', height=200)
ideal_generated_text = st.text_area('Enter your ideal AI generated response:', height=200)

# Format the data
data = {
    "messages": [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": ideal_generated_text}
    ]
}

if st.button('Append Data'):
    if system_message and prompt_text and ideal_generated_text:
        # Append the data to the in-memory list
        st.session_state.jsonl_data_list.append(data)
        st.success('Data has been appended!')
        # Display the current contents of the in-memory list for verification
        st.write("Current Data:", st.session_state.jsonl_data_list)
    else:
        st.warning('Please ensure all fields are filled before appending.')


if st.button('Validate your data', key="check_data_btn_2"):
        
    # Load dataset

    dataset = st.session_state.jsonl_data_list

    # We can inspect the data quickly by checking the number of examples and the first item

    # Initial dataset stats
    st.write("Num examples:", len(dataset))
    st.write("First example:")

    for message in dataset[0]["messages"]:
        st.write(message)

    # Now that we have a sense of the data, we need to go through all the different examples and check to make sure the formatting is correct and matches the Chat completions message structure

    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            if not content or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        st.write("Found errors:")
        for k, v in format_errors.items():
            st.write(f"{k}: {v}")
    else:
        st.write("No errors found")

    # Beyond the structure of the message, we also need to ensure that the length does not exceed the 4096 token limit.

    # Token counting functions
    encoding = tiktoken.get_encoding("cl100k_base")

    # not exact!
    # simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(values, name):
        st.write(f"\n#### Distribution of {name}:")
        st.write(f"min / max: {min(values)}, {max(values)}")
        st.write(f"mean / median: {np.mean(values)}, {np.median(values)}")
        st.write(f"percentage 5 % / percentage 95 %: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

    # Last, we can look at the results of the different formatting operations before proceeding with creating a fine-tuning job:

    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
    st.write("Num examples missing system message:", n_missing_system)
    st.write("Num examples missing user message:", n_missing_user)
    st.write("num_messages_per_example =", n_messages )


    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
             
    n_too_long = sum(l > 4096 for l in convo_lens)

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096

    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    TARGET_EPOCHS = 3
    MIN_EPOCHS = 1
    MAX_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    
    st.write(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    st.write(f"By default, you'll train for {n_epochs} epochs on this dataset")
    st.write(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
    st.write("See pricing page to estimate total costs")

# Provide a button to download the data as a JSONL file
if st.download_button('Download JSONL File', data="\n".join([json.dumps(item) for item in st.session_state.jsonl_data_list]), file_name='output.jsonl', mime='text/plain'):
    st.write('Download initiated!')


def mock_response_201():
    response = requests.Response()
    response.status_code = 201
    response._content = b'{"message": "Successfully created."}'  # Sample JSON response content
    return response


def upload_dataset_to_openai(api_key):
    # Convert the dataset in memory to .jsonl format
    jsonl_data = "\n".join([json.dumps(item) for item in st.session_state.jsonl_data_list])
    
    # Define the URL and headers
    url = "https://api.openai.com/v1/files"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/jsonl"
    }
    
    # Since we're sending content directly, we'll use the 'data' parameter of requests.post 
    # and not the 'json' or 'files' parameter.
    response = requests.post(url, headers=headers, data=jsonl_data)
    
    return response
# Streamlit code to capture and store openai_api_key and org_id in session state

# If OPENAI_API_KEY is not already in session state, initialize it as None
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state.OPENAI_API_KEY = None

# If org_id is not already in session state, initialize it as None
if 'org_id' not in st.session_state:
    st.session_state.org_id = None

# Create input boxes for openai_api_key and org_id
openai_api_key_input = st.text_input("Enter your OpenAI API Key:", value=st.session_state.OPENAI_API_KEY)
org_id_input = st.text_input("Enter your Org ID:")

# Update session state with input values
st.session_state.OPENAI_API_KEY = openai_api_key_input
st.session_state.org_id = org_id_input

# Display the stored values (optional, just for demonstration)
st.write("Stored OpenAI API Key:", st.session_state.OPENAI_API_KEY)
st.write("Stored Org ID:", st.session_state.org_id)

if st.button('Upload to OpenAI'):
    #uncomment for production 
    response = upload_dataset_to_openai(OPENAI_API_KEY)
    
    # Using the mock response Using only for testing    
    # response = mock_response_201()

    if response.status_code == 201:  # HTTP 201 Created indicates a successful upload
        st.success("Successfully uploaded dataset to OpenAI!")
    else:
        st.error(f"Failed to upload dataset. Response: {response.text}")


# Input for TRAINING_FILE_ID
training_file_id = st.text_input('Enter your TRAINING_FILE_ID:* wait until you get an email from OpenAI with your ID')

if st.button('Send for Fine Tuning'):
    if not training_file_id:
        st.warning("Please enter a TRAINING_FILE_ID before sending for fine tuning.")
    else:
        # Send the output.jsonl for fine tuning
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.session_state.OPENAI_API_KEY}"
        }
        data = {
            "training_file": training_file_id,
            "model": "gpt-3.5-turbo-0613"
        }
        # response = requests.post("https://api.openai.com/v1/fine_tuning/jobs", headers=headers, json=data)
        # st.write(response.json())
        st.write("data sent to OpenAI for training")


# Chat window to test the fine-tuned model
st.subheader("Test Fine-tuned Model")
user_message_chat = st.text_area('User Message:')
if st.button('Get Response', disabled=not training_file_id):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "model": "ft:gpt-3.5-turbo:"f"{st.session_state.org_id}",
        "messages": [
            {"role": "user", "content": user_message_chat},
        ]
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    assistant_message = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    st.text_area('Assistant Response:', assistant_message)

button(username="raybernardv", floating=False, width=221)