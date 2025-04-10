from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Fetch API token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("Hugging Face API token is missing. Set it in a .env file or use 'setx' command.")



# # Initialize the model with the token
# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     huggingfacehub_api_token=hf_token,
#     # max_length=512,
#     # max_tokens=20
# )
# Initialize the model correctly
model = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)


chat_history = []  # Initialize chat history
while True:
    user_input = input('you:')
    chat_history.append(user_input)  # Initialize chat history for each session
    if user_input.lower() == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)  # Append AI response to chat history
    print('AI:',result.content)
    
print(chat_history)