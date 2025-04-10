from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
import warnings

# Optionally, suppress FutureWarnings (like the one from huggingface_hub)
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize HuggingFaceEndpoint with repo_id directly and set parameters properly
llm = HuggingFaceEndpoint(
    repo_id="microsoft/DialoGPT-medium",  # Pass the model as repo_id
    max_new_tokens=70,  # Model-specific parameter
    temperature=0.7     # Model-specific parameter
)

# Create prompt template for chat, including a placeholder for the history
prompt = ChatPromptTemplate.from_template(
    "User: {question}\nAssistant:"
)

# Store the conversation history
conversation_history = ""

# Chat loop
while True:
    user_input = input('you: ')
    if user_input.lower() == 'exit':
        break
    
    # Add the user input to the conversation history
    conversation_history += f"User: {user_input}\n"
    
    # Format the prompt with the conversation history
    prompt_input = conversation_history + "Assistant:"
    
    # Get the model's response by passing the prompt as a plain string
    result = llm.invoke(prompt_input)
    
    # Update the conversation history with the assistant's response
    conversation_history += f"Assistant: {result}\n"
    
    # Print the assistant's response
    print('AI:', result)
