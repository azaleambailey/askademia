import openai
from openai import OpenAI
from dotenv import load_dotenv
import transcribe_index_ask_video


def chatgpt_chatbot(messages, model='gpt-4o'):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        # max_tokens=100,
        # temperature=0.1,
    )
    answer = completion.choices[0].message.content.strip()
    return answer

def chat_response(user_input, model='gpt-4o'):
    """
    Processes user input and returns the chatbot's response.
    
    Args:
        user_input (str): The user's input message.
        model (str): The model to use for the chat (default is 'gpt-4o').
    
    Returns:
        str: The chatbot's response.
    """
    messages = [
        {'role': 'system', 'content': "You are a helpful assistant."},
        {'role': 'user', 'content': user_input}
    ]
    return transcribe_index_ask_video.user_input_thread(user_input) #"you said" + chatgpt_chatbot(messages, model)



"""
# Example usage (can be commented out when integrating with Flask)
if __name__ == "__main__":
    response = chat_response("What is Obama's birthday?")
    print(response)"""