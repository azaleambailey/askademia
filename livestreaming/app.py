from flask import Flask, request, render_template
import multiturn  # Import your multiturn.py logic here

app = Flask(__name__)

# Temporary variable to store the chat history
conversation = []

@app.route("/", methods=["GET", "POST"])
def chat():
    
    global conversation  # Use the global conversation variable

    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            # Add user's message to the conversation
            conversation.append({"user": "You", "text": user_input})
            
            # Get bot response using multiturn logic
            # Replace this with your chatbot response logic
            response = multiturn.chat_response(user_input)
            
            # Add bot's response to the conversation
            conversation.append({"user": "Bot", "text": response})

    return render_template("index.html", conversation=conversation)

if __name__ == "__main__":
    # Clear conversation history on app restart
    conversation = []
    app.run(debug=True)
