import requests
import pandas as pd
import time
import json
import sys

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python script_name.py <sheet_name>")
    sys.exit(1)

# Get the sheet name from the command-line argument
sheet_name = sys.argv[1]

# Define the URL of your Flask app
url = "http://127.0.0.1:5000/"

# Load your Excel file
file_path = "live_questions.xlsx"
questions_df = pd.read_excel(file_path, sheet_name=sheet_name)

# Sort questions by timestamp
questions_df = questions_df.sort_values(by="Timestamp")

# List to store responses
responses = []

start_time = time.time()

# Loop through each row in the DataFrame
for _, row in questions_df.iterrows():
    question = row["Question"]
    timestamp = row["Timestamp"]

    # Calculate the wait time
    current_time = time.time()
    wait_time = max(0, timestamp - (current_time - start_time))
    time.sleep(wait_time)

    # Send the POST request with question and timestamp
    payload = {"question": question}
    response = requests.post(url, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        response_json = response.json()
        response_json['Question'] = question
        response_json['Timestamp'] = timestamp
        response_json['TA Response'] = row['TA Response']
        response_json['Lecture'] = row['Lecture']
        responses.append(response_json)  # Store the response in the list
        print(f"Response received: {response_json}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

# Save all responses to a JSON file
output_file = f"responses/lec{sheet_name}.json"
with open(output_file, "w") as f:
    json.dump(responses, f, indent=4)

print(f"All responses have been saved to {output_file}.")