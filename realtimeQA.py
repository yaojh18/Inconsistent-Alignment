import os
import json

# Define the directory containing the files
directory = './realtimeqa_public/past/2023'

# Create a list to hold all QA entries
merged_data = []

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('qa.jsonl'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                # Read each line as a JSON object
                data = json.loads(line.strip())
                merged_data.append(data)

# Define the output file path
output_filepath = './dataset/realtime_qa.jsonl'

# Write the merged data to the output file
with open(output_filepath, 'w', encoding='utf-8') as output_file:
    for entry in merged_data:
        output_file.write(json.dumps(entry) + '\n')