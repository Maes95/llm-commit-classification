import json
import requests

# Read the JSON file from the URL
url = 'https://raw.githubusercontent.com/codeurjc/linux-bugs/refs/heads/main/linux-commits-2023-11-12_random-filtered-1.json'
response = requests.get(url)
JSON_file = response.json()

# Write each entry as a separate line in the output file (JSON Lines format)
with open('1000-linux-commits.jsonl', 'w') as outfile:
    for entry in JSON_file:
        json.dump(entry, outfile)
        outfile.write('\n')