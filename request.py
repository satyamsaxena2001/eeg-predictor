import requests
import pandas as pd

url = 'http://localhost:5000/predict'  # Updated endpoint to match your Flask app route
excel_file_path = './data.xlsx'  # Path to your Excel file

# Read the Excel file
data = pd.read_excel(excel_file_path)  # Adjust based on your file format (e.g., CSV, Excel)

# Convert DataFrame to CSV string (assuming your Flask endpoint expects data in a specific format)
data_csv = data.to_csv(index=False)

# Prepare the payload with the Excel data as a file-like object
files = {'file': ('data.xlsx', data_csv)}

# Send POST request with the Excel data
r = requests.post(url, files=files)

# Check the response
if r.status_code == 200:
    print(f"Predictions: {r.json()['predictions']}")
else:
    print(f"Error: {r.json()['error']}")
