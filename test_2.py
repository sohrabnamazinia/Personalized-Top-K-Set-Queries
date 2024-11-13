import pandas as pd

# Specify the input and output file paths
input_file = 'dataset_hotels/hotels_small.csv'
output_file = 'dataset_hotels/hotels.csv'

# Read the first 100,000 rows and handle encoding errors
chunk_size = 30_000

try:
    # Attempt to read the CSV file with UTF-8 encoding
    df = pd.read_csv(input_file, nrows=chunk_size, encoding='utf-8')
except UnicodeDecodeError:
    # Retry with ISO-8859-1 (Latin-1) encoding if UTF-8 fails
    df = pd.read_csv(input_file, nrows=chunk_size, encoding='ISO-8859-1')

# Save the extracted rows to a new CSV file
df.to_csv(output_file, index=False)

print(f"The first 100,000 rows have been saved to {output_file}")
