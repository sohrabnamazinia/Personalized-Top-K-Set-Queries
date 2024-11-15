import pandas as pd

input_file = 'dataset_hotels/hotels_small.csv'
output_file = 'dataset_hotels/hotels.csv'

chunk_size = 30_000

try:
    df = pd.read_csv(input_file, nrows=chunk_size, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(input_file, nrows=chunk_size, encoding='ISO-8859-1')


df.to_csv(output_file, index=False)

print(f"The first 100,000 rows have been saved to {output_file}")
