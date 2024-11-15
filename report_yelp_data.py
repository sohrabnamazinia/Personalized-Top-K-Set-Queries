import pandas as pd

csv_file_path = 'businesses.csv'

businesses_df = pd.read_csv(csv_file_path, engine="python")

print("Number of non-null values for each column:")
print(businesses_df.count())

print("\nUnique photo labels:")
print(businesses_df['photo_label'].unique())
