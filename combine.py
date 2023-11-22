import pandas as pd
import pandas as pd

# Specify the filenames
csv_filename = "lamini-instructions-to-french.csv"
tsv_filename = "pairs.tsv"

# Read the CSV file
csv_data = pd.read_csv(csv_filename)


# Specify the filenames
csv_filename = "lamini-instructions-to-french.csv"
tsv_filename = "pairs.tsv"

# Read the CSV file
csv_data = pd.read_csv(csv_filename)

# Read the TSV file
tsv_data = pd.read_csv(tsv_filename, delimiter='\t')

# Combine the dataframes
combined_data = pd.merge(csv_data, tsv_data)

# Write the combined data to a TSV file
combined_data.to_csv("combined.tsv", sep='\t', index=False)
