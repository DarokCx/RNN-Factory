from datasets import load_dataset

# def extract_column(dataset_name, split, column_name):
#     # Load the dataset
#     dataset = load_dataset(dataset_name)

#     # Extract the specified column
#     extracted_data = []
#     for row in dataset[split]:
#         value = row[column_name]
#         extracted_data.append(value)

#     return extracted_data
def extract_column(dataset_name, split, column_name):
    dataset = load_dataset(dataset_name)
    extracted_data = []
    for row in dataset[split]:
        value = row[column_name]
        extracted_data.append(value)
    return extracted_data

# Using the function
# dataset_name = 'MBZUAI/LaMini-instruction'
# split = 'train'  # Choose the dataset split, e.g., 'train', 'test', etc.
# column_name = 'response'  # Specify the column you want to extract

# data = extract_column(dataset_name, split, column_name)

# # Example: Print the first 10 entries
# for i in range(10):
#     print(data[i])
