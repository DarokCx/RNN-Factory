from datasets import load_dataset
import pymongo

# Load the dataset
dataset = load_dataset("victorych22/lamini-embedded-instructions-only")

# Select the MongoDB URI and database name
mongo_uri = "mongodb://localhost:8090/"
database_name = "your_database_name"

# Establish a connection to MongoDB
client = pymongo.MongoClient(mongo_uri)
db = client[database_name]

# Specify the collection name in MongoDB
collection_name = "your_collection_name"
collection = db[collection_name]

# Iterate through the dataset and insert data into MongoDB
for data in dataset['train']:
    instruction = data['instruction']
    response = data['response']

    # Create a document to insert into MongoDB
    document = {
        "instruction": instruction,
        "response": response
    }

    # Insert the document into the collection
    collection.insert_one(document)

# Close the MongoDB client
client.close()
