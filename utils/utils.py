import os
from pathlib import Path
import pickle
from tqdm import tqdm
from datasets import load_dataset
import chromadb
import matplotlib.pyplot as plt


def set_directories():
    curr_dir = Path(os.getcwd())
    
    data_dir = curr_dir / 'data'
    data_pickle_path = data_dir / 'data_set.pkl'

    vectordb_dir = curr_dir / 'vectore_storage'
    chroma_dir = vectordb_dir / 'chroma'
    
    for dir in [data_dir, vectordb_dir, chroma_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    return data_pickle_path, chroma_dir


def load_data(data_pickle_path, dataset="vipulmaheshwari/GTA-Image-Captioning-Dataset"):
    if not os.path.exists(data_pickle_path):
        print(f"Data set hasn't been loaded. Loading from the datasets library and save it as a pickle.")
        data_set = load_dataset(dataset)
        with open(data_pickle_path, 'wb') as outfile:
            pickle.dump(data_set, outfile)
    else:
        print(f"Data set already exists in the local drive. Loading it.")
        with open(data_pickle_path, 'rb') as infile:
            data_set = pickle.load(infile)

    return data_set


def get_embeddings(data, model):
    # Get the id and embedding of each data/image
    ids = []
    embeddings = []
    for id, image in tqdm(zip(list(range(len(data))), data)):
        ids.append("image "+str(id))

        embedding = model.encode(image)
        embeddings.append(embedding.tolist())

    return ids, embeddings


def get_collection(chroma_dir, model, collection_name, data):
    client = chromadb.PersistentClient(path=chroma_dir.__str__())
    collection = client.get_or_create_collection(name=collection_name)

    if collection.count() != len(data):
        print("Adding embeddings to the collection.")
        ids, embeddings = get_embeddings(data, model)
        collection.add(
            ids=ids,
            embeddings=embeddings
        )
    else:
        print("Embeddings are already added to the collection.")

    return collection


def get_result(collection, data_set, query, model, n_results=2):
    # Query the vector store and get results
    results = collection.query(
        query_embeddings=model.encode([query]),
        n_results=2
    )

    # Get the id of the most relevant image
    img_id = int(results['ids'][0][0].split('image ')[-1])
    
    # Get the image and its caption
    image = data_set['train']['image'][img_id]
    text = data_set['train']['text'][img_id]

    return image, text


def show_image(image, text, query):
    plt.ion()
    plt.axis("off")
    plt.imshow(image)
    plt.show()
    print(f"User query: {query}")
    print(f"Original description: {text}")