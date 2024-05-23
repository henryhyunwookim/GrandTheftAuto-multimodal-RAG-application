import os
import pickle
from tqdm import tqdm
from datasets import load_dataset
import chromadb


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