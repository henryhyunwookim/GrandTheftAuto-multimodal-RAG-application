import os
import logging
from datetime import datetime
from pathlib import Path
import pickle
from tqdm import tqdm
from datasets import load_dataset
import chromadb
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv


def set_directories():
    curr_dir = Path(os.getcwd())
    
    data_dir = curr_dir / 'data'
    data_pickle_path = data_dir / 'data_set.pkl'

    vectordb_dir = curr_dir / 'vector_storage'
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


def get_search_result(collection, data_set, query, model, n_results=2):
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
    print(f"Original description: {text}\n")
    

def get_logger():
    log_path = "./log/"
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    cur_date = datetime.utcnow().strftime("%Y%m%d")
    log_filename = f"{log_path}{cur_date}.log"

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    
    logger = logging.getLogger(__name__)
    
    return logger


def get_image_description(image):
    _ = load_dotenv()
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)

    vision_model = genai.GenerativeModel(
        "gemini-pro-vision",
        generation_config={
            "temperature": 0.0
            }
    )
    
    # image = Image.open(image_path)
    
    prompt = f"""
    Describe what you explicitly see in the given image in detail.
    Begin your description with "In this image," or "This image is about," to provide context.
    Your response should be a hard description of the given image without any thoughts or suggestions.
    """

    response = vision_model.generate_content([prompt, image])
    description_by_llm = response.text

    return description_by_llm


def initialization(logger):
    print("Initializing...")
    logger.info("Initializing...")
    print("-------------------------------------------------------")
    logger.info("-------------------------------------------------------")

    print("Set directories...")
    logger.info("Set directories...")
    # Set directories
    data_pickle_path, chroma_dir = set_directories()

    print("Loading data...")
    logger.info("Loading data...")
    # Load dataset
    data_set = load_data(data_pickle_path)

    print("Loading CLIP model...")
    logger.info("Loading CLIP model...")
    # Load CLIP model
    model = SentenceTransformer("sentence-transformers/clip-ViT-L-14")

    print("Getting vector embeddings...")
    logger.info("Getting vector embeddings...")
    # Get vector embeddings
    collection = get_collection(chroma_dir, model, collection_name='image_vectors', data=data_set['train']['image'])

    print("-------------------------------------------------------")
    logger.info("-------------------------------------------------------")
    print("Initialization completed! Ready for search.")
    logger.info("Initialization completed! Ready for search.")

    return collection, data_set, model, logger