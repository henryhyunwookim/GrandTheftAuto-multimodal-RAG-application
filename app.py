print("Initializing...")

print("Importing functions...")
# Import module, classes, and functions
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils.utils import load_data, get_collection, get_result, show_image

print("Set directories...")
# Set directories
curr_dir = Path(os.getcwd())
data_dir = curr_dir / 'data'
data_pickle_path = data_dir / 'data_set.pkl'
vectordb_dir = curr_dir / 'vectore_storage'
chroma_dir = vectordb_dir / 'chroma'
for dir in [data_dir, vectordb_dir, chroma_dir]:
    if not os.path.exists(dir):
        os.mkdir(dir)

print("Loading data...")
# Load dataset
data_set = load_data(data_pickle_path)

print("Loading CLIP model...")
# Load CLIP model
model = SentenceTransformer("sentence-transformers/clip-ViT-L-14")

print("Getting vector embeddings...")
# Get vector embeddings
collection = get_collection(chroma_dir, model, collection_name='image_vectors', data=data_set['train']['image'])
print("Initialization completed! Ready for search.")