import logging

logger = logging.getLogger(__name__)


def main():
    ################################################################
    ################################################################
    print("Initializing...")
    logger.info("Initializing...")
    print("-------------------------------------------------------")
    logger.info("-------------------------------------------------------")

    print("Importing functions...")
    logger.info("Importing functions...")
    # Import module, classes, and functions
    from sentence_transformers import SentenceTransformer
    from utils.utils import set_directories, load_data, get_collection, get_result, show_image

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
    ################################################################
    ################################################################
    print("Starting search...")
    logger.info("Starting search...")
    print("-------------------------------------------------------")
    logger.info("-------------------------------------------------------")
    exit = False
    while not exit:
        # Collect user query
        query = input('Type your query, or "exit" if you want to exit: ')

        if query == "exit":
            exit = True
            print("-------------------------------------------------------")
            logger.info("-------------------------------------------------------")
            print("Search exited.")
            logger.info("Search exited.")
        else:
            # Get search result including the original descriptions of the images
            image, text = get_result(collection, data_set, query, model, n_results=2)

            # Display the image, its caption, and user query
            show_image(image, text, query)
    ################################################################
    ################################################################


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise e