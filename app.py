from utils.utils import get_logger, initialization, get_result
import gradio as gr
import logging


logger = get_logger()
collection = None


def main(query):
    logger = logging.getLogger(__name__)
    print("Starting search...")
    logger.info("Starting search...")
    print("-------------------------------------------------------")
    logger.info("-------------------------------------------------------")
    exit = False
    while not exit:
        # Collect user query
        # query = input('Type your query, or "exit" if you want to exit: ')

        if query == "exit":
            exit = True
            print("-------------------------------------------------------")
            logger.info("-------------------------------------------------------")
            print("Search terminated.")
            logger.info("Search terminated.")
            return None, "Search terminated."
        else:
            # Get search result including the original descriptions of the images
            image, text = get_result(collection, data_set, query, model, n_results=2)

            # Display the image, its caption, and user query
            # show_image(image, text, query)
            return image, text


if __name__ == "__main__":
    try:
        if collection == None:
            collection, data_set, model, logger = initialization(logger)
        # main()
        app = gr.Interface(
            fn=main,
            inputs=[gr.Textbox(label="Describe the scene that you are looking for:")],
            outputs=[gr.Image(label="Here's the scene found based on your description:"),
                     gr.Textbox(label="Original description of the found scene:")],
            title="Search for a scene in the world of GTA!"
        )
        app.launch(share=True)
    except Exception as e:
        logger.exception(e)
        raise e