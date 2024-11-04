# QuerySage

This project demonstrates how to build a Question-Answering (QA) bot using LangChain, FAISS, and Google Generative AI. The bot is designed to answer questions based on a given context from a CSV file containing FAQs.

## Requirements

- Python 3.7+
- `langchain`
- `langchain_google_genai`
- `faiss-cpu`
- `python-dotenv`
- `streamlit`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Yuval728/QuerySage.git
    cd QuerySage
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file and add your environment variables:
    ```env
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

1. Load environment variables.
2. Initialize the language model and embedding encoder.
3. Create the vector database.
4. Create the QA chain.
5. Run the script.

## Streamlit UI

To create a user interface using Streamlit, follow these steps:

1. Install Streamlit:
    ```bash
    pip install streamlit
    ```

2. Create a `streamlit_app.py` file and add the necessary code.

3. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Google Generative AI](https://cloud.google.com/ai-platform)
