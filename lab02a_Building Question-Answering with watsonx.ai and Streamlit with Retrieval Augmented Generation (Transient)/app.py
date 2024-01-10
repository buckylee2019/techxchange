import logging
import os
import pickle
import tempfile
import pathlib
import streamlit as st
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.callbacks import StdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader

from langchain import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import  Chroma
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
import pathlib

# Most GENAI logs are at Debug level.
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header("Retrieval Augmented Generation with watsonx.ai 💬")
# chunk_size=1500
# chunk_overlap = 200

load_dotenv()

handler = StdOutCallbackHandler()

api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)

if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }


GEN_API_KEY = os.getenv("GENAI_KEY", None)
INDEX_NAME = os.path.join(pathlib.Path(__file__).parent.resolve(),'VectorDB')


system_prompt = """
You are an AI assistant tasked with providing answers by summarizing related documents. You should follow these rules:
1. Summarize the content from the provided documents, using the following format:
文件標題: Describe the topic of the document.
按步指示: Provide user question-specific instructions or information from the document.
2. If no relevant information is found in the chat history, respond with "I can't answer the question".
By adhering to these rules, you will help users find accurate and valuable information.
"""
# Sidebar contents
with st.sidebar:
    st.title("RAG App")
    st.markdown('''
    ## About
    This app is an LLM-powered RAG built using:
    - [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai/)
    - [HuggingFace](https://huggingface.co/)
    - [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) LLM model
 
    ''')
    st.write('Powered by [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai)')
    # image = Image.open('/Users/buckylee/Documents/github/Incubation_watsonx_Chinese/lab06a_Building Question-Answering with watsonx.ai and Streamlit with Retrieval Augmented Generation (Transient)/watsonxai.jpg')
    # st.image(image, caption='Powered by watsonx.ai')
    max_new_tokens = st.number_input('max_new_tokens',1,1024,value=300)
    min_new_tokens= st.number_input('min_new_tokens',0,value=15)
    client = chromadb.PersistentClient(path=INDEX_NAME)
    collection_name = st.sidebar.selectbox("Select the documents",
        set([cols.name for cols in client.list_collections()]))
    
    repetition_penalty = st.number_input('repetition_penalty',1,2,value=2)
    decoding = st.text_input(
            "Decoding",
            "greedy",
            key="placeholder",
        )
    
uploaded_file = st.file_uploader("Choose a PDF file", accept_multiple_files=False)

@st.cache_data
def read_pdf(uploaded_file,chunk_size = 300 ,chunk_overlap=20):
    
    
    bytes_data = uploaded_file.read()
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
    # Write content to the temporary file
        temp_file.write(bytes_data)
        filepath = temp_file.name
        with st.spinner('Waiting for the file to upload'):
            loader = PyPDFLoader(filepath)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.split_documents(data)
            return docs


def read_push_embeddings(docs,collection_name):
    
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

    if docs != "":
        
        db = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=INDEX_NAME
            )
    else:
        db = Chroma(
                    embedding_function=embeddings,
                    collection_name=collection_name,
                    persist_directory=INDEX_NAME
                
        )
    return db

def get_prompt_template(system_prompt=system_prompt):
    
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    
    instruction = """
    Context: {summaries}
    User: {question}
    Answer the question in Markdown format.
    Markdown:
    """

    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    prompt = PromptTemplate(input_variables=["summaries", "question"], template=prompt_template)
    

    return prompt


def retrieval_qa_pipline(db, llm, system_prompt):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - db (vectorestore): Specifies the preload vector db
    - system_prompt (str): Define from default or from web UI
    - llm (langchain interface): Specifies the llm.


    Returns:
    - RetrievalQAWithSourcesChain: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt obtained from the `get_prompt_template` function, might be used in the QA system.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    retriever = db.as_retriever(search_kwargs={'k': 3})

    # get the prompt template and memory if set by the user.
    prompt = get_prompt_template( system_prompt=system_prompt)

    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
        retriever=retriever,
        return_source_documents=True,  # verbose=True,
        chain_type_kwargs={
            "prompt":prompt,
        },
    )

    return qa


# show user input
if user_question := st.text_input(
    "Ask a question about your Policy Document:"
):  
    docs = ""
    if uploaded_file:
        docs = read_pdf(uploaded_file)
        collection_name = uploaded_file.name.split(".")[-2]
    
    db = read_push_embeddings(docs,collection_name)
    
    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.TEMPERATURE: 0.0,
        # GenParams.TOP_K: 100,
        # GenParams.TOP_P: 1,
        GenParams.REPETITION_PENALTY: 1
    }
    model_llm = Model(model_id=ModelTypes.LLAMA_2_70B_CHAT.value, credentials=creds, params=params, project_id=project_id).to_langchain()
    
    qa_chain = retrieval_qa_pipline(db,model_llm,system_prompt)
    res = qa_chain(user_question,return_only_outputs=True)
    
    st.markdown(res['answer'])
    st.write()
