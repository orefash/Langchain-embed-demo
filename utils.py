from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

from dotenv.main import load_dotenv
import os
# os.environ["OPENAI_API_KEY"] = "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"

load_dotenv()
# OPEN_AI_KEY = os.environ['OPEN_AI_KEY']

def text_extractor(path):
    raw_text = ''
    with open(path, 'rb') as f:
        pdf = PdfReader(f)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                raw_text += text
    return raw_text
    
def text_processor(path):

    raw_text = text_extractor(path)
    if(len(raw_text)!=0):
        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)
        # print("len: "+str(len(texts)))
        # print(texts[0])
        return texts
    else:
        return
    
def embed_OPENAI():
    path = os.environ["FILEPATH"]
    # print('pth: ', path)

    texts = text_processor(path)
    if texts is not None:
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        docsearch.save_local("faiss_index")

def query_data(query):
    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI
    embeddings = OpenAIEmbeddings()

    try:
        ds = FAISS.load_local("faiss_index", embeddings)
    except RuntimeError:
        print("Issue exists in local save")
    except Exception as e:
        print("The fetch query error is: ",e)
    else:
        # print(f'ds - {ds}')

        docs = ds.similarity_search(query)
        # print(f'sch - {docs}')

        chain = load_qa_chain(OpenAI(temperature=0.7), chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        # print("res: "+response)
        return response





# embed_OPENAI()

# response = query_data("What branches are in Ibadan?")
# print(response)

