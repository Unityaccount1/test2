import os
import PyPDF2
import streamlit as st
import pytesseract
import cv2
from PIL import Image
from io import StringIO
from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager

st.set_page_config(page_title="Demo", page_icon=':book:')
@st.cache_data
def extract_text_from_image(file_path):
    if os.path.isfile(file_path):
        if file_path.endswith(".jpg"):
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pyteseract.image_to_string(gray)
            return text
@st.cache_data
def load_Documentos(file_path):
    st.info(f"Leyendo documento en la ruta: {file_path}")
    all_text = ""
    if os.path.isfile(file_path):
        if file_path.endswith((".pdf", ".txt")):
            if file_path.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(file_path)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                all_text += text
            elif file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    all_text += text
    return all_text
@st.cache_data
def load_docs(files):
    folder_path="documentos"
    all_text = ""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith((".pdf", ".txt")):
            if filename.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(file_path)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                all_text += text
            elif filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    all_text += text
    st.info("`Leyendo documento ...`")
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Por favor, proporcione un archivo txt o pdf.', icon="⚠️")
    return all_text

@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "BÚSQUEDA DE SIMILITUD":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error al crear el vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)

    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):
    st.info("`Dividiendo documento ...`")
    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Error al dividir el documento")
        st.stop()

    return splits

def main():

    st.write(
        f"""
        <div style="display: flex; align-items: center; margin-left: 0;">
            <h1 style="display: inline-block;">DemoPDF</h1>
            <sup style="margin-left:5px;font-size:small; color: green;">beta v0.4</sup>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Menú")

    folder_path = "documentos"

    # Nueva opción para seleccionar archivos PDF dentro de la carpeta
    selected_file = st.sidebar.selectbox("Selecciona un archivo:", options=os.listdir(folder_path), index=0)

    embedding_option = st.sidebar.radio(
        "Elige Embeddings", ["OpenAI Embeddings"])

    retriever_type = st.sidebar.selectbox(
        "Elige Retriever", ["BÚSQUEDA DE SIMILITUD"])

    temperature = st.sidebar.slider(
        "Temperatura", 0.0, 1.5, 0.8, step=0.1)
    
    chunk_size = st.sidebar.slider(
        "Tamaño de Chunk (chunk_size)", 100, 2000, 1000, step=100)
    
    splitter_type = "RecursiveCharacterTextSplitter"
    
    start_app = st.sidebar.checkbox("Iniciar", value=False)
    load_files_option = st.sidebar.checkbox("Cargar archivos", value=False)
    load_files_image = st.sidebar.checkbox("Cargar imagen", value=False)


    if start_app:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"

    if load_files_image:
        embeddings = HuggingFaceEmbeddings()
        question_answering = pipeline("question-answering")
        file_path = os.path.join(folder_path, selected_file)
        extracted_text = extract_text_from_image(file_path)
        
        #db = FAISS.from_texts(splits, embeddings)
        st.write("Procesando imagen...")
        
        user_question = st.text_input("Ingresa tu pregunta:")
        if user_question:
            answer = question_answering(question=user_question, context=extracted_text)
            st.write("Respuesta:", answer)

    if load_files_option:
        uploaded_files = st.file_uploader("Sube un documento PDF o TXT", type=[
                                      "pdf", "txt"], accept_multiple_files=True)
        if uploaded_files:
            if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
                st.session_state.last_uploaded_files = uploaded_files

            loaded_text = load_docs(uploaded_files)
            st.write("Documentos cargados y procesados.")

            splits = split_texts(loaded_text, chunk_size=chunk_size,
                                overlap=0, split_method=splitter_type)

            num_chunks = len(splits)
            st.write(f"Número de chunks: {num_chunks}")

            if embedding_option == "OpenAI Embeddings":
                embeddings = HuggingFaceEmbeddings()

            retriever = create_retriever(embeddings, splits, retriever_type)

            callback_handler = StreamingStdOutCallbackHandler()
            callback_manager = CallbackManager([callback_handler])

            #chat_openai = ChatOpenAI(
             #   streaming=True, callback_manager=callback_manager, verbose=True, temperature=temperature)
            #qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)
            db = FAISS.from_texts(splits, embeddings)

            user_question = st.text_input("Ingresa tu pregunta:")
            if user_question:
                answer = db.similarity_search(user_question)
                st.write("Respuesta:", answer)
    else:
        file_path = os.path.join(folder_path, selected_file)
        loaded_text = load_Documentos(file_path)
        splits = split_texts(loaded_text, chunk_size=chunk_size,
                             overlap=0, split_method=splitter_type)

        num_chunks = len(splits)
        st.write(f"Número de chunks: {num_chunks}")

        if embedding_option == "OpenAI Embeddings":
            embeddings = HuggingFaceEmbeddings()

        retriever = create_retriever(embeddings, splits, retriever_type)

        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        #chat_openai = ChatOpenAI(
        #    streaming=True, callback_manager=callback_manager, verbose=True, temperature=temperature)
        #qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)
        db = FAISS.from_texts(splits, embeddings)
        

        st.write("Listo para responder preguntas.")

        user_question = st.text_input("Ingresa tu pregunta:")
        if user_question:
            answer = db.similarity_search(user_question)
            #answer = qa.run(user_question)
            st.write("Respuesta:", answer)

if __name__ == "__main__":
    main()
