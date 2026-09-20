import streamlit as st
import fitz
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide"
)


st.title("📘 HR Policy Assistant")

st.write(
    "Upload your HR Policy PDF and ask questions. "
    "The assistant answers only from your document."
)


# -----------------------------
# Load Embedding Model
# -----------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()



# -----------------------------
# Get Groq API Key from Secrets
# -----------------------------

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

except Exception:

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )



# -----------------------------
# Extract PDF Text
# -----------------------------

def extract_text(pdf_file):

    document = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    text = ""

    for page in document:

        text += page.get_text()


    return text



# -----------------------------
# Text Chunking
# -----------------------------

def chunk_text(
        text,
        chunk_size=800):

    words = text.split()

    chunks=[]

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunks.append(
            " ".join(
                words[i:i+chunk_size]
            )
        )


    return chunks




# -----------------------------
# Create FAISS Vector Database
# -----------------------------

def create_vector_store(chunks):

    embeddings = embedding_model.encode(
        chunks
    )


    embeddings = np.array(
        embeddings
    ).astype("float32")


    dimension = embeddings.shape[1]


    index = faiss.IndexFlatL2(
        dimension
    )


    index.add(
        embeddings
    )


    return index



# -----------------------------
# Retrieve Relevant Information
# -----------------------------

def retrieve_answer_context(
        question,
        chunks,
        index):


    query_embedding = embedding_model.encode(
        [question]
    )


    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    distances, indexes = index.search(
        query_embedding,
        4
    )


    results=[]


    for idx in indexes[0]:

        results.append(
            chunks[idx]
        )


    return "\n\n".join(results)




# -----------------------------
# Groq LLM Response
# -----------------------------

def generate_response(
        question,
        context):


    client = Groq(
        api_key=GROQ_API_KEY
    )


    prompt=f"""

You are an HR Policy Assistant.

Answer ONLY using the provided HR policy context.

If the answer is not available in the policy,
say:

"I could not find this information in the HR policy."


HR POLICY CONTEXT:

{context}


QUESTION:

{question}

"""


    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],

        temperature=0.2

    )


    return response.choices[0].message.content




# -----------------------------
# Upload PDF
# -----------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload HR Policy PDF",
    type="pdf"
)



# -----------------------------
# Main App
# -----------------------------

if uploaded_file:


    with st.spinner(
        "Processing HR Policy..."
    ):


        text = extract_text(
            uploaded_file
        )


        chunks = chunk_text(
            text
        )


        vector_index = create_vector_store(
            chunks
        )


    st.success(
        "HR Policy Loaded Successfully!"
    )



    question = st.text_input(
        "Ask your HR question:"
    )



    if question:


        if not GROQ_API_KEY:

            st.error(
                "Groq API Key is missing. Add it in Streamlit Secrets."
            )


        else:


            with st.spinner(
                "Searching HR Policy..."
            ):


                context = retrieve_answer_context(
                    question,
                    chunks,
                    vector_index
                )


                answer = generate_response(
                    question,
                    context
                )


            st.subheader(
                "Answer"
            )


            st.write(
                answer
            )


else:

    st.info(
        "Please upload an HR Policy PDF to start."
    )
