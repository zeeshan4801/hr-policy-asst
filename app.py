import streamlit as st
import fitz
import faiss
import numpy as np
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
# Load Models
# -----------------------------

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model



embedding_model = load_embedding_model()



# -----------------------------
# Extract PDF Text
# -----------------------------

def extract_text(pdf_file):

    doc = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    text = ""

    for page in doc:
        text += page.get_text()


    return text



# -----------------------------
# Chunk Text
# -----------------------------

def chunk_text(text, size=800):

    words = text.split()

    chunks=[]

    for i in range(0,len(words),size):

        chunk=" ".join(
            words[i:i+size]
        )

        chunks.append(chunk)


    return chunks



# -----------------------------
# Create FAISS Index
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


    return index, embeddings



# -----------------------------
# Retrieve Relevant Chunks
# -----------------------------

def search_policy(
        question,
        chunks,
        index,
        k=4):


    query_embedding = embedding_model.encode(
        [question]
    )


    query_embedding=np.array(
        query_embedding
    ).astype("float32")


    distances,indices=index.search(
        query_embedding,
        k
    )


    results=[]


    for i in indices[0]:

        results.append(
            chunks[i]
        )


    return results



# -----------------------------
# Groq Answer
# -----------------------------

def ask_groq(
        question,
        context,
        api_key):


    client = Groq(
        api_key=api_key
    )


    prompt=f"""

You are an HR Policy Assistant.

Answer the question ONLY using the provided HR policy context.

If the answer is not available in the document,
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
# Sidebar
# -----------------------------


with st.sidebar:

    st.header("Settings")


    groq_key = st.text_input(
        "Enter Groq API Key",
        type="password"
    )


    uploaded_file = st.file_uploader(
        "Upload HR Policy PDF",
        type="pdf"
    )




# -----------------------------
# Main Application
# -----------------------------


if uploaded_file:


    with st.spinner(
        "Reading HR Policy..."
    ):


        text = extract_text(
            uploaded_file
        )


        chunks = chunk_text(
            text
        )


        index,_ = create_vector_store(
            chunks
        )


    st.success(
        "HR Policy Loaded Successfully!"
    )


    question = st.text_input(
        "Ask your HR question:"
    )



    if question:


        if not groq_key:

            st.warning(
                "Please enter Groq API Key"
            )

        else:

            with st.spinner(
                "Searching policy..."
            ):


                relevant_chunks = search_policy(
                    question,
                    chunks,
                    index
                )


                context="\n\n".join(
                    relevant_chunks
                )


                answer = ask_groq(
                    question,
                    context,
                    groq_key
                )


            st.subheader(
                "Answer"
            )

            st.write(
                answer
            )


else:

    st.info(
        "Please upload HR Policy PDF"
    )
