import streamlit as st
import fitz
import faiss
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from groq import Groq


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide"
)


st.title("📘 HR Policy Assistant")

st.write(
    "Upload an HR Policy PDF and ask questions. "
    "The AI answers only from your uploaded document."
)



# =====================================
# LOAD EMBEDDING MODEL
# =====================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()



# =====================================
# GROQ API KEY FROM STREAMLIT SECRETS
# =====================================

try:

    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

except:

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )



# =====================================
# PDF TEXT EXTRACTION
# =====================================

def extract_text(pdf_file):

    document = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )


    text = ""


    for page in document:

        text += page.get_text()


    return text




# =====================================
# TEXT CHUNKING
# =====================================

def chunk_text(
        text,
        chunk_size=700):


    words = text.split()


    chunks=[]


    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[i:i+chunk_size]
        )


        chunks.append(
            chunk
        )


    return chunks




# =====================================
# CREATE FAISS DATABASE
# =====================================

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




# =====================================
# SEARCH RELEVANT INFORMATION
# =====================================

def retrieve_context(
        question,
        chunks,
        index):


    question_embedding = embedding_model.encode(
        [question]
    )


    question_embedding = np.array(
        question_embedding
    ).astype("float32")



    distances, results = index.search(
        question_embedding,
        4
    )


    context=[]


    for result in results[0]:

        context.append(
            chunks[result]
        )


    return "\n\n".join(context)




# =====================================
# GROQ RESPONSE GENERATION
# =====================================

def generate_answer(
        question,
        context):


    client = Groq(
        api_key=GROQ_API_KEY
    )


    prompt = f"""

You are an HR Policy Assistant.

Answer ONLY from the HR Policy context below.

Do not use outside knowledge.

If the answer is not available, reply:

"I could not find this information in the HR policy."


HR POLICY CONTEXT:

{context}


USER QUESTION:

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




# =====================================
# SESSION CHAT HISTORY
# =====================================

if "messages" not in st.session_state:

    st.session_state.messages=[]



# =====================================
# SIDEBAR
# =====================================

with st.sidebar:


    st.header(
        "📂 Upload Document"
    )


    uploaded_file = st.file_uploader(

        "Upload HR Policy PDF",

        type=["pdf"]

    )



    st.divider()


    st.caption(
        "Powered by RAG + FAISS + Groq"
    )




# =====================================
# MAIN APPLICATION
# =====================================

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


        vector_index = create_vector_store(
            chunks
        )


    st.success(
        "✅ HR Policy Loaded Successfully"
    )



    # =================================
    # SAMPLE QUESTIONS
    # =================================


    st.subheader(
        "💡 Suggested Questions"
    )


    sample_questions = [

        "What is the annual leave policy?",

        "How many sick leaves are allowed?",

        "What are the working hours?",

        "What is the probation period?",

        "What is the termination procedure?",

        "What employee benefits are available?",

        "What is the maternity leave policy?",

        "How is overtime calculated?",

        "What is the promotion policy?",

        "What is the attendance policy?"

    ]


    selected_question = None


    cols = st.columns(2)



    for i, question in enumerate(sample_questions):


        with cols[i % 2]:


            if st.button(

                question,

                use_container_width=True

            ):

                selected_question = question





    st.divider()



    # =================================
    # MANUAL QUESTION
    # =================================


    manual_question = st.text_input(

        "✍️ Ask your own HR question"

    )



    final_question = (

        selected_question

        if selected_question

        else manual_question

    )



    if final_question:



        if not GROQ_API_KEY:


            st.error(

                "Groq API key missing. Add GROQ_API_KEY in Streamlit Secrets."

            )


        else:



            with st.spinner(

                "Finding answer..."

            ):



                context = retrieve_context(

                    final_question,

                    chunks,

                    vector_index

                )


                answer = generate_answer(

                    final_question,

                    context

                )




            st.subheader(
                "🤖 Answer"
            )


            st.write(
                answer
            )



            # Save chat

            st.session_state.messages.append(

                {

                "question":final_question,

                "answer":answer

                }

            )




    # =================================
    # CHAT HISTORY
    # =================================


    if st.session_state.messages:


        st.divider()


        st.subheader(
            "📝 Previous Questions"
        )


        for chat in reversed(
            st.session_state.messages
        ):


            with st.expander(
                chat["question"]
            ):

                st.write(
                    chat["answer"]
                )



else:


    st.info(
        "👈 Please upload an HR Policy PDF from the sidebar."
    )
