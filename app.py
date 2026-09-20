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


    # -----------------------------
    # Sample Questions
    # -----------------------------

    st.subheader(
        "💡 Sample Questions"
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


    for index, question in enumerate(sample_questions):

        with cols[index % 2]:

            if st.button(
                question,
                use_container_width=True
            ):

                selected_question = question



    st.divider()



    # -----------------------------
    # Manual Question Input
    # -----------------------------

    manual_question = st.text_input(
        "✍️ Ask your HR question:"
    )


    # Priority:
    # Clicked question first,
    # otherwise manual input

    final_question = (
        selected_question
        if selected_question
        else manual_question
    )



    if final_question:


        if not GROQ_API_KEY:

            st.error(
                "Groq API Key is missing. Add it in Streamlit Secrets."
            )


        else:


            with st.spinner(
                "Searching HR Policy..."
            ):


                context = retrieve_answer_context(
                    final_question,
                    chunks,
                    vector_index
                )


                answer = generate_response(
                    final_question,
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
