import streamlit as st
from openai import OpenAI, AuthenticationError, APIConnectionError


import pdfplumber #decided on pdfplumber because it was the easiest to get set up

def read_pdf(pdf_path): #declaring a function read the pdf files
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# Show title and description.
st.title("Tony D's Document Question Answering for HW1")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    try: 
        # Create an OpenAI client.
        client = OpenAI(api_key=openai_api_key)

        # Test API key with a lightweight call
        client.models.list()  

        # Let the user upload a file via `st.file_uploader`.
        uploaded_file = st.file_uploader(
            "Upload a document (.txt or .pdf)", type=("txt", "pdf")
        )

        # Ask the user for a question via `st.text_area`.
        question = st.text_area(
            "Now ask a question about the document!",
            placeholder="Can you give me a short summary?",
            disabled=not uploaded_file,
        )

        if uploaded_file and question:

            # Process the uploaded file using code from HW1
            file_extension = uploaded_file.name.split('.')[-1]
            if file_extension == 'txt':
                document = uploaded_file.read().decode()
            elif file_extension == 'pdf':
                document = read_pdf(uploaded_file)
            else:
                st.error("Unsupported file type.") #This line also makes it so that if there is not a supported file type in the application, a user cannot move forward and ask the AI a question.

            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {question}",
                }
            ]

            # Generate an answer using the OpenAI API.
            stream = client.chat.completions.create(
                model="gpt-5-nano",
                messages=messages,
                stream=True,
            )

            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)


    except AuthenticationError:
        st.error("Invalid OpenAI API key. Please check and try again.")
    except APIConnectionError:
        st.error("Network error: Unable to connect to OpenAI servers.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")