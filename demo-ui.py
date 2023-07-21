#
# Open terminal, make sure .venv is activated, and then
#   streamlit run demo-ui.py

import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Cassandra

# Load Chat Model and summarize chain for writing summary of reviews and ad copy
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents import initialize_agent

import pandas as pd
import vectorstore
from CONSTANTS import *

# Load the .env file
if not load_dotenv(find_dotenv(),override=True):
    raise Exception("Couldn't load .env file")

enableShowAd = False
##################################

@st.cache_resource()
def init_connections():
    df_meta = pd.read_parquet('metadata.parquet.gz')
    productName=df_meta.loc[df_meta['asin'] == example_asin]['title'].values[0]

    chat = ChatOpenAI(model_name=chat_model_name,temperature=0.2)
    embedding_function = OpenAIEmbeddings(model=embed_model)

    cloud_config = {'secure_connect_bundle': os.environ['ASTRA_SECUREBUNDLE_PATH']}
    auth_provider = PlainTextAuthProvider(os.environ['ASTRA_CLIENT_ID'], os.environ['ASTRA_CLIENT_SECRET'])
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()

    vstore = vectorstore.Cassandra(embedding_function, session, keyspace_name, table_name)

    zapier = ZapierNLAWrapper()
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    
    from langchain.llms import OpenAI
    llm = OpenAI(temperature=0.2)
    agent = initialize_agent(toolkit.get_tools(), llm, 
        agent="zero-shot-react-description", verbose=True)

    return chat, vstore, agent, productName

@st.cache_data()
def fetch_and_summarize_reviews(review_text_search):
    st.write("Searching for reviews...")
    docs=vstore.similarity_search(review_text_search, k=100, asin=example_asin, overall_rating=5)

    # Write summary of reviews
    prompt_template_summary = """
    Write a summary of the reviews:

    {text}

    The summary should be about ten lines long.
    """
    PROMPT = PromptTemplate(template=prompt_template_summary, input_variables=["text"])
    
    st.write("Summarizing reviews...")
    chain = load_summarize_chain(chat, chain_type="stuff", prompt=PROMPT)
    summary=chain.run(docs)
    return docs, summary, True

@st.cache_data()
def generateAd(text: str):
    prompt_template_fb = """
    Write the copy for a single Facebook ad based on the reviews:

    {text}

    As far as text goes, you can have up to 40 characters in your headline, 
    125 characters in your primary text, and 30 characters in your description. 
    The primary text should be a bullet-point style, and include emoji.
    It description should have a quote from a reviewer.
    """
    st.write("Generating ad copy...")
    PROMPT = PromptTemplate(template=prompt_template_fb, input_variables=["text"])
    chain = load_summarize_chain(chat, chain_type="stuff", prompt=PROMPT)
    fb_copy=chain.run(docs)
    return fb_copy

@st.cache_data()
def generatePopup():
    prompt_template_fb = """
    Write the for a web-popup based on the reviews:

    {text}

    It should be formatted in Markdown, and include the quote of a single reviewer 
    attributed to them. It should be about 50 words long, with a headline of about 10 words.
    The call to action should be "add to cart".
    """
    st.write("Generating popup copy...")
    PROMPT = PromptTemplate(template=prompt_template_fb, input_variables=["text"])
    chain = load_summarize_chain(chat, chain_type="stuff", prompt=PROMPT)
    popup_copy=chain.run(docs)
    return popup_copy


@st.cache_resource()
def sendEmail(toFirstName: str, eMail: str, review: str, reviewerName: str, summary: str, fromFirstName: str):
    st.write(f"Sending e-mail to {toFirstName} at {eMail}...")
    emailPrompt=f"""
    The customer {reviewerName} just gave the following review of a product named '{productName}':

    '{review}'

    Send a HTML-formatted email to {toFirstName} with email address {eMail}, 
    based on the review that {reviewerName} gave, and take into account the overall 
    summary of the review given here:

    '{summary}'

    The object of the e-mail is to encourage them to buy the product. Be sure to mention
    the customer review.

    The email should be signed with {fromFirstName}.
    """

    llmAgent.run(emailPrompt)

st.title('🦜🔗 LLM Marketing Assistant')
chat, vstore, llmAgent, productName = init_connections()

st.write("## Product:")
st.write(f"### {productName}")

st.write("---")
st.write("## Review Summary:")
review_text_search = st.text_input('What text would you like reviews based on?')
st.write("*For example:* This is a great torch!")

if review_text_search:
    docs, summary, enableShowAd = fetch_and_summarize_reviews(review_text_search)

    st.write("### LLM Summary:")
    st.write(summary)
    
    # With a streamlit expander  
    with st.expander('#### Document Similarity Search Results'):
        for i in range(len(docs)):
            st.write(f"{i+1}. {docs[i].page_content}")             

if (enableShowAd):
    st.write("### Facebook Ad Copy:")
    adCopy = generateAd(review_text_search)
    print(adCopy)
    st.text(adCopy)

if (enableShowAd):
    st.write("### Popup:")
    if st.button("Generate Popup"):
        adCopy = generatePopup()
        print(adCopy)
        st.write(adCopy)

if (enableShowAd):
    st.write("### Personalised E-mail:")
    with st.form("send_email"):
        toFirstName = st.text_input('To First Name')
        fromFirstName = st.text_input('From First Name')
        eMail = st.text_input('E-mail')
        
        submitted = st.form_submit_button("Send E-mail")
   
        if submitted:
            sendEmail(toFirstName=toFirstName, eMail=eMail, review=docs[0].page_content, reviewerName=docs[0].metadata['reviewer_name'], summary=summary, fromFirstName=fromFirstName)
            st.write("E-mail sent!")
