import telebot
import os
import google.generativeai as genai
from google.api_core.exceptions import InvalidArgument
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
import telebot
from dotenv import load_dotenv
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from langchain_google_genai import ChatGoogleGenerativeAI

bot = telebot.TeleBot("7024267014:AAFst9UtDkDV0o9R-Hr8FdrU6ZtBMGukRf0", parse_mode=None)
os.environ["GOOGLE_API_KEY"] = "AIzaSyC9_igugvIzqj3Zm2NDAENIFczjw7gCfKk"

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3, max_length=10000)

@bot.message_handler(func=lambda m: True)
def echo_all(message):
    response = llm.invoke(message.text)
    bot.send_message(message.chat.id, response.content)


bot.polling(none_stop=True, interval=0)
