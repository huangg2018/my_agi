import os
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("DASHSCOPE_BASE_URL")
api_key = os.getenv("DASHSCOPE_API_KEY")

#print(base_url, api_key)
path = "rag/resources/deepseek百度百科.txt"

#from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
loader = TextLoader(path)
documents = loader.load()

#print(documents)


from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

#print(chunks)
# for (i, chunk) in enumerate(chunks):
#     print(f"chunk {i+1}: {chunk.page_content[:100]}")

from langchain_community.embeddings import DashScopeEmbeddings

dash_embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=api_key
)

from langchain_community.chat_models.tongyi import ChatTongyi
llm = ChatTongyi(model="qwen-max", base_url=base_url, api_key=api_key)

#创建摘要生成链
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    {"chunks": lambda x: x.page_content}
    |ChatPromptTemplate.from_template("请根据以下内容生成一个简洁的摘要：{chunks}")
    |llm 
    |StrOutputParser()
)
#生成摘要
summaries = chain.batch(chunks,{"max_concurrency":10})

for i,summary in enumerate(summaries):
    print(f"摘要块 {i+1} :   {repr(summary[:50])}...")

#初始化摘要向量数据库
from langchain_community.vectorstores import Chroma

vector_store = Chroma(
    documents=chunks,
    embedding=dash_embeddings,
    persist_directory="rag/resources/chroma_db"
)

#向量数据库持久化
vector_store.persist()

#创建检索器












