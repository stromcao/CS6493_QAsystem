import os
import warnings
import logging
import chardet  # 新增编码检测库
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 1️⃣  屏蔽不必要的警告，方便调试
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2️⃣  启动本地 Ollama (Mistral-7B)
llm = Ollama(model="mistral")

# 3️⃣  记忆机制（存储对话上下文）
memory = ConversationBufferMemory()

# 4️⃣  对话模式（普通对话）
conversation = ConversationChain(llm=llm, memory=memory)

# 5️⃣  文档问答模式（上传文件后使用）
vector_db = None

def detect_encoding(file_path):
    """ 自动检测文本文件编码 """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def load_document(file_path):
    """ 加载 & 处理文档 """
    global vector_db
    print(f"📄  读取文件：{file_path}")

    try:
        # 根据文件类型选择Loader
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(file_path)
        else:
            # 文本文件自动检测编码
            encoding = detect_encoding(file_path)
            loader = TextLoader(file_path, encoding=encoding)

        documents = loader.load()

        # 优化文本分割参数
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 增大块大小保留更多上下文
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "；", "……", "…", "　"]  # 中文友好分隔符
        )
        docs = text_splitter.split_documents(documents)

        # 显示处理信息
        print(f"✅ 成功加载 {len(docs)} 个文本块")
        print(f"📝 首文本块示例：{docs[0].page_content[:200]}...")

        # 向量化处理
        embeddings = HuggingFaceEmbeddings()
        vector_db = FAISS.from_documents(docs, embeddings)

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    except Exception as e:
        print(f"❌ 文件加载失败：{str(e)}")
        return None

qa_chain = None  # 存储问答链

# 6️⃣  交互式对话
while True:
    user_input = input("You: ")

    # 退出条件
    if user_input.lower() in ["exit", "quit"]:
        print("👋  退出对话")
        break

    # 判断用户是否提到“上传文件”
    if any(word in user_input.lower() for word in ["上传", "文件", "文档"]):
        file_path = input("📂 请输入本地文件路径: ").strip()

        # 确保文件存在
        if not os.path.exists(file_path):
            print("❌ 文件不存在")
            continue

        try:
            qa_chain = load_document(file_path)
            if qa_chain:
                print("✅ 文件已加载，现在可以基于文档提问了！")
        except Exception as e:
            print(f"❌ 加载失败：{str(e)}")
        continue

    # 如果已经上传文件，则进入文档问答模式
    if qa_chain:
        try:
            result = qa_chain({"query": user_input})
            response = f"{result['result']}\n\n📚 来源文档：{result['source_documents'][0].metadata['source']}"
        except Exception as e:
            response = f"回答时出错：{str(e)}"
    else:
        # 普通对话模式
        response = conversation.predict(input=user_input)

    print("Bot:", response)
