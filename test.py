import os
import time
import warnings
import logging
import chardet
import torch  # 新增显存监控依赖
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 1️⃣ 初始化配置
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2️⃣ 模型注册表（根据项目文档要求实现双模型对比）
llm_registry = {
    "mistral": Ollama(
        model="mistral",  # 量化版本（符合SmoothQuant建议）
        temperature=0.7,
        num_ctx=4096  # 长上下文支持
    ),
    "qwen": Ollama(
        model="qwen:1.8b",     # 量化版T5-base
        temperature=0.5,
        num_ctx=2048   # T5的标准上下文长度
    )
}
current_model = "mistral"  # 默认模型

# 3️⃣ 系统组件初始化
memory = ConversationBufferMemory()
vector_db = None
qa_chain = None

def detect_encoding(file_path):
    """ 自动检测文本文件编码 """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# 4️⃣ 文档处理模块（保持原有分块策略优化）
def load_document(file_path):
    global qa_chain, vector_db
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

        # 使用当前激活模型生成问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_registry[current_model],  # 动态绑定当前模型
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        print(f"❌ 文件加载失败：{str(e)}")
        return None

# 5️⃣ 交互系统升级（新增模型控制功能）
def print_help():
    print("""\n🔧 系统指令手册：
    /switch [mistral|qwen]  - 切换大语言模型
    /compare [问题]       - 对比模型性能
    /help                - 显示帮助信息
    upload               - 上传文档进入问答模式
    exit/quit            - 退出系统""")

print_help()

while True:
    user_input = input("\nYou: ").strip()

    # 指令处理模块
    if user_input.startswith("/"):
        # 模型切换指令
        if user_input.startswith("/switch"):
            model_name = user_input.split()[-1].lower()
            if model_name in llm_registry:
                current_model = model_name
                # 显存清理（重要！）
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"🔄 已切换至 {model_name.upper()} 模型（显存占用：{torch.cuda.memory_allocated()/1024**2:.1f}MB）")
            else:
                print(f"⚠️ 可用模型：{list(llm_registry.keys())}")
            continue

        # 性能对比指令（符合项目文档的评估要求）
        elif user_input.startswith("/compare"):
            query = user_input[8:].strip()
            if not query:
                print("❌ 请输入对比问题，例如：/compare 解释注意力机制")
                continue

            print(f"\n🔍 正在对比模型性能...")
            results = {}
            for name, model in llm_registry.items():
                start_time = time.time()
                try:
                    response = model.invoke(query)
                    latency = time.time() - start_time
                    mem_usage = torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0

                    results[name] = {
                        "response": response,
                        "latency": f"{latency:.2f}s",
                        "memory": f"{mem_usage:.1f}MB"
                    }
                except Exception as e:
                    print(f"❌ {name.upper()} 模型响应失败：{str(e)}")

            print("\n🆚 性能对比结果：")
            for model, data in results.items():
                print(f"{model.upper()}:")
                print(f"⏱️ 响应时间: {data['latency']}")
                print(f"💾 显存占用: {data['memory']}")
                print(f"📝 响应示例: {data['response'][:150]}...\n")
            continue

        elif user_input == "/help":
            print_help()
            continue

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


    # 6️⃣ 动态绑定当前模型（关键更新点！）
    conversation = ConversationChain(
        llm=llm_registry[current_model],
        memory=memory
    )

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