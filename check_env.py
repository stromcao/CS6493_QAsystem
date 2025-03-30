# check_env.py
import sys, torch, numpy
from langchain_community.embeddings import HuggingFaceEmbeddings

print("[系统信息]")
print(f"Python版本: {sys.version}")
print(f"numpy版本: {numpy.__version__}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

print("\n[关键功能验证]")
try:
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("✅ 嵌入模型初始化成功")
    test_text = "测试中文向量化"
    embedding = emb.embed_query(test_text)
    print(f"向量维度: {len(embedding)}")
except Exception as e:
    print(f"❌ 初始化失败: {str(e)}")