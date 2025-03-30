# CS6493_QAsystem
This project is a QA Q&A system, mainly supporting short dialogs, file upload dialogs, and the big prophecy models are chosen to be deployed on ollama with mistral and t5-base, so make sure that you have ollama downloaded locally and pull the corresponding models when you use it. The project supports the use of LangChain framework to process the uploaded files and provide them to the model Q&A.

How to use?
1. download Ollama and pull models on Ollama.
   Make sure you have implemented Ollama, run these commands.
   
   ollama list # You will see the models have been pull.
   
   ollama pull mistral
   
   ollama run mistral

2. download the correct software package.
   
   pip install numpy==1.23.5
   
   torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu --index-url https://download.pytorch.org/whl/cpu
   
   langchain==0.0.346
   
   langchain-community==0.0.19
   
   transformers==4.34.1
   
   sentence-transformers==2.2.2
   
   faiss-cpu==1.7.4
   
   chardet==5.2.0
   
   python-docx==0.8.11
   
   pypdf==3.17.4
   
   ollama==0.1.4
   
3. Before run the main function, please run check_env.py. If you get the result like this, that will be fine.
   
   [系统信息]
   
    Python版本: 3.9.13 (...)
   
    numpy版本: 1.23.5
   
    PyTorch版本: 2.0.1+cpu
   
    CUDA可用: False

   [关键功能验证]
   
    ✅ 嵌入模型初始化成功
   
    向量维度: 384

4. run QA_LangChain.py
