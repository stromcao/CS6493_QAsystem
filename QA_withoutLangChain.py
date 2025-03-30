from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 使用 Ollama 运行本地的 Mistral-7B
llm = Ollama(model="mistral")

# 短期记忆（存储最近的对话）
memory = ConversationBufferMemory()

# 构建对话链（包含 LLM + 记忆）
conversation = ConversationChain(llm=llm, memory=memory)

# 交互式对话
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = conversation.predict(input=user_input)
    print("Bot:", response)