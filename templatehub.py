import os
from langchain import hub
from langchain_openai import ChatOpenAI

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# プロンプトを取得
prompt = hub.pull("zaziedlm/japanese_most_old_book")

# LLMモデルの設定
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Chainの設定
chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# プロンプトを実行する
answer = chain.invoke("2番目は何でしょうか？")
 

# 結果を表示
print(answer)