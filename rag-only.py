"""
このプログラムは、LangChainライブラリを使用して、OpenAIのGPT-4oモデルを利用したチャットボットを構築するための設定を行います。
環境変数を読み込み、LLMモデルと埋め込みモデルを初期化します。
また、簡易なRAG（Retrieval-Augmented Generation）検索を実装しています。
具体的には、以下の処理を行います：
1. 必要なライブラリのインポート
2. 環境変数の読み込み
3. OpenAIのGPT-4oモデルを使用したLLMモデルの設定
4. チャットプロンプトの設定
5. RAG検索のためのChainの設定
6. 質問内容の設定とチャットの実行
"""
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


# Load the environment variables
load_dotenv()



# LLMモデルの設定
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 検索対象の「社内文書」を作成
texts = [
    "情報1：KDDIアジャイル開発センター株式会社は、KAGという略称で親しまれています。",
    "情報2：KAG社は、かぐたんというSlackアプリを開発しました。"
]

# ベクトルDBをローカルPC上に作成
vectorstore = FAISS.from_texts(texts=texts,embedding=embeddings)

# retrieverの設定
retriever = vectorstore.as_retriever()

# チャットプロンプトの設定
prompt = ChatPromptTemplate.from_template(
    "与えられたコンテキスト情報をもとに質問に回答してください。コンテキスト情報： {context} 質問内容： {question}"
)

# Chainの設定
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 質問内容を設定
question = "かぐたんとは何でしょうか？"
print(f"質問内容: {question}")

# チャットの実行
answer = chain.invoke(question)
print(f"回答: {answer}")