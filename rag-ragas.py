import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

##################################################################
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
##################################################################


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

##################################################################
# ragasによる評価

# 評価用データセットの設定
dataset = Dataset.from_dict(
    {
        "question": [question],
        "answer": [answer],
        "context": [texts],
        "ground_truth": ["かぐたんとは、KAG社が開発したSlackアプリケーションです。"],
        "retrieved_contexts": [["かぐたんは、Slackアプリケーションです。",
                                "かぐたんは、KAG社が開発しました。"]],
    }
)

# 評価を実行
metrics = evaluate(
    dataset=dataset,
    llm=LangchainLLMWrapper(llm),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
    metrics=[
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
    ]
)

# 評価結果の表示
print(metrics)
##################################################################
