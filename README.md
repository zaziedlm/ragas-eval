# プロジェクト名

LangChainとOpenAI GPT-4oを使用したチャットボットとRAG検索の評価

## 概要

このプロジェクトは、LangChainライブラリを使用して、OpenAIのGPT-4oモデルを利用したチャットボットを構築し、RAG（Retrieval-Augmented Generation）検索を実装します。また、ragasライブラリを使用してモデルの評価を行います。

## 機能

- LangChainを使用したチャットボットの構築
- OpenAI GPT-4oモデルの利用
- 環境変数の読み込みと設定
- RAG検索の実装
- ragasによるモデルの評価

## 使用技術

- Python
- LangChain
- OpenAI GPT-4o
- FAISS
- ragas

## インストール

1. リポジトリをクローンします。

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. 仮想環境を作成し、必要なパッケージをインストールします。

    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # Windowsの場合
    source .venv/bin/activate  # macOS/Linuxの場合
    pip install -r requirements.txt
    ```

3. 環境変数を設定します。`.env` ファイルを作成し、以下の内容を記述します。

    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## 使い方

1. プログラムを実行します。

    ```bash
    python rag-ragas.py
    ```

2. 質問内容を設定し、チャットボットの回答を確認します。

3. ragasによるモデルの評価結果を確認します。

## ライセンス

このプロジェクトはMITライセンスのもとで公開されています。

## 貢献

貢献を歓迎します。バグ報告やプルリクエストはGitHubリポジトリで受け付けています。

## 作者

- 名前: zaziedlm
- GitHub: [zaziedlm](https://github.com/zaziedlm)
