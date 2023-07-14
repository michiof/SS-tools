# SS-tools
テスト用に作ったツールを保管しています。

## pre-setup
.env.localを作成し、以下を登録してください。
`OPENAI_API_KEY=...`
`PINECONE_API_KEY=...`
`PINECONE_ENVIRONMENT=...`

必要なモジュールはrequirements.txtを参照してください。

## to_vector.py
csvファイル内の指定した1列を行ごとにembeddings化し、全列の情報をMeta dataとして、embeddingsと一緒にPineconeに登録するスクリプトです。
GPTのQ&A botなどで取り出したいと思っている情報を列ごとにCSVで配置すると、Pineconeにその通りに登録されます。（先頭行はMeta dataの名前になります: (例) Title, Article, published date, URLを1行目に、2行目以降はデータ）
エクセルファイルでQAで使いたい情報を編集後、csv(UTF-8)で保存すれば、そのままPineconeにベクトルデータとともに登録できます。
embeddings化とMeta dataの登録のみをするスクリプトのため、QAの出力結果をテストしたいときはqa.pyを実行してください。データフォーマットを色々と変更して試したい時にご利用ください。

まれにPineconeのIndex新規作成直後のupsertは失敗することがあります。その時はoption 2を選んで再開すれば、前回embeddings化したデータを使って、そのままPineconeに登録できます。(多少の料金の節約にはなると思います)

### 起動方法
1. 登録したいcsvファイルをdataフォルダに保存する
2. `python3 to_vector.py` で起動
3. Option 1を選び、画面指示にしたがって、必要な情報を入力する


## qa.py
Pineconeに入力したデータ（to_vector.pyのものを含む）の出力テスト用に利用できるQ&A botです。
embeddingsに入力した内容に関連した質問を入力すると、コサイン度が近いデータをPinecone上で見つけ出し、それをもとにGPTが回答を作成します。
Pineconeに登録しているMeta dataは全てGPTに渡されます。
繰り返し質問を入力できますが、前の質問文は考慮されません。

### 起動方法
1. `python3 qa.py` で起動
2. 確認したいPineconeのIndexを入力
3. 終了時は`exit`と入力する