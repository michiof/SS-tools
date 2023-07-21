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
例えば、UniqueID, Title, Article, published date, URLを1行目に、2行目以降はデータを配置し、スクリプト内でEmebeddings化する列を"Article"に指定すると、Pineconeには各行のArticleの内容に対するEmbeddingsベクトルデータと、その行に対応するMeta dataとして{"UniqueID": "UniqueIDのデータ", "Title": "Title列のデータ", "Article": "Aricle列のデータ"...}が保存されます。
（エクセルファイルでQAで使いたい情報を編集後、csv(UTF-8)で保存すれば、そのままPineconeにベクトルデータとともに登録できます。embeddings化とMeta dataの登録のみをするスクリプトのため、QAの出力結果をテストしたいときはqa.pyを実行してください。データフォーマットを色々と変更して試したい時にご利用ください。）

- 前回のデータとの差分のみembeddings化して登録した場合にはOption 2を選んでください。前回の実行時のファイルとの差分を検出して、自動で追加、入れ替え、削除を行なってくれます。
- データの入れ替えを予定している場合には、csvの一番左の項目にUniqueID(重複のない番号)をあらかじめ入れておく方がベターです。スクリプトは一番左の列をまず確認し、同じIDがあった場合にはその他の列項目が同一かを確認します。
- まれにPineconeのIndex新規作成直後のupsertは失敗することがあります。その時はoption 3を選んで再開すれば、前回embeddings化したデータを使って、そのままPineconeに登録できます。

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