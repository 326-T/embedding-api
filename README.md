# Embedding API - ハイブリッド検索API

FastAPIとPgVectorを使用したテキスト埋め込み・ハイブリッド検索APIです。

## 🚀 機能

- **テキスト埋め込み**: 単一またはバッチでテキストをベクトル化
- **ハイブリッド検索**: ベクトル検索とテキスト検索を組み合わせた高精度検索
- **カテゴリ検索**: カテゴリ別でのフィルタリング検索

## 📋 必要な環境

- Python 3.13以上
- Docker & Docker Compose
- PostgreSQL（PgVector拡張付き）

## 🛠️ セットアップ

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd embedding-api
```

### 2. データベースの起動
```bash
docker compose up -d
```

### 3. 依存関係のインストール
```bash
# uvを使用する場合
uv sync

# pipを使用する場合
pip install -e .
```

### 4. アプリケーションの起動
```bash
# uvを使用する場合
uv run uvicorn app.main:app --reload

# pipを使用する場合
python -m uvicorn app.main:app --reload
```

アプリケーションは `http://localhost:8000` で起動します。

## 📝 API仕様

### ヘルスチェック
```
GET /health
```

### 単一テキストの埋め込み
```
POST /embed
Content-Type: application/json

{
  "text": "埋め込みたいテキスト"
}
```

### バッチ埋め込み
```
POST /embed/batch
Content-Type: application/json

{
  "texts": [
    "テキスト1",
    "テキスト2",
    "テキスト3"
  ]
}
```

### データの挿入
```
POST /hybrid/insert
Content-Type: application/json

{
  "items": [
    {
      "category": "カテゴリ名",
      "title": "タイトル",
      "text": "本文テキスト"
    }
  ]
}
```

### ハイブリッド検索
```
POST /hybrid/search
Content-Type: application/json

{
  "category": "カテゴリ名（省略可）",
  "query": "検索クエリ"
}
```

## 🧪 テスト方法

### VS Code REST Clientを使用

1. VS Codeの拡張機能「REST Client」をインストール
2. `test.http` ファイルを開く
3. 各リクエストの上に表示される「Send Request」をクリック

### 基本的な使用フロー

1. **ヘルスチェック**: APIが正常に動作しているか確認
   ```
   GET http://localhost:8000/health
   ```

2. **サンプルデータの挿入**: 星座データを挿入
   ```
   POST http://localhost:8000/hybrid/insert
   ```
   `test.http` にある星座データを使用

3. **検索テスト**: 挿入したデータを検索
   ```
   POST http://localhost:8000/hybrid/search
   {
     "category": "前半",
     "query": "牡羊座の主要な星は何ですか？"
   }
   ```

### 検索結果の説明

レスポンスには以下の情報が含まれます：
- `vector_score`: ベクトル類似度スコア
- `text_score`: テキスト検索スコア  
- `hibrid_score`: ハイブリッドスコア（両方の組み合わせ）
- `category`, `title`, `text`: 元のデータ
- `created_at`: 作成日時

## 🔧 開発者向け

### テストの実行
```bash
# 全てのテスト
uv run pytest

# 単体テスト
uv run pytest -m ut

# 結合テスト
uv run pytest -m it
```

### コードフォーマット
```bash
uv run ruff format
uv run ruff check
```

## 🗂️ プロジェクト構造

```
embedding-api/
├── app/
│   ├── main.py              # FastAPIアプリケーション
│   ├── settings.py          # 設定管理
│   ├── router/              # APIルーター
│   │   ├── embed.py         # 埋め込みAPI
│   │   ├── health.py        # ヘルスチェック
│   │   └── hybrid.py        # ハイブリッド検索API
│   └── repository/          # データアクセス層
│       ├── pgvector.py      # PgVectorデータベース操作
│       └── sentence_transformer.py  # 埋め込みモデル
├── tests/                   # テストファイル
├── test.http               # REST APIテスト用ファイル
├── compose.yaml            # Docker Compose設定
└── pyproject.toml          # プロジェクト設定
```

## 🤝 貢献

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成