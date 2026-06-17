# OfficeAIPj01

オフライン・CPU 環境向けの、ローカル LLM 活用プロジェクト。

## chat_app.py — ローカルLLM チャットアプリ (Tkinter)

Python 標準ライブラリ (Tkinter) だけで動く、エージェント的に拡張可能なチャット UI。
`llama-cpp-python` 経由で GGUF モデルを CPU 推論し、生成結果をストリーミング表示します。

### 特徴
- 追加インストール不要 (Tkinter は Python 標準。要 `python3-tk` パッケージ)
- モデル選択 / temperature / max_tokens を UI から調整
- 生成トークンのストリーミング表示
- ターミナルに動作状況 (状態遷移・初トークン遅延・tok/s 等) を随時デバッグ出力
- `llama-cpp-python` 未導入時やモデル未検出時は「モックモード」で UI 確認が可能

### 実行
```bash
python chat_app.py

# モデル (.gguf) の置き場所を指定する場合
LLM_MODELS_DIR=/path/to/models python chat_app.py
```

### 必要に応じて
実推論には `llama-cpp-python` と GGUF モデル (`Models.txt` 参照) が必要です。
GUI を使うには OS 側に Tk が必要です (例: Debian/Ubuntu なら `apt install python3-tk`)。
