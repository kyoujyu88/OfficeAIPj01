# OfficeAIPj01

オフライン・CPU 環境向けの、ローカル LLM 活用プロジェクト。

## chat_app.py — ローカルLLM チャットアプリ (Tkinter)

Python 標準ライブラリ (Tkinter) だけで動く、エージェント的に拡張可能なチャット UI。
`llama-cpp-python` 経由で GGUF モデルを CPU 推論し、生成結果をストリーミング表示します。

### 特徴
- 追加インストール不要 (Tkinter は Python 標準。要 `python3-tk` パッケージ)
- モデル選択 / temperature / max_tokens を UI から調整
- 生成トークンのストリーミング表示
- ファイル添付 (画像 / PDF / Word / Excel / テキスト系。PDF はページ画像としても渡せる)
- ターミナルに動作状況 (状態遷移・初トークン遅延・tok/s 等) を随時デバッグ出力
- `llama-cpp-python` 未導入時やモデル未検出時は「モックモード」で UI 確認が可能

### 実行
```bash
python chat_app.py

# モデル (.gguf) の置き場所を指定する場合
LLM_MODELS_DIR=/path/to/models python chat_app.py
```

### 環境変数
| 変数 | 既定値 | 用途 |
| --- | --- | --- |
| `LLM_MODELS_DIR` | カレントディレクトリ | `.gguf` を探すフォルダ |
| `LLM_MMPROJ` | 自動検出 | 画像入力に使う mmproj ファイルの明示指定 |
| `LLM_N_CTX` | `8192` | コンテキスト長 (添付を使うと入力が伸びるため既定を広めに設定) |
| `LLM_STDIO_ENCODING` | `utf-8` | ログ出力の文字コード。`none` で変更しない |

### ログが文字化けするとき
Windows の Python は既定でコンソールのコードページ (日本語環境なら cp932) で出力しますが、
VS Code の統合ターミナルは出力を UTF-8 として解釈するため、日本語のログが化けます。
起動時に stdout/stderr を UTF-8 へ切り替えて回避しています。

素の PowerShell や コマンドプロンプト (cp932 のまま) で使っていて逆に化ける場合は、
ターミナル側を UTF-8 にする (`chcp 65001`) か、次のように出力側を合わせてください。

```powershell
$env:LLM_STDIO_ENCODING="cp932"; python chat_app.py
```

### ファイル添付
入力欄の上にある「ファイル添付」から、次の送信に添付するファイルを選びます。

**文書ファイル** はテキストへ変換して本文に埋め込むため、モデル側の対応は不要です。

| 形式 | 必要なライブラリ |
| --- | --- |
| `.txt` `.md` `.csv` `.json` `.py` などテキスト系 | 不要 (UTF-8 → CP932 の順で試行) |
| `.pdf` | `pypdf` |
| `.docx` | `python-docx` |
| `.xlsx` `.xlsm` | `openpyxl` |

ライブラリが入っていない形式を選んだ場合は、必要な `pip install` を案内するだけで
アプリは落ちません。1 ファイルあたりの取り込みは 6000 文字で打ち切ります。

#### PDF を画像として読ませる
PDF はまずテキスト抽出を試み、**文字が取れなかった場合 (スキャン PDF など) は
自動でページ画像に変換**してモデルへ渡します。図表やレイアウトごと見せたいときは
「PDFを画像として読む」にチェックを入れると、常にページ画像として渡します。

画像化には `pypdfium2` と `pillow`、そして画像入力なので **mmproj が必要**です。
1 ファイルにつき先頭 4 ページまで、144dpi 相当 (`PDF_IMAGE_MAX_PAGES` /
`PDF_IMAGE_SCALE`) で変換します。

**画像** (`.png` `.jpg` `.webp` など) を渡すには、マルチモーダル対応モデル (Gemma 4 等) と、
対になる **mmproj ファイル** (`mmproj-*.gguf`) の両方が必要です。モデルと同じフォルダに
置くと自動検出します (`LLM_MMPROJ` で明示指定も可)。mmproj が見つからない場合は
画像なしで読み込み、画像を添付して送信しようとした時点でその旨を表示します。
状態は読み込み完了メッセージの「画像入力 可 / 不可」で確認できます。

なお、会話履歴の JSON には画像そのもの (base64) は保存しません。会話を再開すると
`[添付画像: 名前]` という記述だけが残り、画像は再度添付し直す必要があります。

### 必要に応じて
実推論には `llama-cpp-python` と GGUF モデル (`Models.txt` 参照) が必要です。
画像入力には `llama-cpp-python` 0.3.33 以降 (`Gemma4ChatHandler` を含むバージョン) が必要です。
GUI を使うには OS 側に Tk が必要です (例: Debian/Ubuntu なら `apt install python3-tk`)。
