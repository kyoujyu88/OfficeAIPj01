# 引き継ぎメモ

最終更新: 2026-10-01 / CPU 速度対策（ブランチ `ccr-41acd84f-feyasi`）を反映

次のセッションが背景を読み直さずに作業を再開できるよう、**経緯と判断理由**を残す。
使い方そのものは `README.md` にあるので、ここでは「なぜそうなっているか」を中心に書く。

---

## 1. プロジェクト概要

`chat_app.py` 1 ファイルで動く、オフライン・CPU 向けのローカル LLM チャットアプリ。
Tkinter 製の GUI で、`llama-cpp-python` 経由で GGUF モデルを推論する。

リポジトリの構成は意図的に小さく保っている。

| ファイル | 役割 |
| --- | --- |
| `chat_app.py` | アプリ本体（約 2400 行、単一ファイル） |
| `README.md` | 使い方・設定・トラブルシュート |
| `Models.txt` | 手元にある GGUF の一覧（実ファイル未検出時のフォールバック表示にも使う） |
| `Library.txt` | 利用者環境の `pip list` スナップショット |
| `HANDOVER.md` | このファイル |

`chat_sessions/`（会話履歴）と `chat_settings.json`（アプリ設定）は実行時に生成され、
どちらも `.gitignore` 済み。

---

## 2. 実行環境（これが設計を強く縛っている）

利用者の実機は**非力なノート PC** で、ここが多くの設計判断の前提になっている。

| 項目 | 値 |
| --- | --- |
| CPU | Intel Core i3-1315U（2 P-core + 4 E-core / 8 スレッド、15W） |
| RAM | 16 GB |
| GPU | Intel UHD Graphics 64EU（**未使用。CPU 推論のみ**） |
| OS | Windows 11 Enterprise 24H2 |
| Python | 3.12 |
| 実行場所 | VS Code の統合ターミナル |

**生成速度は数 tok/s 程度**。この前提を忘れると、机上では良くても実機で使い物にならない
設計になる（長いコンテキスト、多段のエージェントループ、重い前処理など）。

### GPU を使っていない理由

- `llama-cpp-python` の `n_gpu_layers` 既定は 0 で、コードでも指定していない
- そもそも PyPI 版は **CPU 専用ビルド**。GPU を使うにはビルドし直しが要る
- iGPU は CPU と**同じメインメモリを共有**するため、トークン生成（メモリ帯域律速）は速くならない
- 検討した結果「手間の割にリターンが薄い」と判断して見送った
- `n_threads` を 4 に減らす案も試したが、体感差は出なかった
- 現在の既定は「生成 = 物理コア数（psutil）、入力の読み込み = 論理コア数」（6.10 参照）

確認コマンド: `python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"`

---

## 3. モデル構成

- **Gemma 4 E4B-it（Q4_K_M）** を主対象とする。2026-04-02 リリース、Apache-2.0
- 画像入力には**別途 `mmproj-*.gguf` が必要**（モデル本体に画像エンコーダは入っていない）
- `llama-cpp-python` は **0.3.33 以降が必須**。`Gemma4ChatHandler`（`MTMDChatHandler` 派生）が
  このバージョンから入った
- 0.3.19 では Gemma 4 を読めず `Failed to load model from file` で落ちていた（これが調査の発端）

### Gemma 4 固有の仕様（調査済み）

| 項目 | 値 |
| --- | --- |
| 推奨サンプリング | `temperature 1.0` / `top_p 0.95` / `top_k 64`（Google 公表値） |
| ターン区切り | `<\|turn>` … `<turn\|>`（Gemma 3 までの `<start_of_turn>` とは**別物**） |
| 思考モードの有効化 | システムプロンプト先頭に `<\|think\|>` |
| 思考の出力形式 | `<\|channel>thought` … `<channel\|>` の後に最終回答 |
| 最大コンテキスト | E2B / E4B は 128K、12B 以上は 256K |
| ツール呼び出し | ネイティブ対応。`<\|tool_call>call:名前{引数}<tool_call\|>` |

**注意**: 思考チャネルの区切り文字列は Unsloth のドキュメント記載のものを採用しており、
実機の出力と突き合わせた検証はしていない。もし `<|channel>thought` が本文にそのまま
出てくるようなら `THOUGHT_OPEN` / `THOUGHT_CLOSE` の調整が要る。

---

## 4. 現在の機能一覧

| 機能 | 対応 PR |
| --- | --- |
| モデル読み込み失敗時に UI が固まらない | #7 |
| 画像・文書ファイルの添付（画像 / PDF / Word / Excel / テキスト系） | #8 |
| ログの文字化け対策、PDF のページ画像化 | #9 |
| Gemma 4 向け最適化（サンプリング、思考モード、停止・再生成、システムプロンプト） | #10 |
| 回答の途中切れ解消、パラメータ既定値、設定ファイル | #11 |
| コンテキスト予算への自動調整 | #12 |
| カメラからの取り込み | #13 |
| 添付画像のサムネイル表示、思考／回答の区切り | #14 |
| 画像が 2 枚と解釈される問題の修正 | #15 |
| モデルフォルダの設定ファイル保存と UI 選択 | #16 |
| UI の整理（詳細設定の折りたたみ、会話一覧の刷新、ショートカット、ステータスバー、文字サイズ） | #18 |
| 回答のファイル保存（回答全体 / コードブロック単体、CSV は BOM 付き） | #19 |
| Excel / Word 出力（JSON スキーマで出力を縛る、Markdown の回答からの変換） | #20 |
| CPU 速度対策（テキスト経路での計算再利用、Flash Attention、KV q8_0、スレッド数）、検索欄の案内文の修正 | 本ブランチ |

---

## 5. コードの地図（`chat_app.py`）

上から順に、設定定数 → ユーティリティ関数群 → 3 つのクラス、という構成。

| 範囲 | 内容 |
| --- | --- |
| 〜210 行 | 設定定数（環境変数で上書き可能なものが多い）。`COLORS` / フォント候補もここ |
| 210〜295 | 文字コード設定（Windows コンソールのコードページ切り替え含む） |
| 295〜365 | フォント選択・日時表示、設定ファイルの読み込み、モデル / mmproj の探索 |
| 366〜480 | 添付ファイルのテキスト抽出（`extract_document_text`）、PDF の画像化 |
| 485〜580 | カメラ（`open_camera` 等）、サムネイル生成、画像添付の生成 |
| 579〜640 | コンテキスト予算の計算と調整（`fit_to_budget`） |
| 643〜725 | `ThoughtSplitter`、`plain_content` / `content_images` |
| 726〜767 | `SessionStore`（会話履歴の JSON 保存） |
| 768〜925 | `LLMEngine`（モデル読み込み、ストリーミング、トークン数計算） |
| 926〜1024 | `CameraWindow`（プレビュー付きの小窓） |
| 1025〜 | `ChatApp`（GUI 本体）。`_build_*` で画面の部品ごとに組み立てる |

### 押さえておきたい関数

- `fit_to_budget(messages, budget, count_tokens)` — 送信前にコンテキスト予算へ収める
- `ThoughtSplitter` — ストリーム中のテキストを思考と回答に振り分ける
- `plain_content(content, with_images=False)` — パート配列を文字列化。`with_images` の
  扱いを間違えると画像が 2 枚と解釈される（後述）
- `stop_words_for(model_name)` — モデル別に停止語を切り替える

---

## 6. 設計判断と理由

**ここが引き継ぎで一番重要。** 同じ結論に再到達するのに時間がかかるものを残す。

### 6.1 モデルへ送る本文に画像名を書かない

`[添付画像: foo.png]` という文字列を本文に入れると、**「画像について書かれた文」と
「画像そのもの」で 2 枚あると解釈される**（PR #15 で実際に発生した）。

画像名は画像パートの `"name"` キーに持たせ、`plain_content(..., with_images=True)` で
保存・タイトル生成のときだけ書き出す。chat handler は `type` と `image_url` しか
見ないので、余分なキーを足しても影響しない。

**逆に、コンテキスト予算で古い画像を外すときは `with_images=True` を使う**。
実体が渡らないので二重に数えられず、何があったかの痕跡だけが残る。

### 6.2 停止語をモデル別に切り替える

初期実装では「あなた:」「User:」といった自然文を停止語にしていた（モデルの一人芝居対策）。
しかし**チャットテンプレートが効く Gemma 4 では不要なうえ、応答本文にその語が出た瞬間に
切れてしまう**。モデル名に `gemma-4` を含む場合は特殊トークンのみを使う。
他モデル（`Models.txt` にある ELYZA 等）では従来どおりの停止語も付ける。

### 6.3 思考トークンは回答と同じ予算を消費する

思考モードを有効にすると、`max_tokens` を思考が食い尽くして回答が出ない。
そのため思考モード時は `THINKING_EXTRA_TOKENS`（既定 2048）を**自動で上乗せ**し、
UI の `max_tokens` は「回答に使える枠」として扱えるようにしてある。

### 6.4 コンテキスト超過は例外になる（黙って切り詰められない）

`llama-cpp-python` はプロンプトが `n_ctx` を超えると例外を投げる。

- `llama.py:1337` → `Requested tokens (N) exceed context window of M`
- `llama_chat_format.py:3532` → `Prompt exceeds n_ctx`（mmproj 使用時）

そのため**送信前に自前で収める**必要がある。`fit_to_budget` の順序は意図的に
「①そのまま → ②最新以外の画像を外す → ③古い発話を落とす」。画像は 1 枚で数百トークン
使うわりに後続ターンで参照されないことが多いため、発話ごと捨てるより先に削る。

**システムプロンプトと最新の発話は必ず残す。** 削るのは送信内容だけで、画面の表示と
保存済み履歴はそのまま。

### 6.5 文字化けは両側を UTF-8 に揃える

出力側（Python）だけ UTF-8 にしても、**コンソールのコードページが cp932 のままだと
UTF-8 バイト列を cp932 として解釈するので化ける**。`SetConsoleOutputCP(65001)` で
コードページも切り替えて両側を揃えている。

切り分け用に、起動直後に **ASCII のみの診断行**を出す（化けていても読めるように）。

```
console codepage: out 932->65001, in 932->65001
stdout: cp932 -> utf-8
state: encoding=utf-8 isatty=True platform=win32
```

### 6.6 外部ライブラリは遅延 import して、無ければ案内するだけ

`pypdf` / `python-docx` / `openpyxl` / `pypdfium2` / `Pillow` / `opencv-python` は
すべて関数内で import する。未導入なら `pip install ...` を案内するだけで、
**アプリは落とさない**。サムネイル生成は Pillow → OpenCV の順にフォールバックし、
どちらも無ければサムネイルを出さないだけで添付機能は動く。

### 6.7 UI を固めない

`cap.read()` は 30ms 程度ブロックするため、カメラの読み取りは別スレッドで回して
UI 側は最新フレームを描くだけにしている。生成も同様にワーカースレッド＋
`token_queue` 経由で UI へ渡す（`_poll_queue` が 40ms ごとに取りに行く）。

### 6.8 n_ctx の既定を 16384 に留めている

Gemma 4 はもっと長く取れるが、**KV キャッシュのぶんメモリを消費する**。
16GB の実機で黙って増やすのは避け、代わりに使用量を可視化して
`LLM_N_CTX` で調整できるようにした。85% を超えると警告を出す。

### 6.9 UI の構成（なぜこの形か）

- **普段触らないものは畳む。** サンプリング 4 項目とシステムプロンプトは常時表示だと
  画面上部を 2 段占有していたため、折りたたみ式の「詳細設定」パネルへ移した。
  開閉状態は設定ファイルに保存する。思考モードは頻繁に切り替えるので入力欄の横に残した
- **会話一覧は `ttk.Treeview`。** 以前は 1 行ごとに Checkbutton + Button を並べていた。
  1 件選択で開き、Ctrl / Shift で複数選択したときは開かずに削除対象にする
- **開いただけの会話は保存し直さない（`_dirty`）。** 以前は切り替えのたびに保存して
  更新日時が変わり、一覧の並びが入れ替わっていた（クリックした行が動く）
- **入力欄はチャット欄より先に pack する。** pack は後から置いた部品から縮めるため、
  チャット欄 (expand) の後だと狭いウィンドウで入力欄と送信ボタンが押し出される。
  同じ理由で、右端のボタン（送信・再生成）は左側の部品より先に置いている
- **入力欄の案内文は Entry の中に灰色の文字として出す。** Label を重ねる方式は Windows で
  枠からはみ出した。表示用の変数と実際の値（`search_var`）を分け、案内文で絞り込まないようにしている
- **会話一覧の行の高さは `TkDefaultFont` の行間から決める。** 固定値だと高 DPI で文字が切れる
- **ttk ボタンは `width=-8` 等の負の値で最小幅にしている。** 既定の 11 文字幅だと
  最小サイズのウィンドウでヘッダーが溢れた
- **フォントは名前付きフォント（`self.fonts`）。** サイズを変えると表示済みの文字も追従する。
  日本語 UI フォント（Yu Gothic UI → Meiryo UI → …）を探して Tk の既定フォントも差し替える
- **モーダルダイアログ（`SaveChoiceDialog`）の検証は、キーイベントを送る前に `focus_force()` が要る。**
  フォーカスの無いウィンドウに `event_generate("<Return>")` しても届かず、`wait_window` が戻らず固まる
  （実際の利用では `show()` でフォーカスを当てているので問題ない）
- **停止は送信ボタンと兼用。** 生成中は「■ 停止」に変わる。Esc でも止まる
- **Text の既定バインドに注意。** Tk の Text は Ctrl+O（行挿入）などを持つため、
  ショートカットは入力欄にもバインドして `"break"` で止めている

### 6.10 CPU での速度対策（llama-cpp-python 0.3.33 のソースで確認したこと）

- **画像用の handler は毎ターン会話全体を読み直す。** `MTMDChatHandler.__call__` は毎回
  `llama.reset()` と `kv_cache_clear()` を呼んでから全チャンクを評価する。一方、通常の経路
  （`Llama.generate`）は前回の入力との先頭一致（prefix-match）を見て、増えた部分だけを評価する。
  mmproj を置いていると常に前者になり、会話が長くなるほど 1 ターン目の待ちが伸びていた
- **対策: 画像を含まないターンだけ `llm.chat_handler = None` にして通常の経路を使う**（`LLMEngine.stream`）。
  `Gemma4ChatHandler` も通常の経路も、同じ `tokenizer.chat_template` を同じ Jinja 設定
  （trim_blocks / lstrip_blocks / 同じ拡張・tojson）で描くので、プロンプトは同じになる。
  `chat_handler` を渡して読み込むと `chat_format` が None のままなので、読み込み後に
  `"chat_template.default"` を設定しておく（`_setup_text_path`）
- 画像用の経路の直後は KV キャッシュに画像の埋め込みが入っているので、テキスト用の経路に戻るときに
  一度だけ `llm.reset()` する。テキスト用の経路で例外が出たら（テンプレートを描けない等）、
  以後は画像用の経路だけを使う
- **`draft_model`（プロンプト参照デコード等の投機的デコード）は使えない。** 指定すると
  `logits_all=True` が強制され、`n_ctx × 語彙数` の float 配列を確保する。Gemma の語彙は約 26 万なので
  n_ctx=16384 で約 17GB になり、16GB の実機では成立しない
- Flash Attention（`flash_attn=True`）と KV キャッシュの q8_0（`type_k` / `type_v` = 8）を既定で使う。
  量子化した V キャッシュには Flash Attention が必要。読み込みで例外が出たら外して読み直す
- **実モデルでの速度は未計測**（このコンテナから Hugging Face に接続できず、モデルを取得できない）。
  経路の切り替えは偽の Llama で検証した

---

## 7. 既知の制約・ハマりどころ

### 未検証のもの

- **思考チャネルの区切り文字列**（3 章の注意書き参照）
- 文字化けの解消が実機で効いているか（Windows でしか再現しない）
- カメラのプレビューの滑らかさ、DirectShow での起動時間
- サムネイルの大きさ（`THUMBNAIL_WIDTH = 200`）が実機で適切か

### 仕様上の制約

- **会話履歴の JSON に画像は保存しない**（base64 で肥大するため）。過去の会話を
  開き直すとサムネイルは出ず、`[添付画像: 名前]` の記述だけが残る
- **長い会話では古い発話がモデルに渡らなくなる**。有限のコンテキストで会話を続ける
  以上は避けられないトレードオフで、代わりに何を外したかを毎回表示している
- `LLM_MODELS_DIR` を設定していると、UI の「フォルダ...」で変えても次回起動時は
  環境変数が優先される（その旨をチャット欄に表示する）
- `cv2.face`（LBPH）は contrib 版限定で、`opencv-python` には入っていない

### 外部要因

- **`llama-cpp-python` は Gemma 4 のツール呼び出しトークンを解析しない**
  （[issue #2227](https://github.com/abetlen/llama-cpp-python/issues/2227)、未解決）。
  `tools` を渡すと `message.content` に生テキストで返る。詳細は 9.3 を参照
- OpenCV Zoo のモデルを GitHub の raw URL から落とすと **git-lfs のポインタファイル**が
  落ちてくる。実体は `media.githubusercontent.com/media/...` 側

---

## 8. 検証方法

**既定の `python3`（3.11）には Tkinter が無く、llama-cpp-python も無い**（クラウドの Linux コンテナ）ため、
スタブを差し込んで検証してきた。この手法は再現性があるので踏襲してほしい。

### 実際の Tk で動かす（UI の見た目の確認）

コンテナには **`/usr/bin/python3.12` + Tk 8.6 と `xvfb-run`、ImageMagick の `import`** がある
（無ければ `apt-get install -y python3-tk`）。モックモードのまま実画面を動かして
スクリーンショットを撮れるので、レイアウト変更はこちらで確認すること。

```bash
xvfb-run -a -s "-screen 0 1200x780x24" /usr/bin/python3.12 drive.py
# drive.py の中で ChatApp(tk.Tk()) を作り、root.after() で操作を順に流し、
# subprocess.run(["import", "-window", "root", "shot.png"]) で撮影する。
# SESSIONS_DIR / SETTINGS_PATH はスクラッチ領域へ差し替えておく（リポジトリを汚さない）
```

日本語 UI フォント（Yu Gothic UI 等）はコンテナに無いため、見た目は実機と多少異なる。

### 書き出したファイルの検証

- openpyxl / python-docx は 3.12 に `pip install --user --break-system-packages` で入れて読み戻す
- 自前で組んだ .xlsx が Office で開けるかは **LibreOffice で変換して確認**する。
  `libreoffice-core` だけでは開けないので `apt-get install -y --no-install-recommends libreoffice-calc libreoffice-writer`。
  `HOME` を書き込める場所にし、`LANG=C.UTF-8` にしないと日本語のシート名で出力ファイル名が化けて失われる

```bash
HOME=/tmp/lo LANG=C.UTF-8 soffice --headless --norestore \
  --convert-to "csv:Text - txt - csv (StarCalc):44,34,76,1,,0,false,true,false,false,false,-1" \
  --outdir out t.xlsx      # 全シートを CSV に書き出す
```

### スタブの作り方

```python
class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()
    def __call__(self, *a, **k): return _Any()
    def __bool__(self): return False
    def __setitem__(self, k, v): pass
    def __getitem__(self, k): return _Any()
    def __iter__(self): return iter(())      # これが無いと winfo_children() で無限ループ

for name in ("tkinter", "tkinter.ttk", "tkinter.scrolledtext",
             "tkinter.messagebox", "tkinter.filedialog"):
    m = types.ModuleType(name)
    m.__getattr__ = lambda n: _Any()
    sys.modules[name] = m
for attr in ("ttk", "scrolledtext", "messagebox", "filedialog"):
    setattr(sys.modules["tkinter"], attr, sys.modules["tkinter." + attr])
```

これで `ChatApp(_Any())` を実際に構築でき、**UI 配線の typo を検出できる**。
個別のメソッドは `ChatApp.__new__(ChatApp)` で空のインスタンスを作り、
必要な属性だけ差し込んで呼ぶ。

### よく使う差し替え

| 対象 | 方法 |
| --- | --- |
| `Llama` | `chat_app.Llama = FakeLlama; chat_app.HAS_LLAMA = True` |
| カメラ | `cv2.VideoCapture` を差し替え |
| ファイルダイアログ | `C.filedialog.askdirectory = lambda **k: path` |
| ライブラリ未導入 | `builtins.__import__` を差し替えて `ImportError` を出す |
| スレッド | `C.threading.Thread` を差し替えて引数を捕捉 |

### テストで検証していた項目（約 200 項目）

セッションのスクラップ領域に 8 本のスクリプトがあったが、**リポジトリには入れておらず、
セッション終了とともに失われる**。カバーしていた範囲は以下。

| スクリプト | 項目数 | 範囲 |
| --- | --- | --- |
| `t_attach` | 26 | テキスト抽出、エンコーディング、切り詰め、`_build_content` |
| `t_pdf` | 15 | PDF の画像化、スキャン PDF の自動フォールバック |
| `t_gemma4` | 33 | サンプリング既定値、停止語、`ThoughtSplitter`、停止処理 |
| `t_params` | 30 | 打ち切り検知、設定の保存と復元 |
| `t_context` | 29 | `fit_to_budget`、`count_tokens` / `context_budget` |
| `t_camera` | 20 | カメラの開閉、PNG 変換、撮影 → 添付 |
| `t_display` | 26 | サムネイル生成、思考／回答の区切り |
| `t_modelsdir` | 21 | モデルフォルダの設定ファイル読み書き |

**次のセッションで最初にやる価値があるのは、このテスト群の再構築とリポジトリへの追加。**
上のスタブ手法と各項目名があれば再現できる。

---

## 9. 未着手・検討中のテーマ

### 9.1 RAG（優先度: 高、方針は合意済み）

対象文書の規模をまだ確認していないため未着手。方針だけ決まっている。

- **数十ファイルまでなら今の添付機能で足りる。RAG は数百〜数千件から**
- 真のボトルネックは検索ではなく、**取ってきたチャンクを読ませる prefill 時間**。
  上位 3 件 × 400 字、合計 1500 字以内に抑える設計にすること
- 埋め込みモデルは日本語特化の **Ruri v3（30m〜130m）** が第一候補。
  CPU でも十分速い。汎用なら multilingual-e5-small
- **LangChain は使わない**。`sentence-transformers` + numpy / faiss を直に叩く方が
  この規模では読みやすくデバッグしやすい（どちらも `Library.txt` に導入済み）
- ハイブリッド検索（BM25 併用）は万能ではない（上位 1 件は改善するが上位 3 件は
  悪化する傾向）。密ベクトルだけで始めて、効果を測ってから足す
- リランカー（cross-encoder）は CPU では重いので最初は入れない
- **出典表示は必須**。小型モデルは幻覚しやすいので、根拠を人が確認できることが要
- `extract_document_text` をインデックス構築でそのまま再利用できる（大きな利点）

進め方は「①インデックス構築スクリプト単体 → ②chat_app へ組み込み → ③出典表示と差分更新」。
①で検索品質を確認してから②に進む。

### 9.2 顔認証による出退勤管理（優先度: 中、技術調査済み）

事務所内利用で、精度や法令要件は厳しく求めないとのこと。

- **`opencv-python` 4.10 だけでほぼ完結する**（検証済み）
  - 顔検出: `cv2.FaceDetectorYN`（YuNet）
  - 特徴量: `cv2.FaceRecognizerSF`（SFace、128 次元）→ コサイン類似度で照合
- ONNX の重みだけ別途ダウンロードが必要（YuNet 228KB、SFace 37MB）。
  **GitHub の raw URL では git-lfs のポインタが落ちてくる**ので
  `media.githubusercontent.com/media/opencv/opencv_zoo/main/models/...` を使う
- 落とせない場合は同梱の Haar カスケード + scikit-learn の PCA で代替可能
- SFace のコサイン類似度の閾値は 0.36 前後が一般的。実環境で要調整
- **なりすまし対策は割り切る方針**だが、写真で通ることは把握して運用すること
- 認識失敗時の手動打刻ボタンは最初から付けること
- カメラ周りは `open_camera` を流用できる

### 9.3 エージェント的な利用 / ツール呼び出し（優先度: 中、調査済み）

**モデル側は公式に対応しているが、`llama-cpp-python` 側が解析しない。**

- 0.3.33 のソースを確認済み。`register_chat_format("gemma")` は Gemma 1〜3 用で、
  `<|tool_call>` を扱う処理は**どこにも無い**
- `create_chat_completion(tools=[...])` を呼ぶと、`message.tool_calls` は `None` で、
  `message.content` に生トークンが入る
- 入力側（ツール宣言の展開）は GGUF 内の Jinja テンプレートが処理するので**既に動く**
- **必要なのは出力側のパーサ（50 行程度）**。思考チャネルを先に剥がす必要があるが、
  これは既存の `ThoughtSplitter` を流用できる

引数の書式:

| 型 | 書式 |
| --- | --- |
| 文字列 | `key:<\|"\|>値<\|"\|>` |
| 整数 / 小数 | `key:30` / `key:3.5` |
| 真偽値 | `key:true` / `key:false` |
| 文字列リスト | `key:[<\|"\|>a<\|"\|>,<\|"\|>b<\|"\|>]` |

実用性の評価: **1〜2 ステップ、ツール 3〜5 個まで**。多段の自律ループは実機の速度では
現実的でない。用途が決まっているなら `tool_choice` で関数を名指しする方が確実
（JSON スキーマから文法を組んで出力形式を強制する経路は 0.3.33 でも動く）。

最初は**読み取り専用のツールから**。書き込みやコマンド実行を持たせる場合は、
実行前に内容を表示して確認を取る形にしないと、判断ミスがそのまま実害になる。

#### ファイル出力の段階的な方針（利用者と合意済み）

1. **回答を保存（実装済み）** — `on_save_answer`。モデルはコードブロックで中身を出すだけで、
   書き込みはアプリが行い、保存先は必ず人が選ぶ。ツール呼び出しの解析が不要なので確実
2. **形式を強制した出力（実装済み）** — 出力形式「Excel 表」「Word 文書」。
   JSON スキーマ（`TABLE_SCHEMA` / `DOCUMENT_SCHEMA`）で出力を縛り、アプリが .xlsx / .docx に変換する。
   「回答を保存」からも、Markdown の回答（表・見出し・箇条書き）を Excel / Word にできる
3. **書き込みツール**（未着手）— 上記パーサを作ったうえで `write_file` を持たせる。
   実行前に内容とパスを確認ダイアログで見せること

1 の設計判断:
- CSV / TSV は `utf-8-sig`（BOM 付き）で書く。日本語版 Excel は BOM の無い CSV を CP932 として読み化けるため
- 閉じていないコードブロック（max_tokens で途中終了）は不完全なので候補に出さない
- 選択画面の既定は「Markdown の表があれば Excel、無ければ最初のコードブロック」。
  `` ```csv `` で頼まれた回答は CSV のまま保存したいことが多いので、CSV ブロックだけなら Excel を既定にしない

2 の設計判断:
- **`response_format` は Gemma4ChatHandler でも効く（0.3.33 のソースで確認）。**
  `MTMDChatHandler.__call__` が `_grammar_for_response_format` でスキーマを GBNF に変換しており、
  `Gemma4ChatHandler` は `__call__` を上書きしていない。mmproj なし（チャットテンプレート経路）も同様。
  スキーマを 0.3.33 の `json_schema_to_gbnf` に通して変換できることも確認済み
- **セルは文字列だけのスキーマにした。** 数値・文字列の混在 (anyOf) は文法が複雑になり小型モデルが迷う。
  数値化はアプリ側（`_excel_value`）で行い、先頭 0 の番号は文字列のまま残す
- **スキーマだけでは各項目の意味が伝わらない**ので、`STRUCTURED_INSTRUCTIONS` をシステムプロンプトに足す
- **思考モードは無効にする。** 文法で縛ると `<|channel>thought` を出せず、思考と JSON がぶつかる
- **Excel は標準ライブラリ（zipfile）で書く。** 利用者の環境（Library.txt）に openpyxl が無く、
  オフラインで追加導入も難しいため。最小構成の Office Open XML（インライン文字列、太字見出し、
  見出し行の固定、列幅）を自前で組む。Word は導入済みの python-docx を使う
- **会話履歴には JSON ではなく Markdown を残す。** 続けて修正を頼めるようにし、「回答を保存」でも再利用できる。
  画面上の JSON は `answer_start` マークから末尾までを消して Markdown に置き換えている
- 途中で止めた・JSON として読めない（max_tokens 不足など）場合はファイルを作らず、理由を表示する

### 9.4 Markdown / コードブロックの装飾表示（優先度: 低）

Tkinter の Text ウィジェットでストリーミングしながら整形する必要があり、
単独で相応の規模になるため見送り中。

---

## 10. 作業の進め方

これまでのセッションで定着した手順。

1. `git fetch origin main` してから作業ブランチを `origin/main` に合わせ直す
2. 実装したら**スタブを使った検証スクリプトを書いて実行**し、全項目通ることを確認する
3. 既存の検証も走らせて退行がないことを確認する
4. `README.md` と `chat_app.py` の docstring も合わせて更新する
5. コミット → push → PR 作成
6. **PR 本文には「なぜそうしたか」と「検証していないこと」を明記する**

### 気をつけていたこと

- **検証できていないことは、できていないと書く。** 実機 GUI、実モデル、Windows 固有の
  挙動はこの環境では確認できない
- 挙動が変わる変更（停止語、既定値など）は理由を添えて明示する
- 既存テストの期待値が変わった場合は、**コードではなくテストを直したことを明記する**

---

## 11. 次のセッションへの申し送り

優先度順。

1. **テスト群の再構築とリポジトリへの追加**（8 章参照）。現状テストはリポジトリに無く、
   回帰検出の手段が失われている
2. **実機での未検証項目の確認**（7 章参照）。特に思考チャネルの区切り文字列
3. **RAG の対象文書の規模確認**。これが決まれば 9.1 の方針で着手できる
4. `Library.txt` の `llama_cpp_python` が 0.3.19 のまま古い（実際は 0.3.33 以降）
