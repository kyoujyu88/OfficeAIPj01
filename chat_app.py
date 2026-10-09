#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ローカルLLM チャットアプリ (Tkinter / CPU・オフライン向け)

特徴:
  - Tkinter 製の GUI チャット (Python 標準ライブラリのみ。追加インストール不要)
  - llama-cpp-python で GGUF モデルを CPU 推論
  - 生成トークンをストリーミング表示 (生成中に逐次表示)
  - 生成の停止 / 直前の応答の再生成
  - モデル選択 / temperature / top_p / top_k / max_tokens を UI から調整
  - システムプロンプトと思考 (reasoning) モード
  - 左サイドバーに会話履歴を一覧表示
      * クリックで過去の会話を再開 / タイトルで絞り込み
      * Ctrl・Shift + クリックで複数選択してまとめて削除
  - サンプリング等は折りたたみ式の詳細設定パネルに収納 (開閉状態は保存)
  - キーボードショートカット (Esc で停止、Ctrl+N 新規、Ctrl +/- で文字サイズ 等)
  - 回答をファイルに保存 (回答全体を .md / Word、表を Excel、コードブロック単体を言語に合った拡張子で)
  - 出力形式「Excel 表」「Word 文書」: モデルの出力を JSON スキーマで縛り、アプリが .xlsx / .docx に変換
  - スキル: 用途ごとの指示・出力形式・思考モード・ひな形を保存しておき、選ぶだけで切り替える
  - 資料検索 (RAG): 資料フォルダから質問に関係する部分を探して回答に使い、出典を表示 (rag.py)
  - 会話は JSON ファイルとして自動保存 (chat_sessions/ フォルダ)
  - ファイル添付 (画像 / PDF / Word / Excel / テキスト系)
  - カメラからの取り込み (OpenCV。プレビューを見ながら撮影して添付)
  - クリップボードからの添付 (スクリーンショット・コピーしたファイル・文字。Ctrl+V / 「貼り付け」)
  - ターミナルに動作状況 (状態遷移・性能) をデバッグ出力

備考:
  既定値は Gemma 4 (E2B / E4B / 12B ほか) を前提に合わせています。
  llama-cpp-python が未導入、またはモデルファイルが見つからない場合は
  「モックモード」で起動し、UI と挙動の確認だけは行えます (実推論は行いません)。

  画像を渡すにはマルチモーダル対応モデル (Gemma 4 等) に加えて、対になる
  mmproj ファイル (mmproj-*.gguf) をモデルと同じフォルダに置く必要があります。
  文書ファイルはテキストへ変換して本文に埋め込むため、モデル側の対応は不要です。
  PDF はテキスト優先で読み、抽出できない場合 (スキャン PDF 等) はページ画像に
  変換して渡します。「PDFを画像として読む」で常に画像化することもできます。

実行:
  python chat_app.py
  # モデルの置き場所は UI の「フォルダ」で選べば chat_settings.json に保存される。
  # 一時的に別の場所を使いたい場合のみ環境変数で上書きする:
  LLM_MODELS_DIR=/path/to/models python chat_app.py
  # mmproj を明示指定する場合:
  LLM_MMPROJ=/path/to/mmproj-gemma-4-E4B.gguf python chat_app.py
"""

import io
import os
import re
import sys
import csv
import json
import time
import queue
import base64
import logging
import mimetypes
import zipfile
import subprocess
import threading
import unicodedata
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from tkinter import font as tkfont

# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------
# モデル (.gguf) を置いているフォルダ。環境変数 LLM_MODELS_DIR で上書き可。
# 優先順位は 環境変数 > 設定ファイル (chat_settings.json) > カレントディレクトリ。
# 設定ファイルの値は起動時に読み込み、UI の「フォルダ」で変更できる。
MODELS_DIR_FROM_ENV = bool(os.environ.get("LLM_MODELS_DIR"))
MODELS_DIR = Path(os.environ.get("LLM_MODELS_DIR", ".")).expanduser()

# 会話履歴 (JSON) の保存先。スクリプトと同じ場所の chat_sessions/ 。
SESSIONS_DIR = Path(__file__).resolve().parent / "chat_sessions"

# アプリ設定 (システムプロンプト・サンプリング値など) の保存先。
# 会話ごとの設定は各会話の JSON 側に持つ。こちらは「新しいチャットの初期値」。
SETTINGS_PATH = Path(__file__).resolve().parent / "chat_settings.json"

# 資料検索 (RAG) を使うとき、検索した抜粋のために空けておくトークン数
# (rag.MAX_CONTEXT_CHARS = 1500 字 + 出典や指示の文。日本語は 1 字 ≒ 1 トークン弱)
RAG_RESERVED_TOKENS = 1200

# スキル (用途ごとの指示のセット) の保存先。「スキルの設定」画面で編集する。
SKILLS_PATH = Path(__file__).resolve().parent / "chat_skills.json"

# マルチモーダル投影ファイル。未指定ならモデルフォルダから mmproj*.gguf を探す。
MMPROJ_OVERRIDE = os.environ.get("LLM_MMPROJ")

# 既定の推論パラメータ。
# 添付ファイルや思考モードで入力・出力とも長くなるため、環境変数で調整できる。
DEFAULT_N_CTX = int(os.environ.get("LLM_N_CTX", 16384))
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", 2048))


def _physical_cores():
    """物理コア数。分からなければ論理コア数。"""
    try:
        import psutil

        return psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except Exception:
        return os.cpu_count() or 4


# スレッド数。トークン生成はメモリ帯域で頭打ちになるため、ハイパースレッドの分まで
# 増やしても速くならず、かえって遅くなることがある -> 生成は物理コア数。
# 入力の読み込み (prefill) は計算量で決まるので論理コア数まで使う。
DEFAULT_N_THREADS = int(os.environ.get("LLM_N_THREADS", 0)) or _physical_cores()
DEFAULT_N_THREADS_BATCH = int(os.environ.get("LLM_N_THREADS_BATCH", 0)) or os.cpu_count() or 4

# ---- 速度向上の設定 ------------------------------------------------------
# Flash Attention: 注意機構の計算をまとめて行い、長い入力の読み込みを速くする。
# KV キャッシュの量子化 (q8_0) にも必要。読み込みに失敗した場合は自動で外して読み直す。
FLASH_ATTN = os.environ.get("LLM_FLASH_ATTN", "1").lower() not in ("0", "false", "off", "no")
# KV キャッシュの型。q8_0 にするとメモリが半分になり、長い会話での読み書きが軽くなる
# (品質への影響はごく小さい)。f16 で従来どおり。Flash Attention が無効なら f16 になる。
KV_CACHE_TYPE = os.environ.get("LLM_KV_TYPE", "q8_0").lower()
# ggml の型番号 (llama_cpp.GGML_TYPE_* と同じ値)
GGML_TYPES = {"f16": 1, "q4_0": 2, "q8_0": 8}

# 履歴をモデルへ送るときの予算計算。
# チャットテンプレートの制御トークン等のぶんを余白として引いておく。
CONTEXT_MARGIN_TOKENS = 256
# 1 メッセージあたりの制御トークン (ロール表記やターン区切り) の見積り
MESSAGE_OVERHEAD_TOKENS = 8
# 画像 1 枚あたりのトークン数の見積り (実際はモデル依存なので多めに見る)
IMAGE_TOKEN_ESTIMATE = 320

# 思考モードのときに max_tokens へ上乗せする枠。
# 思考トークンは回答と同じ予算から消費されるため、上乗せしないと
# 思考だけで使い切って回答が途中で切れる。
THINKING_EXTRA_TOKENS = int(os.environ.get("LLM_THINKING_TOKENS", 2048))

# サンプリングの既定値は Google が公表している Gemma 4 の推奨値に合わせる。
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

# サイドバーのタイトル表示の最大文字数
TITLE_MAXLEN = 20

# ---- Gemma 4 の制御トークン --------------------------------------------
# 思考 (reasoning) はシステムプロンプト先頭の <|think|> で有効になる。
THINK_TOKEN = "<|think|>"
# 思考を有効にすると、最終回答の前に思考チャネルが出力される。
#   <|channel>thought
#   [内部の思考]
#   <channel|>
#   [最終回答]
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"

# 停止語 (これが現れたら生成を打ち切る)。
# チャットテンプレートを使う場合、本来 EOS で止まるので特殊トークンだけで足りる。
STOP_TOKENS = [
    "<turn|>", "<|turn>",           # Gemma 4 のターン区切り
    "<|im_end|>", "<|im_start|>",   # ChatML (Yi-Coder 等)
]
# チャットテンプレートを持たない古いモデル向けの保険。
# 「あなた:」等と続きを勝手に生成する“一人芝居”を防ぐが、正規の応答に
# これらの語が含まれると途中で切れてしまうため、Gemma 4 では使わない。
LEGACY_STOP_WORDS = [
    "\nあなた:", "\nUser:", "\nuser:",
    "あなた:", "User:",
]


def stop_words_for(model_name):
    """モデル名に応じた停止語を返す。"""
    name = (model_name or "").lower()
    if "gemma-4" in name or "gemma4" in name:
        return list(STOP_TOKENS)
    return STOP_TOKENS + LEGACY_STOP_WORDS

# UI がワーカースレッドからの出力を取りに行く間隔 (ミリ秒)
POLL_INTERVAL_MS = 40

# ---- 添付ファイル --------------------------------------------------------
# 画像はモデルへそのまま渡すため、mmproj を読み込めた場合のみ添付できる。
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# テキストとして読めるファイル (そのまま本文へ埋め込む)。
TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".log",
    ".yaml", ".yml", ".ini", ".xml", ".html", ".sql",
    ".py", ".js", ".ts", ".c", ".cpp", ".h", ".java", ".cs", ".bat", ".ps1",
}

# 変換に外部ライブラリが要る文書形式 {拡張子: (pip 名, 読み込み関数)}。
# 関数は下で定義するため、辞書の実体は定義後に組み立てる。
DOC_LIBRARIES = {".pdf": "pypdf", ".docx": "python-docx", ".xlsx": "openpyxl", ".xlsm": "openpyxl"}

# 1 ファイルあたり本文へ埋め込む最大文字数 (n_ctx を溢れさせないための保険)。
MAX_DOC_CHARS = 6000

# PDF をページ画像として渡すときの設定 (スキャン PDF や図表主体の資料向け)。
PDF_IMAGE_SCALE = 2          # 1 = 72dpi 相当。2 で 144dpi
PDF_IMAGE_MAX_PAGES = 4      # 1 ファイルから作る最大ページ数

# 添付一覧ラベルに表示するファイル名の最大文字数
ATTACH_LABEL_MAXLEN = 60

# チャット欄に貼る画像サムネイルの幅 (px)
THUMBNAIL_WIDTH = 200

# ---- 見た目 --------------------------------------------------------------
# 日本語が読みやすいフォントを上から順に探す (どれも無ければ Tk の既定フォント)
UI_FONT_CANDIDATES = (
    "Yu Gothic UI", "Meiryo UI", "Meiryo", "BIZ UDPGothic",
    "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "Hiragino Sans",
)
# チャット欄・入力欄の文字サイズ (pt)。Ctrl + / Ctrl - / Ctrl+ホイールで変更でき、設定に保存される
DEFAULT_FONT_SIZE = 11
FONT_SIZE_MIN = 8
FONT_SIZE_MAX = 24

# 配色 (チャット欄と状態表示)
COLORS = {
    "bg": "#ffffff",
    "border": "#d0d7de",
    "accent": "#0969da",
    "muted": "#6e7781",
    "user_head": "#1a7f37",
    "user_bg": "#f0f6ff",
    "assistant_head": "#0a3069",
    "text": "#1f2328",
    "thought": "#8250df",
    "error": "#cf222e",
    "ok": "#1a7f37",
    "warn": "#9a6700",
}

# ---- 回答の保存 ----------------------------------------------------------
# コードブロックの言語名 -> 保存するときの拡張子。ここに無い言語は .txt
CODE_BLOCK_SUFFIXES = {
    "csv": ".csv", "tsv": ".tsv", "json": ".json", "jsonl": ".jsonl",
    "markdown": ".md", "md": ".md", "text": ".txt", "txt": ".txt", "plaintext": ".txt",
    "html": ".html", "xml": ".xml", "yaml": ".yaml", "yml": ".yaml", "ini": ".ini",
    "toml": ".toml", "sql": ".sql", "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js", "typescript": ".ts", "ts": ".ts",
    "java": ".java", "c": ".c", "cpp": ".cpp", "c++": ".cpp", "csharp": ".cs", "cs": ".cs",
    "vb": ".vb", "vba": ".bas", "bat": ".bat", "batch": ".bat", "cmd": ".bat",
    "powershell": ".ps1", "ps1": ".ps1", "pwsh": ".ps1", "bash": ".sh", "sh": ".sh",
    "shell": ".sh", "css": ".css", "mermaid": ".mmd",
}
# Excel で開いたときに文字化けしないよう BOM 付き UTF-8 で書く拡張子
# (日本語版 Excel は BOM の無い CSV を CP932 として読む)
BOM_SUFFIXES = {".csv", ".tsv"}
# 選択画面で各コードブロックの先頭を見せる行数
SAVE_PREVIEW_LINES = 3

# ---- カメラ --------------------------------------------------------------
# 使うカメラの番号 (内蔵カメラは 0。外付けを使う場合は 1 以降)
CAMERA_INDEX = int(os.environ.get("LLM_CAMERA_INDEX", 0))
# プレビューの表示幅 (px)。撮影される画像はカメラの解像度のまま。
CAMERA_PREVIEW_WIDTH = 480
# プレビューの更新間隔 (ミリ秒)
CAMERA_POLL_MS = 33

# --------------------------------------------------------------------------
# 標準出力の文字コード
# --------------------------------------------------------------------------
# 日本語の Windows では、Python の出力 (既定は cp932) とターミナルの解釈が食い違って
# ログが化ける。片側だけ UTF-8 にしても直らないため、
#   1. コンソールのコードページを UTF-8 (65001) にする
#   2. stdout/stderr も UTF-8 で書く
# の両方を行って揃える。
# cp932 のまま使いたい場合は LLM_STDIO_ENCODING=cp932、
# 何もしてほしくない場合は LLM_STDIO_ENCODING=none を指定する。
STDIO_ENCODING = os.environ.get("LLM_STDIO_ENCODING", "utf-8")


def _configure_windows_console():
    """Windows コンソールのコードページを UTF-8 に切り替える。

    診断しやすいよう、戻り値は ASCII のみの文字列にする (化けていても読める)。
    """
    if os.name != "nt" or STDIO_ENCODING.lower().replace("-", "") != "utf8":
        return []
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        before = (kernel32.GetConsoleOutputCP(), kernel32.GetConsoleCP())
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
        after = (kernel32.GetConsoleOutputCP(), kernel32.GetConsoleCP())
        return [f"console codepage: out {before[0]}->{after[0]}, in {before[1]}->{after[1]}"]
    except Exception as e:
        return [f"console codepage: FAILED ({e})"]


def _configure_stdio_encoding():
    """stdout/stderr の文字コードを揃える。変更内容を ASCII で返す (化けても読める)。"""
    if STDIO_ENCODING.lower() in ("none", "off", ""):
        return []
    changes = _configure_windows_console()
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        current = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if current == STDIO_ENCODING.lower().replace("-", ""):
            continue
        try:
            # errors="replace": 変換できない文字があっても落とさない
            stream.reconfigure(encoding=STDIO_ENCODING, errors="replace")
        except Exception as e:
            changes.append(f"{name}: {current} -> {STDIO_ENCODING} FAILED ({e})")
            continue
        changes.append(f"{name}: {current} -> {STDIO_ENCODING}")
    # まだ化ける場合の切り分け用に、最終状態も必ず残す
    try:
        changes.append(
            "state: encoding=%s isatty=%s platform=%s"
            % (getattr(sys.stdout, "encoding", "?"), sys.stdout.isatty(), sys.platform)
        )
    except Exception:
        pass
    return changes


_STDIO_CHANGES = _configure_stdio_encoding()

# --------------------------------------------------------------------------
# ロギング (ターミナルへのデバッグ出力)
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)-5s] %(threadName)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chat_app")

for _change in _STDIO_CHANGES:
    log.info("stdio encoding %s", _change)

# --------------------------------------------------------------------------
# llama-cpp-python (無ければモックモード)
# --------------------------------------------------------------------------
try:
    from llama_cpp import Llama

    HAS_LLAMA = True
except Exception as exc:  # ImportError 等
    Llama = None
    HAS_LLAMA = False
    log.warning("llama-cpp-python を読み込めません (%s) -> モックモードで起動します", exc)


def pick_ui_font(families):
    """使えるフォントの中から、日本語向けの UI フォントを選ぶ。無ければ None。"""
    available = set(families)
    return next((f for f in UI_FONT_CANDIDATES if f in available), None)


def format_updated(ts, now=None):
    """会話一覧に出す更新日時。今日なら時刻、今年なら月日、それ以前は年月日。"""
    if not ts:
        return ""
    t = time.localtime(ts)
    n = time.localtime(now if now is not None else time.time())
    if t[:3] == n[:3]:
        return time.strftime("%H:%M", t)
    if t.tm_year == n.tm_year:
        return time.strftime("%m/%d", t)
    return time.strftime("%Y/%m/%d", t)


def load_settings_file():
    """設定ファイルを読む。無い / 壊れている場合は空の辞書を返す。"""
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.info("[設定] 保存済みの設定はありません (既定値で起動)")
        return {}
    except Exception as e:
        log.warning("[設定] 読み込みに失敗しました (%s) -> 既定値で起動", e)
        return {}


def set_models_dir(path):
    """モデルフォルダを切り替える。"""
    global MODELS_DIR
    MODELS_DIR = Path(path).expanduser()
    log.info("[設定] モデルフォルダ: %s", MODELS_DIR.resolve())
    return MODELS_DIR


def discover_models():
    """MODELS_DIR 内の .gguf を探す。無ければ Models.txt の一覧をフォールバック表示。"""
    models = sorted(p.name for p in MODELS_DIR.glob("*.gguf"))
    if models:
        log.info("モデル %d 件を検出: %s", len(models), MODELS_DIR.resolve())
        return models
    txt = Path("Models.txt")
    if txt.exists():
        names = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        log.warning("実ファイル未検出。Models.txt の一覧を表示します (%d 件)", len(names))
        return names
    log.warning("モデルが見つかりません (MODELS_DIR=%s)", MODELS_DIR.resolve())
    return []


def discover_mmproj():
    """マルチモーダル投影ファイル (mmproj-*.gguf) を探す。無ければ None。"""
    if MMPROJ_OVERRIDE:
        p = Path(MMPROJ_OVERRIDE).expanduser()
        if p.exists():
            return p
        log.warning("LLM_MMPROJ が指すファイルがありません: %s", p)
        return None
    found = sorted(MODELS_DIR.glob("*mmproj*.gguf"))
    if len(found) > 1:
        log.info("mmproj が複数見つかったため先頭を使用します: %s", [p.name for p in found])
    return found[0] if found else None


# --------------------------------------------------------------------------
# 添付ファイルの読み取り
# --------------------------------------------------------------------------
def _read_plain_text(path):
    """テキスト系ファイルを読む。UTF-8 で駄目なら CP932 (Windows 既定) を試す。"""
    for encoding in ("utf-8", "cp932"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path):
    import docx

    document = docx.Document(str(path))
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:                     # 表はタブ区切りで平坦化
        for row in table.rows:
            lines.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(lines)


def _read_xlsx(path):
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    try:
        lines = []
        for sheet in workbook.worksheets:
            lines.append(f"# シート: {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                if any(v is not None for v in row):
                    lines.append("\t".join("" if v is None else str(v) for v in row))
        return "\n".join(lines)
    finally:
        workbook.close()


DOC_READERS = {".pdf": _read_pdf, ".docx": _read_docx, ".xlsx": _read_xlsx, ".xlsm": _read_xlsx}

# 添付ダイアログに出す対応拡張子
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | TEXT_SUFFIXES | set(DOC_READERS)


def extract_document_text(path, max_chars=MAX_DOC_CHARS):
    """文書ファイルからテキストを取り出す。読めない場合は RuntimeError。

    max_chars: これを超える部分は省略する (添付用)。None なら全文 (資料検索の索引用)。
    """
    suffix = path.suffix.lower()
    if suffix in DOC_READERS:
        try:
            text = DOC_READERS[suffix](path)
        except ImportError:
            lib = DOC_LIBRARIES.get(suffix, "")
            raise RuntimeError(f"{suffix} を読むには {lib} が必要です (pip install {lib})")
        except Exception as e:
            raise RuntimeError(f"読み取りに失敗しました ({e})")
    elif suffix in TEXT_SUFFIXES:
        text = _read_plain_text(path)
    else:
        raise RuntimeError(f"未対応の形式です: {suffix or '(拡張子なし)'}")

    text = text.strip()
    if not text:
        raise RuntimeError("テキストを抽出できませんでした (画像だけの PDF などの可能性)")
    if max_chars is not None and len(text) > max_chars:
        omitted = len(text) - max_chars
        log.warning("[添付] %s が長いため %d 文字を省略しました", path.name, omitted)
        text = text[:max_chars] + f"\n…(以降 {omitted} 文字を省略)"
    return text


def render_pdf_pages(path, max_pages=PDF_IMAGE_MAX_PAGES, scale=PDF_IMAGE_SCALE):
    """PDF の各ページを PNG に変換する。戻り値は [(ページ番号, PNG bytes), ...]。

    テキストを持たないスキャン PDF や、図表・レイアウトごと見せたい資料向け。
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise RuntimeError("PDF の画像化には pypdfium2 が必要です (pip install pypdfium2)")

    import io

    try:
        document = pdfium.PdfDocument(str(path))
    except Exception as e:
        raise RuntimeError(f"PDF を開けませんでした ({e})")
    try:
        total = len(document)
        images = []
        for index in range(min(total, max_pages)):
            buffer = io.BytesIO()
            document[index].render(scale=scale).to_pil().save(buffer, format="PNG")
            images.append((index + 1, buffer.getvalue()))
        if total > max_pages:
            log.warning(
                "[添付] %s は %d ページ中 %d ページのみ画像化しました",
                path.name, total, max_pages,
            )
        return images
    except ImportError:
        raise RuntimeError("PDF の画像化には Pillow が必要です (pip install pillow)")
    except Exception as e:
        raise RuntimeError(f"ページを画像化できませんでした ({e})")
    finally:
        try:
            document.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# カメラ (OpenCV)
# --------------------------------------------------------------------------
def _require_cv2():
    """OpenCV を遅延 import する。無ければ導入方法を添えて RuntimeError。"""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("カメラを使うには opencv-python が必要です (pip install opencv-python)")
    return cv2


def open_camera(index=CAMERA_INDEX):
    """カメラを開く。開けなければ RuntimeError。"""
    cv2 = _require_cv2()
    # Windows の既定 (MSMF) は起動が遅いことがあるため DirectShow を先に試す
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            log.info("[カメラ] index=%d backend=%d で開きました", index, backend)
            return capture
        capture.release()
    raise RuntimeError(
        f"カメラを開けません (index={index})。"
        "他のアプリが使用中か、LLM_CAMERA_INDEX の指定を確認してください"
    )


def encode_png(frame):
    """OpenCV のフレーム (BGR) を PNG バイト列にする。"""
    cv2 = _require_cv2()
    ok, buffer = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("画像の変換に失敗しました")
    return buffer.tobytes()


def frame_to_preview(frame, width=CAMERA_PREVIEW_WIDTH):
    """プレビュー用に縮小し、tk.PhotoImage に渡せる base64 PNG にする。

    Tk 8.6 の PhotoImage は PNG を直接読めるので、PIL に依存せず表示できる。
    """
    cv2 = _require_cv2()
    height, current = frame.shape[:2]
    if current > width:
        frame = cv2.resize(frame, (width, max(1, round(height * width / current))))
    return base64.b64encode(encode_png(frame)).decode("ascii")


def thumbnail_png(data, max_width=THUMBNAIL_WIDTH):
    """画像バイト列を縮小した PNG にする。作れなければ None。

    Pillow を優先し (GIF 等も読める)、無ければ OpenCV を使う。
    どちらも無ければサムネイルを諦めるだけで、添付そのものには影響しない。
    """
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            if image.width > max_width:
                height = max(1, round(image.height * max_width / image.width))
                image = image.resize((max_width, height))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception:
        pass
    try:
        import cv2
        import numpy as np

        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return encode_png(frame) if frame.shape[1] <= max_width else encode_png(
            cv2.resize(frame, (max_width, max(1, round(frame.shape[0] * max_width / frame.shape[1]))))
        )
    except Exception:
        log.debug("[表示] サムネイルを作れませんでした", exc_info=True)
        return None


def data_uri_bytes(uri):
    """data URI から中身のバイト列を取り出す。"""
    return base64.b64decode(uri.split(",", 1)[1])


def image_attachment(name, data, mime="image/png"):
    """画像バイト列を data URI 形式の添付エントリにする。"""
    uri = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    return {"kind": "image", "name": name, "data_uri": uri}


def message_tokens(message, count_tokens):
    """1 メッセージが消費するおおよそのトークン数。"""
    content = message.get("content")
    images = 0 if isinstance(content, str) else sum(
        1 for part in content or []
        if isinstance(part, dict) and part.get("type") == "image_url"
    )
    return (
        count_tokens(plain_content(content))
        + images * IMAGE_TOKEN_ESTIMATE
        + MESSAGE_OVERHEAD_TOKENS
    )


def strip_images(message):
    """画像パートを外して本文だけにする。画像が無ければ元のまま返す。

    画像を外した代わりに [添付画像: 名前] を残す。実体はもう渡らないので
    二重に数えられる心配はなく、何があったかの手がかりだけが残る。
    """
    content = message.get("content")
    if isinstance(content, str):
        return message
    return dict(message, content=plain_content(content, with_images=True))


def fit_to_budget(messages, budget, count_tokens):
    """モデルへ送るメッセージを予算 (トークン数) に収める。

    システムプロンプトは必ず残す。会話が伸びてもシステムプロンプトが
    押し出されないようにするのが目的。

      1. そのまま収まればそのまま
      2. 収まらなければ、最新の 1 件を除いて画像を外す (画像は重いわりに
         後続のターンでは参照されないことが多い)
      3. それでも収まらなければ、古い発話から落とす (最後の 1 件は必ず残す)

    戻り値は (送るメッセージ, 落とした件数, 画像を外した件数)。
    画面と保存済み履歴はそのままで、モデルへの入力だけを削る。
    """
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    base = sum(message_tokens(m, count_tokens) for m in system)
    costs = [message_tokens(m, count_tokens) for m in rest]

    if base + sum(costs) <= budget:
        return messages, 0, 0

    stripped = 0
    for i in range(len(rest) - 1):          # 最新の 1 件は画像を残す
        lighter = strip_images(rest[i])
        if lighter is not rest[i]:
            rest[i] = lighter
            costs[i] = message_tokens(lighter, count_tokens)
            stripped += 1

    dropped = 0
    while len(rest) > 1 and base + sum(costs) > budget:
        rest.pop(0)
        costs.pop(0)
        dropped += 1
    return system + rest, dropped, stripped


class ThoughtSplitter:
    """ストリーム中のテキストを「思考」と「回答」に振り分ける。

    Gemma 4 は思考モード時に <|channel>thought ... <channel|> という区切りで
    内部の思考を先に出力する。トークンは細切れで届き、区切り文字列が
    チャンクをまたぐことがあるため、判定できない末尾は次回に持ち越す。
    """

    def __init__(self):
        self.buffer = ""
        self.in_thought = False

    def feed(self, piece):
        """[(種別, テキスト), ...] を返す。種別は "thought" か "answer"。"""
        self.buffer += piece
        out = []
        while True:
            marker = THOUGHT_CLOSE if self.in_thought else THOUGHT_OPEN
            kind = "thought" if self.in_thought else "answer"
            index = self.buffer.find(marker)
            if index >= 0:
                if index:
                    out.append((kind, self.buffer[:index]))
                self.buffer = self.buffer[index + len(marker):]
                self.in_thought = not self.in_thought
                continue
            # 区切りの一部が末尾に来ている可能性があるぶんだけ残す
            keep = len(marker) - 1
            if len(self.buffer) > keep:
                out.append((kind, self.buffer[:-keep] if keep else self.buffer))
                self.buffer = self.buffer[-keep:] if keep else ""
            break
        return [(k, t) for k, t in out if t]

    def flush(self):
        """残りを吐き出す。生成終了時に呼ぶ。"""
        if not self.buffer:
            return []
        kind = "thought" if self.in_thought else "answer"
        out = [(kind, self.buffer)]
        self.buffer = ""
        return out


def content_images_count(content):
    """content に含まれる画像パートの数。"""
    if not isinstance(content, list):
        return 0
    return sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")


def content_images(content):
    """content に含まれる画像のバイト列を取り出す (チャット欄への表示用)。"""
    if isinstance(content, str):
        return []
    images = []
    for part in content or []:
        if isinstance(part, dict) and part.get("type") == "image_url":
            try:
                data = data_uri_bytes(part["image_url"]["url"])
            except Exception:
                log.debug("[表示] 画像を取り出せませんでした", exc_info=True)
                continue
            if data:
                images.append(data)
    return images


def plain_content(content, with_images=False):
    """content (str または OpenAI 形式のパート配列) を文字列にする。

    with_images=True のときだけ [添付画像: 名前] の行を足す (保存・タイトル用)。
    モデルへ送る本文には入れない。画像そのものと二重に数えられてしまうため。
    """
    if isinstance(content, str):
        return content
    texts = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif with_images and part.get("type") == "image_url":
            texts.append(f"[添付画像: {part.get('name', '画像')}]")
    return "\n".join(t for t in texts if t)


def extract_code_blocks(text):
    """Markdown のコードブロック (```lang ... ```) を [(言語, 中身), ...] で返す。

    閉じていないブロック (max_tokens で途中終了した等) は不完全なので含めない。
    ``` の前のインデント (箇条書きの中のブロック) は許す。
    """
    blocks = []
    lang = None
    body = []
    fence = ""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if lang is None:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence = stripped[:3]
                lang = stripped[3:].strip().split()[0].lower() if stripped[3:].strip() else ""
                body = []
        elif stripped == fence:
            blocks.append((lang, "\n".join(body) + "\n"))
            lang = None
        else:
            body.append(line)
    return blocks


def suffix_for_language(lang):
    """コードブロックの言語名から保存用の拡張子を決める。"""
    return CODE_BLOCK_SUFFIXES.get((lang or "").lower(), ".txt")


def save_text_file(path, text):
    """テキストを保存する。CSV / TSV は Excel 向けに BOM 付きで書く。"""
    path = Path(path)
    encoding = "utf-8-sig" if path.suffix.lower() in BOM_SUFFIXES else "utf-8"
    path.write_text(text, encoding=encoding)
    return path


# --------------------------------------------------------------------------
# Excel / Word への書き出し
# --------------------------------------------------------------------------
# 表と文書は共通の形で扱う。
#   表:   {"title": str, "columns": [str, ...], "rows": [[str, ...], ...]}
#   文書: {"title": str, "blocks": [{"type": ..., "text": str, ...}, ...]}
#         type は heading (level 付き) / paragraph / bullet / number / code / table
#         (table は {"type": "table", "table": 表})
# どちらも「モデルに JSON で出させたもの」と「Markdown の回答から取り出したもの」の
# 両方から作る。書き出しはアプリが行い、モデルにファイル形式そのものは書かせない。

# 出力形式の選択肢 {表示名: 種別}。None は通常のチャット
OUTPUT_MODES = {"チャット": None, "Excel 表": "table", "Word 文書": "document"}

# モデルの出力を縛る JSON スキーマ (llama-cpp-python が文法に変換して強制する)。
# 小さなモデルでも崩れにくいよう、セルは文字列だけにして数値化はアプリ側で行う。
TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "columns": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
    },
    "required": ["title", "columns", "rows"],
}
DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["heading", "paragraph", "bullet"]},
                    "text": {"type": "string"},
                },
                "required": ["type", "text"],
            },
        },
    },
    "required": ["title", "blocks"],
}
STRUCTURED_SCHEMAS = {"table": TABLE_SCHEMA, "document": DOCUMENT_SCHEMA}

# 形式を縛るだけだと各項目の意味が伝わらないため、システムプロンプトにも説明を足す
STRUCTURED_INSTRUCTIONS = {
    "table": (
        "回答は表データとして、JSON だけを出力してください。"
        "title は表の題名、columns は列見出しの配列、rows は各行のセルの配列です。"
        "各行のセルの数は columns と同じにし、数値も文字列で書いてください。"
    ),
    "document": (
        "回答は文書として、JSON だけを出力してください。"
        "title は文書の題名、blocks は本文を上から順に並べた配列です。"
        "type は heading (見出し) / paragraph (段落) / bullet (箇条書きの 1 項目) のいずれかで、"
        "text にその文章を書きます。"
    ),
}

# Excel の数値として扱う文字列 (先頭が 0 の番号やコードは文字列のまま残す)
_NUMBER_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")
_GROUPED_NUMBER_RE = re.compile(r"^-?[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?$")
# XML に入れられない制御文字
_XML_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# Markdown の表の区切り行 (| --- | :---: |)
_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")


def parse_structured(kind, text):
    """モデルが出した JSON を読み、表 / 文書の形に整える。読めなければ ValueError。"""
    text = (text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 前後に説明文やコードブロックの記号が付いた場合に備え、{ ... } の範囲だけ読み直す
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("JSON が見つかりません")
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON として読めません ({e.msg})") from None
    if not isinstance(data, dict):
        raise ValueError("JSON の形が想定と違います")
    title = str(data.get("title") or "").strip()

    if kind == "table":
        columns = [str(c) for c in data.get("columns") or []]
        if not columns:
            raise ValueError("列見出しがありません")
        rows = []
        for row in data.get("rows") or []:
            cells = [str(c) for c in row] if isinstance(row, list) else [str(row)]
            # セル数を列数にそろえる (多ければ切り、少なければ空欄で埋める)
            rows.append((cells + [""] * len(columns))[:len(columns)])
        return {"title": title, "columns": columns, "rows": rows}

    blocks = []
    for block in data.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        kind_ = block.get("type")
        text_ = str(block.get("text") or "").strip()
        if not text_:
            continue
        if kind_ == "heading":
            blocks.append({"type": "heading", "level": 1, "text": text_})
        elif kind_ == "bullet":
            blocks.append({"type": "bullet", "text": text_})
        else:
            blocks.append({"type": "paragraph", "text": text_})
    if not title and not blocks:
        raise ValueError("本文がありません")
    return {"title": title, "blocks": blocks}


def _markdown_cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def table_to_markdown(table):
    lines = ["| " + " | ".join(_markdown_cell(c) for c in table["columns"]) + " |",
             "| " + " | ".join("---" for _ in table["columns"]) + " |"]
    lines += ["| " + " | ".join(_markdown_cell(c) for c in row) + " |" for row in table["rows"]]
    return "\n".join(lines)


def structured_to_markdown(kind, data):
    """表 / 文書を Markdown にする (画面表示と会話履歴用。後から .md でも保存できる)。"""
    parts = []
    if kind == "table":
        if data["title"]:
            parts.append(f"**{data['title']}**")
        parts.append(table_to_markdown(data))
        return "\n\n".join(parts)
    if data["title"]:
        parts.append(f"# {data['title']}")
    for block in data["blocks"]:
        if block["type"] == "heading":
            parts.append("#" * (block.get("level", 1) + 1) + " " + block["text"])
        elif block["type"] == "bullet":
            parts.append(f"- {block['text']}")
        else:
            parts.append(block["text"])
    return "\n\n".join(parts)


def _split_table_row(line):
    """Markdown の表の 1 行をセルに分ける (\\| はセル内の | として扱う)。"""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells = re.split(r"(?<!\\)\|", line)
    return [_strip_inline(c.strip().replace("\\|", "|")) for c in cells]


def _strip_inline(text):
    """**太字** や `コード` などの記号を外す (Excel / Word には素の文字で入れる)。"""
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def _table_from_lines(lines):
    """Markdown の表の行 (見出し・区切り・本文) から表を作る。形が違えば None。"""
    if len(lines) < 2 or not _TABLE_SEPARATOR_RE.match(lines[1].strip()):
        return None
    columns = _split_table_row(lines[0])
    rows = [(_split_table_row(l) + [""] * len(columns))[:len(columns)] for l in lines[2:]]
    return {"title": "", "columns": columns, "rows": rows}


def extract_tables(text):
    """回答から表を取り出す: Markdown の表 (先) と、```csv / ```tsv のコードブロック (後)。"""
    tables = [b["table"] for b in markdown_to_document(text)["blocks"] if b["type"] == "table"]
    for lang, body in extract_code_blocks(text):
        if lang in ("csv", "tsv"):
            rows = [r for r in csv.reader(io.StringIO(body), delimiter="\t" if lang == "tsv" else ",") if r]
            if rows:
                width = max(len(r) for r in rows)
                tables.append({
                    "title": "", "columns": (rows[0] + [""] * width)[:width],
                    "rows": [(r + [""] * width)[:width] for r in rows[1:]],
                })
    return tables


def markdown_to_document(text):
    """Markdown の回答を文書の形にする (見出し・段落・箇条書き・表・コード)。"""
    blocks = []
    paragraph = []
    lines = (text or "").splitlines()

    def flush():
        if paragraph:
            blocks.append({"type": "paragraph", "text": _strip_inline("\n".join(paragraph))})
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush()
            fence, code = stripped[:3], []
            i += 1
            while i < len(lines) and lines[i].strip() != fence:
                code.append(lines[i])
                i += 1
            blocks.append({"type": "code", "text": "\n".join(code)})
        elif stripped.startswith("|"):
            flush()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            table = _table_from_lines(table_lines)
            if table:
                blocks.append({"type": "table", "table": table})
            else:
                blocks.append({"type": "paragraph", "text": "\n".join(table_lines)})
            continue
        elif re.match(r"^#{1,6}\s", stripped):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append({"type": "heading", "level": level,
                           "text": _strip_inline(stripped[level:].strip())})
        elif re.match(r"^[-*+]\s", stripped):
            flush()
            blocks.append({"type": "bullet", "text": _strip_inline(stripped[2:].strip())})
        elif re.match(r"^\d+[.)]\s", stripped):
            flush()
            blocks.append({"type": "number",
                           "text": _strip_inline(re.sub(r"^\d+[.)]\s", "", stripped))})
        elif not stripped or re.match(r"^(-{3,}|\*{3,})$", stripped):
            flush()
        else:
            paragraph.append(stripped)
        i += 1
    flush()

    # 先頭が見出しなら文書の題名にする
    title = ""
    if blocks and blocks[0]["type"] == "heading" and blocks[0]["level"] == 1:
        title = blocks.pop(0)["text"]
    return {"title": title, "blocks": blocks}


def _excel_value(text):
    """数値に見える文字列は数値に、それ以外はそのまま。"""
    s = str(text).strip()
    if _NUMBER_RE.match(s):
        return float(s) if "." in s else int(s)
    if _GROUPED_NUMBER_RE.match(s):
        s = s.replace(",", "")
        return float(s) if "." in s else int(s)
    return str(text)


def _column_letter(index):
    """0 始まりの列番号 -> A, B, ..., Z, AA, ..."""
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _display_width(text):
    """全角を 2、半角を 1 と数えた幅 (列幅の目安)。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WFA" else 1 for ch in str(text))


def _sheet_name(title, index, used):
    """シート名に使えない文字を除き、31 文字以内で重複しない名前にする。"""
    name = re.sub(r"[\[\]:*?/\\]", "_", title or "").strip()[:31] or f"表{index}"
    base, n = name, 2
    while name in used:
        suffix = f" ({n})"
        name = base[:31 - len(suffix)] + suffix
        n += 1
    used.add(name)
    return name


def write_xlsx(path, tables):
    """表を .xlsx に書く。1 表 1 シート。見出し行は太字・固定、列幅は内容に合わせる。

    利用者の環境には openpyxl が無いため、標準ライブラリ (zipfile) だけで最小構成の
    Office Open XML を組み立てる。
    """
    def esc(value):
        return xml_escape(_XML_ILLEGAL_RE.sub("", str(value)))

    sheets = []
    used = set()
    for n, table in enumerate(tables, 1):
        name = _sheet_name(table.get("title"), n, used)
        rows = [table["columns"]] + table["rows"]
        cells_xml = []
        for r, row in enumerate(rows, 1):
            cells = []
            for c, value in enumerate(row):
                ref = f"{_column_letter(c)}{r}"
                style = ' s="1"' if r == 1 else ""
                v = str(value) if r == 1 else _excel_value(value)
                if isinstance(v, (int, float)):
                    cells.append(f'<c r="{ref}"{style}><v>{v}</v></c>')
                elif v != "":
                    cells.append(f'<c r="{ref}" t="inlineStr"{style}>'
                                 f'<is><t xml:space="preserve">{esc(v)}</t></is></c>')
            cells_xml.append(f'<row r="{r}">{"".join(cells)}</row>')
        widths = [
            min(60, max(8, max((_display_width(row[c]) for row in rows if c < len(row)), default=0) + 2))
            for c in range(len(table["columns"]))
        ]
        cols_xml = "".join(
            f'<col min="{c + 1}" max="{c + 1}" width="{w}" customWidth="1"/>'
            for c, w in enumerate(widths)
        )
        sheets.append((name, (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            '</sheetView></sheetViews>'
            f'<cols>{cols_xml}</cols>'
            f'<sheetData>{"".join(cells_xml)}</sheetData>'
            '</worksheet>'
        )))

    ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        + "".join(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType='
            '"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for i in range(1, len(sheets) + 1)
        )
        + '</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{ns_rel}/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        f' xmlns:r="{ns_rel}"><sheets>'
        + "".join(
            f'<sheet name="{xml_escape(name, {chr(34): "&quot;"})}" sheetId="{i}" r:id="rId{i}"/>'
            for i, (name, _) in enumerate(sheets, 1)
        )
        + '</sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="rId{i}" Type="{ns_rel}/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(sheets) + 1)
        )
        + f'<Relationship Id="rId{len(sheets) + 1}" Type="{ns_rel}/styles" Target="styles.xml"/>'
        '</Relationships>'
    )
    # スタイル 0 = 標準、1 = 見出し行 (太字 + 薄い灰色の背景)
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Yu Gothic"/><family val="2"/></font>'
        '<font><b/><sz val="11"/><name val="Yu Gothic"/><family val="2"/></font></fonts>'
        '<fills count="3"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFE7E6E6"/><bgColor indexed="64"/>'
        '</patternFill></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )
    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        z.writestr("xl/styles.xml", styles)
        for i, (_, sheet) in enumerate(sheets, 1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", sheet)
    return path


def write_docx(path, document):
    """文書を .docx に書く (python-docx を使う)。"""
    try:
        import docx
        from docx.shared import Pt
    except ImportError:
        raise RuntimeError("Word 形式で保存するには python-docx が必要です (pip install python-docx)")
    doc = docx.Document()
    if document.get("title"):
        doc.add_heading(document["title"], level=0)
    for block in document["blocks"]:
        kind = block["type"]
        if kind == "heading":
            doc.add_heading(block["text"], level=min(max(block.get("level", 1), 1), 4))
        elif kind == "bullet":
            doc.add_paragraph(block["text"], style="List Bullet")
        elif kind == "number":
            doc.add_paragraph(block["text"], style="List Number")
        elif kind == "code":
            run = doc.add_paragraph().add_run(block["text"])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif kind == "table":
            table = block["table"]
            grid = doc.add_table(rows=1, cols=len(table["columns"]))
            grid.style = "Table Grid"
            for cell, text in zip(grid.rows[0].cells, table["columns"]):
                cell.paragraphs[0].add_run(str(text)).bold = True
            for row in table["rows"]:
                for cell, text in zip(grid.add_row().cells, row):
                    cell.text = str(text)
        else:
            doc.add_paragraph(block["text"])
    path = Path(path)
    doc.save(str(path))
    return path


def safe_filename(title, fallback="answer"):
    """題名をファイル名に使える形にする。"""
    name = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", title or "").strip(" .")[:40]
    return name or fallback


# --------------------------------------------------------------------------
# スキル (用途ごとに保存しておく指示のセット)
# --------------------------------------------------------------------------
# 1 スキル = {"name", "description", "instructions", "output", "thinking", "template"}
#   instructions: システムプロンプトに足す指示
#   output:       出力形式 (OUTPUT_MODES の表示名。"チャット" / "Excel 表" / "Word 文書")
#   thinking:     思考モード ("そのまま" / "オン" / "オフ")
#   template:     選んだときに入力欄へ入れるひな形 (空なら何もしない)
NO_SKILL = "（なし）"
THINKING_CHOICES = ("そのまま", "オン", "オフ")

# スキルファイルが無いときの初期スキル (事務作業でよく使うもの)
DEFAULT_SKILLS = [
    {
        "name": "議事録まとめ",
        "description": "会議のメモや文字起こしから議事録を作る",
        "instructions": (
            "入力された会議のメモや文字起こしから議事録を作成してください。"
            "見出しは「日時・参加者」「議題」「決定事項」「宿題（担当・期限）」「次回予定」とします。"
            "メモに書かれていない項目は「記載なし」とし、推測で補わないでください。"
        ),
        "output": "チャット",
        "thinking": "そのまま",
        "template": "会議のメモ:\n",
    },
    {
        "name": "メール作成",
        "description": "要件からビジネスメールの文面を作る",
        "instructions": (
            "丁寧なビジネスメールの文面を作成してください。件名と本文を分け、"
            "本文は宛名、挨拶、要件、結びの順に書いてください。簡潔にまとめてください。"
        ),
        "output": "チャット",
        "thinking": "オフ",
        "template": "宛先:\n用件:\n",
    },
    {
        "name": "文章の校正",
        "description": "誤字脱字や不自然な表現を直す",
        "instructions": (
            "入力された文章の誤字脱字と、不自然な表現を直してください。"
            "最初に修正後の全文を示し、そのあとに主な修正点を箇条書きで説明してください。"
            "意味が変わる書き換えはしないでください。"
        ),
        "output": "チャット",
        "thinking": "オフ",
        "template": "",
    },
    {
        "name": "要約",
        "description": "文章や添付ファイルの要点をまとめる",
        "instructions": (
            "入力された文章や添付ファイルの要点を、3〜5 行の箇条書きでまとめてください。"
            "数字・日付・固有名詞は正確に残してください。"
        ),
        "output": "チャット",
        "thinking": "オフ",
        "template": "",
    },
    {
        "name": "表に整理 (Excel)",
        "description": "内容を表に整理して Excel で保存する",
        "instructions": "入力された内容を表に整理してください。列は内容に合わせて決めてください。",
        "output": "Excel 表",
        "thinking": "そのまま",
        "template": "",
    },
]


def normalize_skill(data):
    """読み込んだスキルを正しい形にそろえる。名前が無ければ None。"""
    if not isinstance(data, dict):
        return None
    name = str(data.get("name") or "").strip()
    if not name or name == NO_SKILL:
        return None
    output = data.get("output")
    thinking = data.get("thinking")
    return {
        "name": name,
        "description": str(data.get("description") or "").strip(),
        "instructions": str(data.get("instructions") or "").strip(),
        "output": output if output in OUTPUT_MODES else "チャット",
        "thinking": thinking if thinking in THINKING_CHOICES else "そのまま",
        "template": str(data.get("template") or ""),
    }


def load_skills(path=None):
    """スキルを読む。ファイルが無ければ初期スキル、壊れていれば初期スキル (警告を出す)。"""
    path = Path(path or SKILLS_PATH)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [dict(s) for s in DEFAULT_SKILLS]
    except Exception as e:
        log.warning("[スキル] 読み込みに失敗しました (%s) -> 初期スキルを使います", e)
        return [dict(s) for s in DEFAULT_SKILLS]
    skills, seen = [], set()
    for item in raw.get("skills", []) if isinstance(raw, dict) else []:
        skill = normalize_skill(item)
        if skill and skill["name"] not in seen:     # 同じ名前は先にあるものを使う
            seen.add(skill["name"])
            skills.append(skill)
    return skills


def save_skills(skills, path=None):
    path = Path(path or SKILLS_PATH)
    path.write_text(
        json.dumps({"skills": skills}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("[スキル] 保存: %s (%d 件)", path, len(skills))
    return path


# --------------------------------------------------------------------------
# 会話履歴ストア (1 会話 = 1 JSON ファイル)
# --------------------------------------------------------------------------
class SessionStore:
    def __init__(self, base_dir):
        self.dir = Path(base_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        log.info("[履歴] 保存先: %s", self.dir.resolve())

    def path(self, session_id):
        return self.dir / f"{session_id}.json"

    def save(self, session):
        """session: dict(id, title, created, updated, model, messages)"""
        p = self.path(session["id"])
        p.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("[履歴] 保存: %s (%s, %d発話)", session["id"], session["title"], len(session["messages"]))

    def load(self, session_id):
        data = json.loads(self.path(session_id).read_text(encoding="utf-8"))
        log.info("[履歴] 読み込み: %s (%d発話)", session_id, len(data.get("messages", [])))
        return data

    def delete(self, session_id):
        p = self.path(session_id)
        if p.exists():
            p.unlink()
            log.info("[履歴] 削除: %s", session_id)

    def list_meta(self):
        """一覧用のメタ情報 (id, title, updated) を更新日時の新しい順で返す。"""
        items = []
        for p in self.dir.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                items.append({"id": d["id"], "title": d.get("title", "(無題)"), "updated": d.get("updated", 0)})
            except Exception:
                log.warning("[履歴] 読み込み失敗: %s", p.name)
        items.sort(key=lambda x: x["updated"], reverse=True)
        return items


# --------------------------------------------------------------------------
# 推論エンジン (実モデル / モックの両対応)
# --------------------------------------------------------------------------
class LLMEngine:
    def __init__(self):
        self.llm = None
        self.model_name = None
        self.load_error = None      # 直近のロード失敗理由 (成功/モック時は None)
        self.vision = False         # 画像入力が使えるか (mmproj を読み込めた場合のみ True)
        self.last_finish_reason = None   # 直近の生成の終了理由 ("length" なら打ち切り)
        # 画像用の chat handler (mmproj 使用時)。画像を含まないターンでは外して
        # テキスト用の経路を使う (前回までの計算結果を再利用できるため。stream 参照)
        self._vision_handler = None
        self._text_path = False     # テキスト用の経路に切り替えられるか
        self._kv_from_vision = False    # 直前の生成が画像用の経路だったか
        self.last_path = None       # 直近の生成で使った経路 ("text" / "vision")
        self.speed_options = {}     # 読み込みに使えた速度向上の設定

    def load(self, model_name, n_ctx=DEFAULT_N_CTX, n_threads=DEFAULT_N_THREADS):
        """モデルを読み込む。成功で True、モックで False を返す。

        ファイルはあるが llama.cpp が読めない場合 (破損・未対応アーキテクチャ等)
        は例外をそのまま送出する。呼び出し側で捕捉して UI に伝えること。
        """
        path = MODELS_DIR / model_name
        if not HAS_LLAMA or not path.exists():
            self.llm = None
            self.model_name = model_name
            self.load_error = None
            self.vision = False
            reason = "llama-cpp-python 未導入" if not HAS_LLAMA else "ファイル未検出"
            log.warning("モックモードでロード: %s (%s)", model_name, reason)
            return False

        log.info(
            "モデル読み込み開始: %s (n_ctx=%d, n_threads=%d, n_threads_batch=%d)",
            model_name, n_ctx, n_threads, DEFAULT_N_THREADS_BATCH,
        )
        t0 = time.time()
        chat_handler = self._build_chat_handler()
        base = dict(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_threads_batch=DEFAULT_N_THREADS_BATCH,
            chat_handler=chat_handler,
            verbose=False,
        )
        fast = self._fast_options()
        try:
            try:
                llm = Llama(**base, **fast)
            except Exception:
                if not fast:
                    raise
                # 古いビルド等で速度向上の設定が使えない場合は、外して読み直す
                log.warning("速度向上の設定 %s で読み込めないため、外して読み直します", fast,
                            exc_info=True)
                llm = Llama(**base)
                fast = {}
        except Exception as e:
            # 失敗理由を残しておき、モック応答へ暗黙に落ちないようにする。
            self.llm = None
            self.model_name = None
            self.load_error = str(e)
            self.vision = False
            log.exception("モデル読み込み失敗: %s", model_name)
            raise
        self.llm = llm
        self.model_name = model_name
        self.load_error = None
        self.vision = chat_handler is not None
        self._setup_text_path(chat_handler)
        self.speed_options = fast
        log.info(
            "モデル読み込み完了: %.1f 秒 (画像入力 %s, 速度設定 %s, テキスト経路の切り替え %s)",
            time.time() - t0, "有効" if self.vision else "無効", fast or "なし",
            "可" if self._text_path else "不要" if not self.vision else "不可",
        )
        return True

    @staticmethod
    def _fast_options():
        """Flash Attention と KV キャッシュの量子化の設定。"""
        if not FLASH_ATTN:
            return {}
        options = {"flash_attn": True}
        kv_type = GGML_TYPES.get(KV_CACHE_TYPE)
        if kv_type is None:
            log.warning("LLM_KV_TYPE=%s は未対応です (f16 / q8_0 / q4_0) -> f16", KV_CACHE_TYPE)
        elif KV_CACHE_TYPE != "f16":
            options.update(type_k=kv_type, type_v=kv_type)
        return options

    def _setup_text_path(self, chat_handler):
        """画像用の handler を使う場合に、テキスト用の経路へ切り替えられるようにする。

        llama-cpp-python の画像用 handler (MTMDChatHandler) は、毎回 KV キャッシュを消して
        会話全体を読み直す。テキスト用の経路 (GGUF 内のチャットテンプレート) なら、前回と
        共通する先頭部分の計算結果を再利用し、新しく増えた部分だけを読む。
        両者は同じ tokenizer.chat_template を同じ Jinja 設定で描くので、プロンプトは変わらない。
        """
        self._vision_handler = chat_handler
        self._kv_from_vision = False
        self._text_path = False
        if chat_handler is None:
            return
        handlers = getattr(self.llm, "_chat_handlers", None) or {}
        if "chat_template.default" in handlers:
            # chat_handler を外したときに、こちらのテンプレートが使われるようにしておく
            self.llm.chat_format = "chat_template.default"
            self._text_path = True
        else:
            log.info("GGUF にチャットテンプレートが無いため、常に画像用の経路で生成します")

    @staticmethod
    def _build_chat_handler():
        """mmproj があればマルチモーダル用の chat handler を作る。無ければ None。

        mmproj とモデルが噛み合っているかは実際に生成するまで分からないため、
        ここで作れても画像を送った時点で失敗することはある (その場合は生成エラー)。
        """
        mmproj = discover_mmproj()
        if mmproj is None:
            log.info("mmproj が見つかりません -> 画像入力なしで読み込みます")
            return None
        try:
            from llama_cpp.llama_chat_format import Gemma4ChatHandler

            handler = Gemma4ChatHandler(clip_model_path=str(mmproj), verbose=False)
        except Exception:
            # mmproj が壊れていても、テキストだけは使えるようにして続行する。
            log.exception("mmproj の読み込みに失敗 -> 画像入力なしで続行: %s", mmproj)
            return None
        log.info("mmproj を使用: %s", mmproj.name)
        return handler

    def stream(self, messages, max_tokens, temperature, top_p=DEFAULT_TOP_P, top_k=DEFAULT_TOP_K,
               response_format=None):
        """応答トークンを順次 yield するジェネレータ。

        response_format: {"type": "json_object", "schema": ...} を渡すと、llama-cpp-python が
        スキーマを文法 (GBNF) に変換し、その形の JSON しか出力できないよう縛る。
        チャットテンプレート経路と MTMDChatHandler (Gemma4ChatHandler の親) の両方が対応している。
        """
        self.last_finish_reason = None
        if self.llm is None:
            if self.load_error:
                raise RuntimeError(f"モデルが読み込まれていません: {self.load_error}")
            yield from self._mock_stream(messages, response_format)
            return
        kwargs = dict(
            messages=messages, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, stop=stop_words_for(self.model_name), stream=True,
        )
        if response_format:
            kwargs["response_format"] = response_format

        use_text = (
            self._vision_handler is not None and self._text_path
            and not any(content_images_count(m.get("content")) for m in messages)
        )
        if self._vision_handler is not None:
            if use_text:
                self.llm.chat_handler = None
                if self._kv_from_vision:
                    # 画像用の経路が残した KV キャッシュには画像の埋め込みが混ざっており、
                    # 先頭一致の判定に使えない。一度だけ全体を読み直す
                    self.llm.reset()
                    self._kv_from_vision = False
            else:
                self.llm.chat_handler = self._vision_handler
                self._kv_from_vision = True
        self.last_path = "vision" if self._vision_handler is not None and not use_text else "text"
        log.info("[GEN] 経路: %s", "テキスト (前回までの計算を再利用)" if self.last_path == "text"
                 else "画像用 (会話全体を読み直し)")

        started = False
        try:
            for piece in self._completion_pieces(kwargs):
                started = True
                yield piece
        except Exception:
            if not use_text or started:
                raise
            # テキスト用の経路で描けないテンプレートだった等。以後は画像用の経路だけを使う
            log.warning("テキスト用の経路で生成できないため、画像用の経路に戻します", exc_info=True)
            self._text_path = False
            self.llm.chat_handler = self._vision_handler
            self._kv_from_vision = True
            self.last_path = "vision"
            yield from self._completion_pieces(kwargs)

    def _completion_pieces(self, kwargs):
        completion = self.llm.create_chat_completion(**kwargs)
        for chunk in completion:
            choice = chunk["choices"][0]
            # "length" なら max_tokens 到達で打ち切られている (回答が途中で切れる原因)
            self.last_finish_reason = choice.get("finish_reason") or self.last_finish_reason
            piece = choice["delta"].get("content")
            if piece:
                yield piece

    def count_tokens(self, text):
        """テキストのトークン数。モデル未読み込みなら文字数からの概算。"""
        if not text:
            return 0
        if self.llm is not None:
            try:
                return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))
            except Exception:
                pass
        return max(1, len(text) // 2)       # 日本語は 1 トークン ≒ 1〜2 文字

    def context_budget(self, max_tokens):
        """履歴に使えるトークン数 = コンテキスト長 - 生成枠 - 余白。"""
        n_ctx = DEFAULT_N_CTX
        if self.llm is not None:
            try:
                n_ctx = int(self.llm.n_ctx())
            except Exception:
                pass
        return max(512, n_ctx - max_tokens - CONTEXT_MARGIN_TOKENS)

    def context_usage(self):
        """(使用トークン数, コンテキスト長) を返す。分からなければ None。"""
        if self.llm is None:
            return None
        try:
            used = int(getattr(self.llm, "n_tokens", 0))
            total = int(self.llm.n_ctx())
        except Exception:
            return None
        if total <= 0:
            return None
        return used, total

    @staticmethod
    def _mock_stream(messages, response_format=None):
        """UI 確認用のダミー応答 (1文字ずつ返す)。"""
        user = plain_content(messages[-1]["content"]) if messages else ""
        if response_format:
            # Excel / Word 出力の確認用に、スキーマどおりの JSON を返す
            schema = response_format.get("schema") or {}
            if "columns" in schema.get("properties", {}):
                data = {"title": "モック表", "columns": ["項目", "数量", "備考"],
                        "rows": [["りんご", "3", "青森産"], ["みかん", "12", ""],
                                 ["合計", "1,500", "円"]]}
            else:
                data = {"title": "モック文書", "blocks": [
                    {"type": "heading", "text": "概要"},
                    {"type": "paragraph", "text": f"「{user}」に対するダミーの文書です。"},
                    {"type": "bullet", "text": "一つ目の要点"},
                    {"type": "bullet", "text": "二つ目の要点"}]}
            for ch in json.dumps(data, ensure_ascii=False):
                time.sleep(0.005)
                yield ch
            return
        thinking = any(
            m["role"] == "system" and THINK_TOKEN in plain_content(m["content"])
            for m in messages
        )
        text = (
            f"[モック応答] 受け取りました:「{user}」\n"
            "これは UI 動作確認用のダミー応答です。"
            "実モデルを読み込むと、ここに生成結果がストリーミング表示されます。"
        )
        if thinking:
            # 思考モードの表示確認用に、思考チャネル付きの応答を模擬する
            text = f"{THOUGHT_OPEN}\nユーザーの意図を整理している……\n{THOUGHT_CLOSE}\n{text}"
        for ch in text:
            time.sleep(0.015)
            yield ch


# --------------------------------------------------------------------------
# GUI 本体
# --------------------------------------------------------------------------
def grab_clipboard():
    """クリップボードの画像かファイルを取り出す (Pillow の ImageGrab を使う)。

    戻り値: ("files", [Path, ...]) / ("image", PNG のバイト列) / (None, None)
    Windows ではエクスプローラーでコピーしたファイルと、スクリーンショット等の画像を読める。
    文字はここでは扱わない (Tk の clipboard_get で読む)。
    """
    try:
        from PIL import ImageGrab
    except ImportError:
        raise RuntimeError("クリップボードの画像を読むには Pillow が必要です (pip install pillow)")
    try:
        data = ImageGrab.grabclipboard()
    except Exception as e:      # Linux で xclip / wl-paste が無い場合など
        raise RuntimeError(f"クリップボードを読めません ({e})")
    if data is None:
        return None, None
    if isinstance(data, list):
        return "files", [Path(p) for p in data]
    image = data
    if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "image", buffer.getvalue()


class CameraWindow:
    """カメラのプレビューを出し、撮影した画像を添付として渡す小窓。

    読み取りは別スレッドで回し、UI 側は最新フレームを描くだけにする
    (cap.read() は 30ms 程度ブロックするため、UI スレッドで回すと固まる)。
    """

    def __init__(self, parent, on_capture, index=CAMERA_INDEX):
        self.on_capture = on_capture
        self.capture = open_camera(index)       # 失敗時はここで例外 -> 呼び出し側で表示
        self._frame = None                      # 最新フレーム (別スレッドが更新)
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._photo = None                      # PhotoImage は参照を保持しないと消える

        self.window = tk.Toplevel(parent)
        self.window.title("カメラ")
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.preview = ttk.Label(self.window)
        self.preview.pack(padx=8, pady=8)

        row = ttk.Frame(self.window, padding=(8, 0, 8, 8))
        row.pack(fill="x")
        ttk.Button(row, text="撮影", command=self.capture_frame).pack(side="left")
        self.mirror_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="左右反転", variable=self.mirror_var).pack(side="left", padx=8)
        ttk.Button(row, text="閉じる", command=self.close).pack(side="right")
        self.status_var = tk.StringVar(value="プレビュー中 ...")
        ttk.Label(self.window, textvariable=self.status_var, foreground="#666666").pack(
            anchor="w", padx=8, pady=(0, 8)
        )

        self._reader = threading.Thread(target=self._read_loop, name="camera", daemon=True)
        self._reader.start()
        self.window.after(CAMERA_POLL_MS, self._update_preview)

    def _read_loop(self):
        """カメラから読み続けて、最新フレームだけ持っておく。"""
        while not self._closing.is_set():
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame

    def _current_frame(self):
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        if self.mirror_var.get():
            frame = frame[:, ::-1]              # 左右反転 (プレビューと撮影で揃える)
        return frame

    def _update_preview(self):
        if self._closing.is_set():
            return
        frame = self._current_frame()
        if frame is not None:
            try:
                self._photo = tk.PhotoImage(data=frame_to_preview(frame))
                self.preview.config(image=self._photo)
            except Exception:
                log.exception("[カメラ] プレビューの描画に失敗")
        self.window.after(CAMERA_POLL_MS, self._update_preview)

    def capture_frame(self):
        """今のフレームを PNG にして添付へ渡す。"""
        frame = self._current_frame()
        if frame is None:
            self.status_var.set("まだ映像を取得できていません")
            return
        try:
            data = encode_png(frame)
        except Exception as e:
            log.exception("[カメラ] 撮影に失敗")
            self.status_var.set(f"撮影に失敗しました ({e})")
            return
        name = time.strftime("camera_%Y%m%d_%H%M%S.png")
        log.info("[カメラ] 撮影: %s (%d x %d, %.1f KB)",
                 name, frame.shape[1], frame.shape[0], len(data) / 1024)
        self.on_capture(image_attachment(name, data))
        self.status_var.set(f"添付しました: {name}")

    def close(self):
        if self._closing.is_set():
            return
        self._closing.set()
        self._reader.join(timeout=1.0)
        try:
            self.capture.release()
        except Exception:
            pass
        log.info("[カメラ] 終了")
        self.window.destroy()


def center_on(window, parent):
    """小窓を親ウィンドウの中央に出す (既定だと画面の左上に出ることがある)。"""
    window.update_idletasks()
    w, h = window.winfo_reqwidth(), window.winfo_reqheight()
    x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - h) // 3
    window.geometry(f"+{max(0, x)}+{max(0, y)}")


class SaveChoiceDialog:
    """回答のうち、どこを保存するかを選ぶ小さなダイアログ。

    candidates: [(見出し, プレビュー文字列), ...]。選ばれた添字 (取り消しなら None) を返す。
    """

    def __init__(self, parent, candidates, default=0):
        self.result = None
        self.window = tk.Toplevel(parent)
        self.window.title("回答をファイルに保存")
        self.window.transient(parent)
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)

        body = ttk.Frame(self.window, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="保存する内容を選んでください").pack(anchor="w", pady=(0, 8))
        self.choice = tk.IntVar(value=default)
        for i, (label, preview) in enumerate(candidates):
            ttk.Radiobutton(body, text=label, variable=self.choice, value=i).pack(anchor="w")
            if preview:
                ttk.Label(body, text=preview, style="Muted.TLabel", justify="left").pack(
                    anchor="w", padx=(24, 0), pady=(0, 6)
                )

        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="キャンセル", command=self.cancel).pack(side="right")
        ttk.Button(
            buttons, text="保存先を選ぶ", style="Accent.TButton", command=self.ok
        ).pack(side="right", padx=(0, 6))
        self.window.bind("<Return>", lambda e: self.ok())
        self.window.bind("<Escape>", lambda e: self.cancel())
        center_on(self.window, parent)

    def show(self):
        """閉じられるまで待って結果を返す (モーダル)。"""
        self.window.grab_set()
        self.window.focus_set()
        self.window.wait_window()
        return self.result

    def ok(self):
        self.result = self.choice.get()
        self.window.destroy()

    def cancel(self):
        self.result = None
        self.window.destroy()


class SkillEditor:
    """スキルの追加・編集・削除を行う画面 (モーダル)。

    編集は手元のコピーに対して行い、「保存」で初めてファイルへ書く。
    「キャンセル」なら何も変わらない。
    """

    FIELDS_HINT = (
        "指示はシステムプロンプトに足されます。出力形式と思考モードは、"
        "スキルを選んだときに切り替わります。ひな形は入力欄が空のときに入ります。"
    )

    def __init__(self, parent, skills, selected=None):
        self.skills = [dict(s) for s in skills]
        self.result = None          # 保存したらスキルの一覧、キャンセルなら None
        self.current = None         # 編集中の添字

        self.window = tk.Toplevel(parent)
        self.window.title("スキルの設定")
        self.window.transient(parent)
        self.window.minsize(640, 460)
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)

        body = ttk.Frame(self.window, padding=12)
        body.pack(fill="both", expand=True)

        # 左: 一覧と操作
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="スキル", style="Muted.TLabel").pack(anchor="w")
        self.listbox = tk.Listbox(left, width=22, height=14, exportselection=False,
                                  activestyle="none")
        self.listbox.pack(fill="y", expand=True, pady=(2, 6))
        self.listbox.bind("<<ListboxSelect>>", lambda e: self._on_select())
        buttons = ttk.Frame(left)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="新規", width=-5, command=self.add).pack(side="left")
        ttk.Button(buttons, text="複製", width=-5, command=self.duplicate).pack(side="left", padx=4)
        ttk.Button(buttons, text="削除", width=-5, command=self.delete).pack(side="left")
        order = ttk.Frame(left)
        order.pack(fill="x", pady=(4, 0))
        ttk.Button(order, text="▲ 上へ", width=-6, command=lambda: self.move(-1)).pack(side="left")
        ttk.Button(order, text="▼ 下へ", width=-6, command=lambda: self.move(1)).pack(
            side="left", padx=4
        )

        # 右: 編集欄
        form = ttk.Frame(body, padding=(14, 0, 0, 0))
        form.pack(side="left", fill="both", expand=True)
        form.columnconfigure(1, weight=1)
        self.name_var = tk.StringVar()
        self.desc_var = tk.StringVar()
        self.output_var = tk.StringVar(value="チャット")
        self.thinking_var = tk.StringVar(value="そのまま")

        ttk.Label(form, text="名前").grid(row=0, column=0, sticky="w", pady=2)
        self.name_entry = ttk.Entry(form, textvariable=self.name_var)
        self.name_entry.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(form, text="説明").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(form, textvariable=self.desc_var).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(form, text="指示").grid(row=2, column=0, sticky="nw", pady=2)
        self.instructions = tk.Text(form, height=8, wrap="word", undo=True)
        self.instructions.grid(row=2, column=1, sticky="nsew", pady=2)
        form.rowconfigure(2, weight=1)

        opts = ttk.Frame(form)
        opts.grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(opts, text="出力形式").pack(side="left")
        ttk.Combobox(opts, textvariable=self.output_var, values=list(OUTPUT_MODES),
                     state="readonly", width=9).pack(side="left", padx=(4, 16))
        ttk.Label(opts, text="思考モード").pack(side="left")
        ttk.Combobox(opts, textvariable=self.thinking_var, values=THINKING_CHOICES,
                     state="readonly", width=7).pack(side="left", padx=(4, 0))

        ttk.Label(form, text="ひな形").grid(row=4, column=0, sticky="nw", pady=2)
        self.template = tk.Text(form, height=3, wrap="word", undo=True)
        self.template.grid(row=4, column=1, sticky="ew", pady=2)
        ttk.Label(form, text=self.FIELDS_HINT, style="Muted.TLabel", wraplength=380,
                  justify="left").grid(row=5, column=1, sticky="w", pady=(4, 0))

        self.error_var = tk.StringVar()
        ttk.Label(form, textvariable=self.error_var, foreground=COLORS["error"]).grid(
            row=6, column=1, sticky="w"
        )

        bottom = ttk.Frame(self.window, padding=(12, 0, 12, 12))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="キャンセル", command=self.cancel).pack(side="right")
        ttk.Button(bottom, text="保存", style="Accent.TButton", command=self.save).pack(
            side="right", padx=(0, 6)
        )
        self.window.bind("<Escape>", lambda e: self.cancel())

        self._fill_list()
        names = [s["name"] for s in self.skills]
        self._select(names.index(selected) if selected in names else 0)
        center_on(self.window, parent)

    # ---- 一覧と編集欄の受け渡し -----------------------------------------
    def _fill_list(self):
        self.listbox.delete(0, "end")
        for skill in self.skills:
            self.listbox.insert("end", skill["name"] or "(名前なし)")

    def _select(self, index):
        if not self.skills:
            self.current = None
            self._load_form(None)
            return
        index = max(0, min(index, len(self.skills) - 1))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(index)
        self.listbox.see(index)
        self.current = index
        self._load_form(self.skills[index])

    def _load_form(self, skill):
        skill = skill or {"name": "", "description": "", "instructions": "",
                          "output": "チャット", "thinking": "そのまま", "template": ""}
        self.name_var.set(skill["name"])
        self.desc_var.set(skill["description"])
        self.output_var.set(skill["output"])
        self.thinking_var.set(skill["thinking"])
        for widget, key in ((self.instructions, "instructions"), (self.template, "template")):
            widget.delete("1.0", "end")
            widget.insert("1.0", skill[key])
            widget.edit_reset()
        self.error_var.set("")

    def _store_form(self):
        """編集欄の内容を手元のコピーへ戻す。一覧の表示名も更新する。"""
        if self.current is None:
            return
        skill = self.skills[self.current]
        skill.update(
            name=self.name_var.get().strip(),
            description=self.desc_var.get().strip(),
            instructions=self.instructions.get("1.0", "end-1c").strip(),
            output=self.output_var.get(),
            thinking=self.thinking_var.get(),
            template=self.template.get("1.0", "end-1c"),
        )
        self.listbox.delete(self.current)
        self.listbox.insert(self.current, skill["name"] or "(名前なし)")
        self.listbox.selection_set(self.current)

    def _on_select(self):
        selected = self.listbox.curselection()
        if not selected or selected[0] == self.current:
            return
        self._store_form()
        self._select(selected[0])

    # ---- 操作 ------------------------------------------------------------
    def _unique_name(self, base):
        names = {s["name"] for s in self.skills}
        name, n = base, 2
        while name in names:
            name = f"{base} ({n})"
            n += 1
        return name

    def add(self):
        self._store_form()
        self.skills.append({"name": self._unique_name("新しいスキル"), "description": "",
                            "instructions": "", "output": "チャット", "thinking": "そのまま",
                            "template": ""})
        self._fill_list()
        self._select(len(self.skills) - 1)
        self.name_entry.focus_set()
        self.name_entry.select_range(0, "end")

    def duplicate(self):
        if self.current is None:
            return
        self._store_form()
        copy = dict(self.skills[self.current])
        copy["name"] = self._unique_name(f"{copy['name']} のコピー")
        self.skills.insert(self.current + 1, copy)
        self._fill_list()
        self._select(self.current + 1)

    def delete(self):
        if self.current is None:
            return
        name = self.skills[self.current]["name"]
        if not messagebox.askyesno("スキルの削除", f"「{name}」を削除しますか？",
                                   parent=self.window):
            return
        del self.skills[self.current]
        index = self.current
        self.current = None
        self._fill_list()
        self._select(index)

    def move(self, step):
        if self.current is None:
            return
        target = self.current + step
        if not 0 <= target < len(self.skills):
            return
        self._store_form()
        self.skills[self.current], self.skills[target] = self.skills[target], self.skills[self.current]
        self._fill_list()
        self._select(target)

    def _validate(self):
        """名前の空欄・重複を調べる。問題があればその項目を選んでメッセージを返す。"""
        seen = set()
        for i, skill in enumerate(self.skills):
            name = skill["name"]
            problem = None
            if not name:
                problem = "名前を入力してください"
            elif name == NO_SKILL:
                problem = f"「{NO_SKILL}」は名前に使えません"
            elif name in seen:
                problem = f"「{name}」という名前が重複しています"
            if problem:
                self._select(i)
                return problem
            seen.add(name)
        return None

    def save(self):
        self._store_form()
        problem = self._validate()
        if problem:
            self.error_var.set(problem)
            self.name_entry.focus_set()
            return
        self.result = [normalize_skill(s) for s in self.skills]
        self.window.destroy()

    def cancel(self):
        self.result = None
        self.window.destroy()

    def show(self):
        """閉じられるまで待って結果を返す (モーダル)。"""
        self.window.grab_set()
        self.window.focus_set()
        self.window.wait_window()
        return self.result


def open_path(path):
    """ファイルを既定のアプリで開く (出典のクリック用)。"""
    if os.name == "nt":
        os.startfile(str(path))                     # noqa: (Windows のみ)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class RagWindow:
    """資料検索 (RAG) の設定画面: 資料フォルダと埋め込みモデルの指定、索引の作成、検索テスト。

    索引の作成と検索テストは別スレッドで行い、画面は queue 経由で更新する
    (埋め込みモデルの読み込みや、数百ファイルの索引作成には時間がかかるため)。
    """

    POLL_MS = 100

    def __init__(self, app):
        self.app = app
        self.queue = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False

        self.window = tk.Toplevel(app.root)
        self.window.title("資料検索 (RAG) の設定")
        self.window.transient(app.root)
        self.window.minsize(680, 620)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        body = ttk.Frame(self.window, padding=12)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)

        rows = (
            ("資料フォルダ", app.rag_docs_dir_var, "資料フォルダを選択"),
            ("埋め込みモデル", app.rag_model_dir_var, "埋め込みモデルのフォルダを選択"),
        )
        for r, (label, var, title) in enumerate(rows):
            ttk.Label(body, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(body, textvariable=var).grid(row=r, column=1, sticky="ew", pady=2, padx=(8, 4))
            ttk.Button(body, text="参照", width=-6,
                       command=lambda v=var, t=title: self._choose_dir(v, t)).grid(row=r, column=2)
        ttk.Label(
            body, style="Muted.TLabel", justify="left", wraplength=560,
            text=("資料フォルダの中の PDF / Word / Excel / テキストを検索できるようにします (サブフォルダも含む)。"
                  "埋め込みモデルは sentence-transformers 形式のフォルダを、あらかじめダウンロードして"
                  "置いてください (例: multilingual-e5-small、ruri-v3-30m)。"),
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 8))

        index_box = ttk.LabelFrame(body, text="索引", padding=8)
        index_box.grid(row=3, column=0, columnspan=3, sticky="ew")
        index_box.columnconfigure(0, weight=1)
        self.summary_var = tk.StringVar()
        ttk.Label(index_box, textvariable=self.summary_var).grid(row=0, column=0, columnspan=3, sticky="w")
        self.build_btn = ttk.Button(index_box, text="索引を作成・更新", style="Accent.TButton",
                                    command=self.build)
        self.build_btn.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.cancel_btn = ttk.Button(index_box, text="中止", width=-5, command=self.on_cancel,
                                     state="disabled")
        self.cancel_btn.grid(row=1, column=1, sticky="w", pady=(6, 0), padx=(6, 0))
        self.progress = ttk.Progressbar(index_box, mode="determinate", maximum=1,
                                        style="Accent.Horizontal.TProgressbar")
        self.progress.grid(row=1, column=2, sticky="ew", pady=(6, 0), padx=(10, 0))
        index_box.columnconfigure(2, weight=1)
        index_box.columnconfigure(0, weight=0)
        self.progress_var = tk.StringVar(value="変更のあったファイルだけを読み直します")
        ttk.Label(index_box, textvariable=self.progress_var, style="Muted.TLabel").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )

        # 操作の記録 (何をしているか・何が起きたかを時刻付きで残す)
        log_box = ttk.LabelFrame(body, text="ログ", padding=8)
        log_box.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_box, height=8, wrap="char", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.tag_config("time", foreground=COLORS["muted"])
        self.log_text.tag_config("warn", foreground=COLORS["error"])
        self.log_text.tag_config("done", foreground=COLORS["ok"])
        log_tools = ttk.Frame(log_box)
        log_tools.grid(row=1, column=0, sticky="e", pady=(4, 0))
        ttk.Button(log_tools, text="コピー", width=-6, command=self._copy_log).pack(side="left")
        ttk.Button(log_tools, text="クリア", width=-6, command=self._clear_log).pack(
            side="left", padx=(6, 0))

        test_box = ttk.LabelFrame(body, text="検索テスト (どの資料が見つかるかを確かめる)", padding=8)
        test_box.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        body.rowconfigure(4, weight=1)
        body.rowconfigure(5, weight=1)
        test_box.columnconfigure(0, weight=1)
        test_box.rowconfigure(1, weight=1)
        self.query_var = tk.StringVar()
        query = ttk.Entry(test_box, textvariable=self.query_var)
        query.grid(row=0, column=0, sticky="ew")
        query.bind("<Return>", lambda e: self.search_test())
        self.search_btn = ttk.Button(test_box, text="検索", width=-5, command=self.search_test)
        self.search_btn.grid(row=0, column=1, padx=(6, 0))
        self.results = scrolledtext.ScrolledText(test_box, height=7, wrap="word", state="disabled")
        self.results.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        self.results.tag_config("head", foreground=COLORS["assistant_head"])

        bottom = ttk.Frame(self.window, padding=(12, 0, 12, 12))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="閉じる", command=self.close).pack(side="right")

        self._build_started = None      # 索引作成の開始時刻 (残り時間の見積もり用)
        self._refresh_summary()
        self._log("設定画面を開きました")
        self._log(f"索引: {self.summary_var.get()}")
        for label, var, _ in rows:
            self._log(f"{label}: {var.get().strip() or '(未設定)'}")
        center_on(self.window, app.root)
        self.window.after(self.POLL_MS, self._poll)

    # ---- ログ ------------------------------------------------------------
    def _log(self, message, tag=None, echo=True):
        """ログ欄に時刻付きで 1 行足す (UI スレッドから)。

        echo=True なら同じ内容をロガーにも出す (rag 側で既にロガーへ出したものは False)。
        """
        if echo:
            log.info("[RAG画面] %s", message)
        if tag is None and ("読めません" in message or "エラー" in message
                            or "ありません" in message or "中止" in message):
            tag = "warn"
        self.log_text.config(state="normal")
        self.log_text.insert("end", time.strftime("%H:%M:%S "), "time")
        self.log_text.insert("end", message + "\n", tag)
        self.log_text.config(state="disabled")
        self.log_text.see("end")

    def _copy_log(self):
        self.window.clipboard_clear()
        self.window.clipboard_append(self.log_text.get("1.0", "end-1c"))
        self.progress_var.set("ログをクリップボードにコピーしました")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _status(self, message, tag=None, echo=True):
        """状態の 1 行とログの両方に出す。"""
        self.progress_var.set(message)
        self._log(message, tag, echo)

    # ---- 小物 ------------------------------------------------------------
    def _choose_dir(self, var, title):
        chosen = filedialog.askdirectory(title=title, initialdir=var.get() or str(Path.home()),
                                         parent=self.window)
        if chosen:
            var.set(chosen)
            self._log(f"{title.replace('を選択', '')}を変更: {chosen}")
            if var is self.app.rag_docs_dir_var:
                self._log_folder_contents(chosen)

    def _log_folder_contents(self, folder):
        """資料フォルダを選んだときに、索引の対象になるファイル数をすぐ見せる。"""
        try:
            import rag

            count = sum(1 for _ in rag.iter_documents(Path(folder)))
        except Exception as e:
            self._log(f"資料フォルダを調べられません: {e}")
            return
        if count:
            self._log(f"索引の対象になるファイル: {count} 件 (「索引を作成・更新」で取り込みます)")
        else:
            self._log("対象のファイルがありません (PDF / Word / Excel / テキストを置いてください)")

    def _refresh_summary(self):
        index = self.app.rag_index
        self.summary_var.set(index.summary() if index is not None else
                             "資料検索のライブラリを読み込めません (numpy が必要です)")

    def _set_busy(self, busy):
        self.busy = busy
        self.window.config(cursor="watch" if busy else "")
        state = "disabled" if busy else "normal"
        self.build_btn.config(state=state)
        self.search_btn.config(state=state)
        self.cancel_btn.config(state="normal" if busy else "disabled")

    def _show_results(self, text_parts):
        self.results.config(state="normal")
        self.results.delete("1.0", "end")
        for text, tag in text_parts:
            self.results.insert("end", text, tag)
        self.results.config(state="disabled")

    def _check_inputs(self, need_docs):
        docs = self.app.rag_docs_dir_var.get().strip()
        model = self.app.rag_model_dir_var.get().strip()
        if self.app.rag_index is None:
            return "資料検索のライブラリを読み込めません (numpy が必要です)"
        if need_docs and not (docs and Path(docs).is_dir()):
            return "資料フォルダを選んでください"
        if not model:
            return "埋め込みモデルのフォルダを選んでください"
        return None

    # ---- 索引の作成 ------------------------------------------------------
    def build(self):
        problem = self._check_inputs(need_docs=True)
        if problem:
            self._status(problem, "warn")
            return
        self.cancel.clear()
        self._set_busy(True)
        self.progress.config(value=0, maximum=1)
        self._build_started = None
        self._log("―― 索引の作成・更新を開始 ――", "done")
        self.progress_var.set("準備中")
        threading.Thread(target=self._build_worker, name="rag-build", daemon=True,
                         args=(self.app.rag_docs_dir_var.get().strip(),
                               self.app.rag_model_dir_var.get().strip())).start()

    def _load_embedder(self, model_dir):
        """(ワーカースレッド) 埋め込みモデルを用意する。初回の読み込みは時間がかかるので知らせる。"""
        if self.app._embedder is not None and self.app._embedder_key == model_dir:
            return self.app._rag_embedder(model_dir)
        self.queue.put(("status", f"埋め込みモデルを読み込んでいます: {Path(model_dir).name} "
                                  "(初回は数秒〜数十秒かかります)"))
        embedder = self.app._rag_embedder(model_dir)
        seconds = getattr(embedder, "load_seconds", None)
        self.queue.put(("log", "埋め込みモデルを読み込みました"
                               + (f" ({seconds:.1f} 秒)" if seconds is not None else "")))
        return embedder

    def _build_worker(self, docs_dir, model_dir):
        try:
            embedder = self._load_embedder(model_dir)
            with self.app._rag_lock:
                stats = self.app.rag_index.build(
                    docs_dir, embedder, lambda p: extract_document_text(p, max_chars=None),
                    progress=lambda n, total, name: self.queue.put(("progress", (n, total, name))),
                    cancel=self.cancel,
                    report=lambda message: self.queue.put(("log", message)),
                )
            self.queue.put(("built", stats))
        except Exception as e:
            log.exception("[RAG] 索引の作成に失敗")
            self.queue.put(("failed", str(e)))

    # ---- 検索テスト ------------------------------------------------------
    def search_test(self):
        query = self.query_var.get().strip()
        problem = self._check_inputs(need_docs=False)
        if not query or problem:
            self._status(problem or "検索する文を入力してください", "warn")
            return
        if not self.app.rag_index.ready:
            self._status("索引がありません。先に索引を作成してください", "warn")
            return
        self._set_busy(True)
        self._status(f"検索テスト: 「{query}」")
        threading.Thread(target=self._search_worker, name="rag-test", daemon=True,
                         args=(query, self.app.rag_model_dir_var.get().strip())).start()

    def _search_worker(self, query, model_dir):
        try:
            embedder = self._load_embedder(model_dir)
            t0 = time.time()
            with self.app._rag_lock:
                hits = self.app.rag_index.search(query, embedder, top_k=5, max_chars=10 ** 9)
            self.queue.put(("searched", (hits, time.time() - t0)))
        except Exception as e:
            log.exception("[RAG] 検索テストに失敗")
            self.queue.put(("failed", str(e)))

    # ---- 画面の更新 ------------------------------------------------------
    def _poll(self):
        if not self.window.winfo_exists():
            return
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    self._show_progress(*payload)
                elif kind == "log":             # rag 側でロガーにも出している
                    self._log(payload, echo=False)
                elif kind == "status":
                    self._status(payload, echo=False)
                elif kind == "built":
                    self._set_busy(False)
                    s = payload
                    if s["cancelled"]:
                        text = "中止しました (それまでの索引はそのまま残ります)"
                    else:
                        self.progress.config(value=self.progress.cget("maximum"))
                        text = (f"完了 ({s['seconds']:.0f} 秒): 追加 {s['added']} / 更新 {s['updated']} / "
                                f"削除 {s['removed']} / 変更なし {s['unchanged']}")
                    if s["failed"]:
                        text += f" / 読めないファイル {len(s['failed'])}"
                        self._show_results(
                            [("読めなかったファイル:\n", "head")]
                            + [(f"・{name} ({reason})\n", None) for name, reason in s["failed"]]
                        )
                    self.progress_var.set(text)
                    self._log(text, "warn" if s["cancelled"] or s["failed"] else "done")
                    self._refresh_summary()
                    self.app._refresh_rag_status()
                elif kind == "searched":
                    self._set_busy(False)
                    hits, seconds = payload
                    if hits:
                        best = hits[0]
                        text = (f"{len(hits)} 件見つかりました ({seconds:.2f} 秒)。"
                                f"最上位 {best['score']:.3f}: {rag_source_label(best)}")
                    else:
                        text = f"見つかりませんでした ({seconds:.2f} 秒)"
                    self._status(text, "done" if hits else "warn")
                    parts = []
                    for hit in hits:
                        parts.append((f"[{hit['no']}] {hit['score']:.3f}  {rag_source_label(hit)}\n", "head"))
                        parts.append((hit["text"] + "\n\n", None))
                    self._show_results(parts or [("見つかりませんでした", None)])
                elif kind == "failed":
                    self._set_busy(False)
                    self._status(f"エラー: {payload}", "warn")
        except queue.Empty:
            pass
        self.window.after(self.POLL_MS, self._poll)

    def _show_progress(self, n, total, name):
        """進み具合 (件数・経過時間・残りの見積もり) をプログレスバーと状態の行に出す。"""
        now = time.time()
        if self._build_started is None or n == 1:
            self._build_started = now
        self.progress.config(maximum=max(total, 1), value=n - 1)
        elapsed = now - self._build_started
        text = f"[{n}/{total}] {name}  経過 {elapsed:.0f} 秒"
        if n > 3 and elapsed >= 5:
            remaining = elapsed / (n - 1) * (total - n + 1)
            text += f" / 残り約 {remaining:.0f} 秒"
        self.progress_var.set(text)

    def on_cancel(self):
        self.cancel.set()
        self.cancel_btn.config(state="disabled")
        self._status("中止します (今のファイルが終わったところで止まります)", "warn")

    def close(self):
        if self.busy:
            self.cancel.set()           # 索引の作成は次のファイルの区切りで止まる
            log.info("[RAG画面] 作業中に閉じたため中止します")
        self.app._refresh_rag_status()
        self.window.destroy()


def rag_source_label(hit):
    """出典の表示 ("規程/就業規則.pdf p.3")。rag.source_label と同じ (rag を読まずに使えるように)。"""
    return f"{hit['file']} {hit['location']}".strip()


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.engine = LLMEngine()
        self.store = SessionStore(SESSIONS_DIR)

        self.history = []              # [{"role": ..., "content": ...}]
        self.attachments = []          # 次の送信に添付するファイル [{kind, name, ...}]
        self.camera_window = None      # 開いているカメラ小窓 (無ければ None)
        self.current_id = None         # 現在の会話 ID (未保存なら None)
        self.current_created = None    # 現在の会話の作成時刻
        self._dirty = False            # 前回保存から会話が変わったか (開くだけでは並び順を変えない)

        self.token_queue = queue.Queue()
        self.generating = False
        self._stop_event = threading.Event()   # 生成の打ち切り要求
        self._assistant_buf = ""       # ストリーミング中のアシスタント発話バッファ
        self._in_thought = False       # 思考チャネルを表示中か
        self._answer_started = False   # 回答の最初のトークンを出したか
        self._thumbnails = []          # チャット欄に貼った画像 (GC されると消えるため保持)
        self._structured = None        # 生成中の回答が Excel / Word 出力なら "table" / "document"
        self._placeholder_on = False   # 入力欄に案内文を表示中か

        # モデルフォルダは UI (モデル一覧) を組み立てる前に確定させる
        saved_dir = load_settings_file().get("models_dir")
        if saved_dir and MODELS_DIR_FROM_ENV:
            log.info("[設定] LLM_MODELS_DIR が指定されているため、保存済みのフォルダは使いません")
        elif saved_dir:
            set_models_dir(saved_dir)

        self.skills = load_skills()    # スキル (用途ごとの指示のセット)
        # 資料検索 (RAG)。索引は起動時に読み、埋め込みモデルは初めて使うときに読む
        self.rag_index = None
        self._rag_lock = threading.Lock()       # 索引の作成と検索を同時に行わない
        self._embedder = None
        self._embedder_key = None
        self._embed_lock = threading.Lock()
        self._pending_sources = None            # 生成中の回答に付ける出典
        self._link_count = 0                    # 出典リンクのタグ名に使う連番

        self._build_ui()
        self._load_settings()
        self._load_rag_index()
        self._refresh_sidebar()
        self._show_welcome()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(POLL_INTERVAL_MS, self._poll_queue)
        log.info(
            "アプリ起動完了 (CPUスレッド=%d, モデルフォルダ=%s)",
            DEFAULT_N_THREADS, MODELS_DIR.resolve(),
        )

    # ---- UI 構築 ---------------------------------------------------------
    # 画面構成:
    #   メニューバー
    #   ┌ サイドバー ─┬ ヘッダー (モデル選択・状態・詳細設定の開閉) ─────┐
    #   │ 新規チャット │ 詳細設定パネル (折りたたみ)                      │
    #   │ 検索         │ チャット欄                                       │
    #   │ 会話一覧     │ 入力欄 (ツール行 / 添付一覧 / テキスト + 送信)    │
    #   └──────────────┴──────────────────────────────────────────────────┘
    #   ステータスバー (状態・進行表示・コンテキスト使用量)
    def _build_ui(self):
        self.root.title("ローカルLLM チャット (CPU / オフライン)")
        self.root.geometry("1100x720")
        self.root.minsize(840, 540)

        self._setup_style()
        self._build_vars()
        self._build_menu()
        # ステータスバーは先に下端へ置く (後から pack すると縮小時に押し出される)
        self._build_statusbar()

        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, padding=(8, 8, 4, 8))
        right = ttk.Frame(paned, padding=(4, 8, 8, 4))
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        self._build_sidebar(left)
        self._build_header(right)
        self._build_settings_panel(right)
        # 入力欄はチャット欄より先に下端へ置く (狭いときに入力欄が押し出されないように)
        self._build_composer(right)
        self._build_chat_view(right)
        self._bind_shortcuts()

    def _setup_style(self):
        """フォントとテーマを整える。"""
        self.style = ttk.Style(self.root)
        # Linux の既定テーマ ("default") は古びて見えるので clam にする。Windows / macOS はネイティブのまま
        if self.style.theme_use() == "default" and "clam" in self.style.theme_names():
            self.style.theme_use("clam")

        family = None
        try:
            family = pick_ui_font(tkfont.families(self.root))
        except Exception:
            log.debug("[表示] フォント一覧を取得できませんでした", exc_info=True)
        if family:
            # ttk ウィジェット・メニュー等が使う既定フォントをまとめて差し替える
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
                try:
                    tkfont.nametofont(name).configure(family=family)
                except Exception:
                    pass
        else:
            family = tkfont.nametofont("TkDefaultFont").actual("family")
        log.info("[表示] フォント: %s", family)

        # チャット欄・入力欄のフォント。文字サイズの変更は _set_font_size でまとめて行う
        self.fonts = {
            "chat": tkfont.Font(root=self.root, family=family, size=DEFAULT_FONT_SIZE),
            "chat_bold": tkfont.Font(root=self.root, family=family, size=DEFAULT_FONT_SIZE, weight="bold"),
            "small": tkfont.Font(root=self.root, family=family, size=DEFAULT_FONT_SIZE - 2),
            "small_italic": tkfont.Font(
                root=self.root, family=family, size=DEFAULT_FONT_SIZE - 2, slant="italic"
            ),
            "ui_bold": tkfont.Font(root=self.root, family=family, size=10, weight="bold"),
        }

        self.style.configure("Muted.TLabel", foreground=COLORS["muted"])
        # 入力欄の中に出す案内文 (灰色の文字)
        self.style.configure("Hint.TEntry", foreground=COLORS["muted"])
        self.style.configure("Accent.TButton", font=self.fonts["ui_bold"])
        # clam の進捗バーは地の色と見分けにくいので、進んだ部分を強調色にする
        self.style.configure("Accent.Horizontal.TProgressbar", background=COLORS["accent"])
        for kind in ("ok", "warn", "error", "muted"):
            self.style.configure(f"State{kind.title()}.TLabel", foreground=COLORS[kind])
        # 行の高さは文字の高さに合わせる (固定値だと高 DPI や大きいフォントで文字が切れる)
        linespace = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        self.style.configure("Treeview", rowheight=max(24, linespace + 8))

    def _build_vars(self):
        """UI の状態を持つ変数。設定ファイルとの読み書きにも使う。"""
        self.model_var = tk.StringVar()
        self.temp_var = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        self.top_p_var = tk.DoubleVar(value=DEFAULT_TOP_P)
        self.top_k_var = tk.IntVar(value=DEFAULT_TOP_K)
        self.maxtok_var = tk.IntVar(value=DEFAULT_MAX_TOKENS)
        self.system_var = tk.StringVar(value="")
        # 思考モード: システムプロンプト先頭に <|think|> を付けて有効化する
        self.thinking_var = tk.BooleanVar(value=False)
        self.show_thought_var = tk.BooleanVar(value=True)
        # PDF は既定でテキスト優先 (抽出できなければ自動で画像化)。
        # チェックすると常にページ画像として渡す (図表やレイアウトを見せたいとき)。
        self.pdf_as_image_var = tk.BooleanVar(value=False)
        self.show_settings_var = tk.BooleanVar(value=False)
        self.font_size_var = tk.IntVar(value=DEFAULT_FONT_SIZE)
        self.save_dir_var = tk.StringVar(value="")      # 最後に回答を保存したフォルダ
        # 回答の形式。Excel 表 / Word 文書を選ぶと、モデルの出力を JSON に縛って書き出す
        self.output_mode_var = tk.StringVar(value="チャット")
        # 使うスキルの名前 (NO_SKILL なら使わない) と、その説明
        self.skill_var = tk.StringVar(value=NO_SKILL)
        self.skill_desc_var = tk.StringVar(value="")
        # 資料検索 (RAG)
        self.rag_enabled_var = tk.BooleanVar(value=False)
        self.rag_docs_dir_var = tk.StringVar(value="")
        self.rag_model_dir_var = tk.StringVar(value="")
        self.rag_status_var = tk.StringVar(value="")
        self.search_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="準備完了")
        self.model_state_var = tk.StringVar(value="● 未読み込み")
        self.ctx_var = tk.StringVar(value="")
        self.attach_var = tk.StringVar(value="")

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="新規チャット", accelerator="Ctrl+N", command=self.on_new_chat)
        file_menu.add_separator()
        file_menu.add_command(label="ファイルを添付...", accelerator="Ctrl+O", command=self.on_attach)
        file_menu.add_command(label="クリップボードから添付", command=self.on_paste_attach)
        file_menu.add_command(label="カメラから取り込む...", command=self.on_camera)
        file_menu.add_separator()
        file_menu.add_command(label="モデルフォルダを選ぶ...", command=self.on_choose_models_dir)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self.on_close)
        menubar.add_cascade(label="ファイル", menu=file_menu)

        chat_menu = tk.Menu(menubar, tearoff=False)
        chat_menu.add_command(label="停止", accelerator="Esc", command=self.on_stop)
        chat_menu.add_command(label="再生成", accelerator="Ctrl+R", command=self.on_regenerate)
        chat_menu.add_command(label="最後の回答をコピー", accelerator="Ctrl+Shift+C",
                              command=self.copy_last_answer)
        chat_menu.add_command(label="最後の回答をファイルに保存...", accelerator="Ctrl+S",
                              command=self.on_save_answer)
        chat_menu.add_separator()
        chat_menu.add_command(label="スキルの設定...", command=self.on_edit_skills)
        chat_menu.add_checkbutton(label="資料を検索して答える", variable=self.rag_enabled_var,
                                  command=self._on_rag_toggle)
        chat_menu.add_command(label="資料検索の設定...", command=self.on_rag_settings)
        chat_menu.add_separator()
        chat_menu.add_checkbutton(label="思考モード", variable=self.thinking_var)
        chat_menu.add_separator()
        for label in OUTPUT_MODES:
            chat_menu.add_radiobutton(
                label=f"出力: {label}", value=label, variable=self.output_mode_var,
                command=self._on_output_mode,
            )
        menubar.add_cascade(label="チャット", menu=chat_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_checkbutton(label="詳細設定を表示", variable=self.show_settings_var,
                                  command=self._apply_settings_visibility)
        view_menu.add_checkbutton(label="思考を表示", variable=self.show_thought_var,
                                  command=self._apply_thought_visibility)
        view_menu.add_separator()
        view_menu.add_command(label="文字を大きく", accelerator="Ctrl++",
                              command=lambda: self._zoom(1))
        view_menu.add_command(label="文字を小さく", accelerator="Ctrl+-",
                              command=lambda: self._zoom(-1))
        view_menu.add_command(label="標準の大きさ", accelerator="Ctrl+0",
                              command=lambda: self._set_font_size(DEFAULT_FONT_SIZE))
        menubar.add_cascade(label="表示", menu=view_menu)
        self.root.config(menu=menubar)

    def _build_statusbar(self):
        bar = ttk.Frame(self.root, padding=(10, 3))
        bar.pack(side="bottom", fill="x")
        ttk.Separator(self.root, orient="horizontal").pack(side="bottom", fill="x")
        # モデルの状態 (未読み込み / 読み込み済み / モック / 失敗) を色付きで常に出しておく
        self.model_state_label = ttk.Label(
            bar, textvariable=self.model_state_var, style="StateMuted.TLabel"
        )
        self.model_state_label.pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(bar, textvariable=self.status_var).pack(side="left")
        # 読み込み・生成中だけ動かす (待っていることが分かるように)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self.ctx_label = ttk.Label(bar, textvariable=self.ctx_var, style="Muted.TLabel")
        self.ctx_label.pack(side="right")

    def _build_sidebar(self, parent):
        ttk.Button(
            parent, text="＋ 新規チャット", style="Accent.TButton", command=self.on_new_chat
        ).pack(fill="x")

        # スキル: 選ぶと指示・出力形式・思考モード・ひな形がまとめて切り替わる
        ttk.Label(parent, text="スキル", style="Muted.TLabel").pack(anchor="w", pady=(12, 2))
        skill_row = ttk.Frame(parent)
        skill_row.pack(fill="x")
        ttk.Button(skill_row, text="設定", width=-5, command=self.on_edit_skills).pack(
            side="right", padx=(4, 0)
        )
        self.skill_combo = ttk.Combobox(
            skill_row, textvariable=self.skill_var, state="readonly", width=16,
        )
        self.skill_combo.pack(side="left", fill="x", expand=True)
        self.skill_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_skill())
        self.skill_desc_label = ttk.Label(
            parent, textvariable=self.skill_desc_var, style="Muted.TLabel",
            wraplength=220, justify="left",
        )
        self.skill_desc_label.pack(anchor="w", pady=(2, 0))
        self._refresh_skill_choices()

        # 資料検索 (RAG): オンにすると、送信のたびに資料から関係する部分を探して回答に使う
        ttk.Label(parent, text="資料検索", style="Muted.TLabel").pack(anchor="w", pady=(12, 2))
        rag_row = ttk.Frame(parent)
        rag_row.pack(fill="x")
        ttk.Button(rag_row, text="設定", width=-5, command=self.on_rag_settings).pack(
            side="right", padx=(4, 0)
        )
        ttk.Checkbutton(
            rag_row, text="資料を検索して答える", variable=self.rag_enabled_var,
            command=self._on_rag_toggle,
        ).pack(side="left")
        ttk.Label(parent, textvariable=self.rag_status_var, style="Muted.TLabel",
                  wraplength=220, justify="left").pack(anchor="w", pady=(2, 0))

        ttk.Label(parent, text="会話履歴", style="Muted.TLabel").pack(anchor="w", pady=(12, 2))
        # 表示用の変数と、絞り込みに使う本当の値 (search_var) を分けておく
        # (案内文を入力欄の中に出すため。案内文で絞り込まないように)
        search_text = tk.StringVar(value="")
        search = ttk.Entry(parent, textvariable=search_text)
        search.pack(fill="x", pady=(0, 4))
        self.search_var.trace_add("write", lambda *a: self._refresh_sidebar())
        self._add_entry_hint(search, search_text, self.search_var, "タイトルで絞り込み")

        wrap = ttk.Frame(parent)
        wrap.pack(fill="both", expand=True)
        # 単一選択で会話を開く。Ctrl / Shift + クリックで複数選んでまとめて削除できる
        self.session_tree = ttk.Treeview(
            wrap, columns=("updated",), show="tree", selectmode="extended"
        )
        self.session_tree.column("#0", width=150, stretch=True)
        self.session_tree.column("updated", width=70, anchor="e", stretch=False)
        self.session_tree.tag_configure("current", font=self.fonts["ui_bold"])
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.session_tree.yview)
        self.session_tree.configure(yscrollcommand=vsb.set)
        self.session_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.session_tree.bind("<<TreeviewSelect>>", self._on_session_select)
        self.session_tree.bind("<Delete>", lambda e: self.on_delete_selected())
        self.session_tree.bind(self._context_click(), self._show_session_menu)

        self.session_menu = tk.Menu(self.root, tearoff=False)
        self.session_menu.add_command(label="開く", command=self._open_selected_session)
        self.session_menu.add_command(label="削除", command=self.on_delete_selected)

        self.delete_btn = ttk.Button(
            parent, text="選択した会話を削除", command=self.on_delete_selected, state="disabled"
        )
        self.delete_btn.pack(fill="x", pady=(6, 0))
        ttk.Label(
            parent, text="Ctrl / Shift + クリックで複数選択", style="Muted.TLabel"
        ).pack(anchor="w", pady=(2, 0))

    def _build_header(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x")
        # 幅が足りないときはモデル一覧が縮む (右端のボタンが切れないように)
        top.columnconfigure(1, weight=1, minsize=160)
        ttk.Label(top, text="モデル").grid(row=0, column=0, padx=(0, 6))
        self.model_combo = ttk.Combobox(
            top, textvariable=self.model_var, state="readonly", width=24
        )
        self.model_combo["values"] = discover_models()
        if self.model_combo["values"]:
            self.model_combo.current(0)
        self.model_combo.grid(row=0, column=1, sticky="ew")
        # ttk ボタンの既定幅 (11 文字) は広すぎるので、負の値 (= 最小幅) で詰める
        self.load_btn = ttk.Button(top, text="読み込み", width=-8, command=self.on_load)
        self.load_btn.grid(row=0, column=2, padx=(4, 0))
        ttk.Button(top, text="フォルダ", width=-8, command=self.on_choose_models_dir).grid(
            row=0, column=3, padx=(4, 0)
        )
        self.settings_btn = ttk.Button(top, width=-7, command=self.toggle_settings)
        self.settings_btn.grid(row=0, column=4, padx=(10, 0))

    def _build_settings_panel(self, parent):
        """サンプリングとシステムプロンプト。普段は畳んでおき、必要なときだけ開く。"""
        self.settings_panel = ttk.LabelFrame(parent, text="詳細設定", padding=(10, 6))
        panel = self.settings_panel
        panel.columnconfigure(1, weight=1)

        ttk.Label(panel, text="システムプロンプト").grid(row=0, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.system_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )

        # 狭い画面でも切れないよう、数値は 2 列ずつ並べる
        nums = ttk.Frame(panel)
        nums.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        spins = (
            ("temperature", self.temp_var, 0.0, 2.0, 0.1),
            ("top_p", self.top_p_var, 0.0, 1.0, 0.05),
            ("top_k", self.top_k_var, 1, 200, 1),
            ("max_tokens", self.maxtok_var, 16, 8192, 16),
        )
        for i, (label, var, lo, hi, step) in enumerate(spins):
            row, col = divmod(i, 2)
            ttk.Label(nums, text=label).grid(row=row, column=col * 2, sticky="w", pady=2)
            ttk.Spinbox(
                nums, from_=lo, to=hi, increment=step, width=6, textvariable=var
            ).grid(row=row, column=col * 2 + 1, sticky="w", padx=(6, 12), pady=2)
        nums.columnconfigure(4, weight=1)
        ttk.Button(nums, text="既定値に戻す", width=-10, command=self.reset_sampling).grid(
            row=0, column=5, sticky="e"
        )
        ttk.Checkbutton(
            nums, text="PDFを画像として読む", variable=self.pdf_as_image_var
        ).grid(row=1, column=5, sticky="e")

        hint = ttk.Label(
            panel,
            text=("既定値は Gemma 4 の推奨値です。max_tokens は回答に使える長さで、"
                  "思考モードでは思考のぶんが自動で上乗せされます"),
            style="Muted.TLabel", justify="left", wraplength=360,
        )
        hint.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        # 折り返し幅をパネルの幅に合わせる
        panel.bind("<Configure>", lambda e: hint.config(wraplength=max(200, e.width - 40)))

    def _build_chat_view(self, parent):
        self.chat_frame = ttk.Frame(parent)
        self.chat_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.chat = scrolledtext.ScrolledText(
            self.chat_frame, wrap="word", state="disabled", font=self.fonts["chat"], width=40,
            relief="flat", borderwidth=0, padx=16, pady=12,
            background=COLORS["bg"], foreground=COLORS["text"],
            highlightthickness=1, highlightbackground=COLORS["border"],
            highlightcolor=COLORS["border"],
        )
        self.chat.pack(fill="both", expand=True)
        chat = self.chat
        chat.tag_config("user_head", foreground=COLORS["user_head"], font=self.fonts["chat_bold"],
                        spacing1=14, spacing3=4)
        chat.tag_config("assistant_head", foreground=COLORS["assistant_head"],
                        font=self.fonts["chat_bold"], spacing1=14, spacing3=4)
        # 自分の発話は背景色で区別する (行末の改行まで含めると右端まで塗られる)
        chat.tag_config("user", background=COLORS["user_bg"], lmargin1=10, lmargin2=10,
                        rmargin=10, spacing2=3)
        chat.tag_config("assistant", lmargin1=10, lmargin2=10, rmargin=10, spacing2=3)
        chat.tag_config("thought", foreground=COLORS["thought"], font=self.fonts["small_italic"],
                        lmargin1=22, lmargin2=22, spacing2=2)
        chat.tag_config("thought_head", font=self.fonts["small"], spacing1=4, spacing3=2)
        # Excel / Word 出力の生成中に流れてくる JSON (完成したら表・文書の表示に置き換える)
        chat.tag_config("json_raw", foreground=COLORS["muted"], font=self.fonts["small"],
                        lmargin1=10, lmargin2=10)
        # 資料検索の出典 (クリックで資料を開く)
        chat.tag_config("source", foreground=COLORS["muted"], font=self.fonts["small"],
                        lmargin1=10, lmargin2=10, spacing1=4)
        chat.tag_config("srclink", foreground=COLORS["accent"], underline=True)
        chat.tag_bind("srclink", "<Enter>", lambda e: chat.config(cursor="hand2"))
        chat.tag_bind("srclink", "<Leave>", lambda e: chat.config(cursor=""))
        chat.tag_config("system", foreground=COLORS["muted"], font=self.fonts["small"],
                        justify="center", spacing1=6, spacing3=6)
        chat.tag_config("error", foreground=COLORS["error"], font=self.fonts["small"],
                        justify="center", spacing1=6, spacing3=6)
        self._apply_thought_visibility()

        # 無効状態の Text はクリックでフォーカスを取らないため、コピーできるように取らせる
        chat.bind("<Button-1>", lambda e: chat.focus_set(), add="+")
        self.chat_menu = tk.Menu(self.root, tearoff=False)
        self.chat_menu.add_command(label="コピー", accelerator="Ctrl+C", command=self.copy_selection)
        self.chat_menu.add_command(label="すべて選択", accelerator="Ctrl+A", command=self.select_all_chat)
        self.chat_menu.add_separator()
        self.chat_menu.add_command(label="最後の回答をコピー", command=self.copy_last_answer)
        self.chat_menu.add_command(label="最後の回答をファイルに保存...", command=self.on_save_answer)
        chat.bind(self._context_click(), self._show_chat_menu)
        chat.bind("<Control-a>", lambda e: self.select_all_chat() or "break")

    def _build_composer(self, parent):
        box = ttk.Frame(parent, padding=(0, 8, 0, 0))
        box.pack(side="bottom", fill="x")

        tools = ttk.Frame(box)
        tools.pack(fill="x")
        # 右端のボタンを先に置く (狭いときに左側の項目より先に切れないように)
        self.regen_btn = ttk.Button(
            tools, text="↻ 再生成", width=-8, command=self.on_regenerate, state="disabled"
        )
        self.regen_btn.pack(side="right")
        self.save_btn = ttk.Button(
            tools, text="回答を保存", width=-8, command=self.on_save_answer, state="disabled"
        )
        self.save_btn.pack(side="right", padx=(0, 4))
        self.attach_btn = ttk.Button(tools, text="ファイル添付", width=-4, command=self.on_attach)
        self.attach_btn.pack(side="left")
        self.camera_btn = ttk.Button(tools, text="カメラ", width=-4, command=self.on_camera)
        self.camera_btn.pack(side="left", padx=(4, 0))
        self.paste_btn = ttk.Button(tools, text="貼り付け", width=-4, command=self.on_paste_attach)
        self.paste_btn.pack(side="left", padx=(4, 0))
        ttk.Separator(tools, orient="vertical").pack(side="left", fill="y", padx=6, pady=2)
        # 狭いときは右側から切れるので、よく変えるものほど左に置く
        # (「思考を表示」は一度決めたら変えないことが多いので、表示メニューにだけ置く)
        ttk.Label(tools, text="出力").pack(side="left")
        self.output_combo = ttk.Combobox(
            tools, textvariable=self.output_mode_var, values=list(OUTPUT_MODES),
            state="readonly", width=9,
        )
        self.output_combo.pack(side="left", padx=(4, 0))
        self.output_combo.bind("<<ComboboxSelected>>", lambda e: self._on_output_mode())
        ttk.Separator(tools, orient="vertical").pack(side="left", fill="y", padx=6, pady=2)
        ttk.Checkbutton(tools, text="思考モード", variable=self.thinking_var).pack(side="left")

        # 添付一覧 (添付があるときだけ表示する)
        self.attach_row = ttk.Frame(box)
        ttk.Label(self.attach_row, textvariable=self.attach_var, style="Muted.TLabel").pack(
            side="left"
        )
        self.attach_clear_btn = ttk.Button(
            self.attach_row, text="添付を解除", command=self.on_clear_attachments
        )
        self.attach_clear_btn.pack(side="right")

        self.entry_row = ttk.Frame(box)
        self.entry_row.pack(fill="x", pady=(6, 0))
        # 生成中は「停止」ボタンに切り替わる。入力欄より先に置いて、狭くても隠れないようにする
        self.send_btn = ttk.Button(
            self.entry_row, text="送信", style="Accent.TButton", width=8, command=self.on_send
        )
        self.send_btn.pack(side="right", fill="y", padx=(6, 0))
        self.input = tk.Text(
            self.entry_row, height=3, width=20, wrap="word", font=self.fonts["chat"], undo=True,
            relief="flat", borderwidth=0, padx=8, pady=6,
            highlightthickness=1, highlightbackground=COLORS["border"],
            highlightcolor=COLORS["accent"],
        )
        self.input.pack(side="left", fill="both", expand=True)
        self.input.tag_config("placeholder", foreground=COLORS["muted"])
        # Enter で送信 / Shift+Enter で改行
        self.input.bind("<Return>", lambda e: self.on_send())
        self.input.bind("<Shift-Return>", self._insert_newline)
        # Ctrl+V: 文字は今までどおり貼り付け、スクリーンショットやコピーしたファイルは添付する
        self.input.bind("<Control-v>", self._on_input_paste)
        self.input.bind("<Control-V>", self._on_input_paste)
        self.input.bind("<FocusIn>", lambda e: self._hide_placeholder())
        self.input.bind("<FocusOut>", lambda e: self._show_placeholder())
        self._show_placeholder()
        self._refresh_attachments()

    def _bind_shortcuts(self):
        """キーボードショートカット。Text の既定バインド (Ctrl+O で改行など) より優先させる。"""
        keys = {
            "<Control-n>": self.on_new_chat,
            "<Control-o>": self.on_attach,
            "<Control-r>": self.on_regenerate,
            "<Control-s>": self.on_save_answer,
            "<Control-Shift-C>": self.copy_last_answer,  # CapsLock 中の Ctrl+C と区別するため Shift を明示
            "<Escape>": self.on_stop,
            "<Control-plus>": lambda: self._zoom(1),
            "<Control-equal>": lambda: self._zoom(1),    # JIS / US 配列で + は Shift が要るため
            "<Control-semicolon>": lambda: self._zoom(1),
            "<Control-KP_Add>": lambda: self._zoom(1),
            "<Control-minus>": lambda: self._zoom(-1),
            "<Control-KP_Subtract>": lambda: self._zoom(-1),
            "<Control-0>": lambda: self._set_font_size(DEFAULT_FONT_SIZE),
        }
        def run(handler):
            handler()
            return "break"          # 後続 (Text クラスの既定バインド) を止める

        for seq, handler in keys.items():
            self.input.bind(seq, lambda e, h=handler: run(h))
            self.root.bind(seq, lambda e, h=handler: run(h))
        for widget in (self.chat, self.input):
            widget.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
            widget.bind("<Control-Button-4>", lambda e: self._zoom(1) or "break")
            widget.bind("<Control-Button-5>", lambda e: self._zoom(-1) or "break")

    # ---- UI の小物 -------------------------------------------------------
    @staticmethod
    def _context_click():
        """右クリックのイベント名 (macOS は Button-2)。"""
        return "<Button-2>" if sys.platform == "darwin" else "<Button-3>"

    def _add_entry_hint(self, entry, text_var, value_var, hint):
        """ttk.Entry の中に灰色の案内文を出す (未入力かつフォーカスが無いとき)。

        以前は Entry の上に Label を重ねていたが、フォントや DPI によって Label が
        入力欄の枠からはみ出したため、案内文そのものを入力欄の文字として出す。
        text_var は入力欄に表示する文字、value_var は実際の値 (案内文中は空)。
        """
        state = {"hint": False}

        def focused():
            try:
                return entry.focus_get() is entry
            except Exception:      # コンボボックスの一覧を開いている間などは取得できない
                return False

        def show(*_):
            if state["hint"] or text_var.get() or focused():
                return
            state["hint"] = True
            text_var.set(hint)
            entry.configure(style="Hint.TEntry")

        def hide(*_):
            if not state["hint"]:
                return
            state["hint"] = False
            text_var.set("")
            entry.configure(style="TEntry")

        def on_text(*_):
            if not state["hint"] and value_var.get() != text_var.get():
                value_var.set(text_var.get())

        def on_value(*_):
            # プログラムから値を変えた場合も表示を合わせる
            value = value_var.get()
            if state["hint"] and value:
                hide()
            if not state["hint"] and text_var.get() != value:
                text_var.set(value)
            if not value:
                show()

        entry.bind("<FocusIn>", hide, add="+")
        entry.bind("<FocusOut>", show, add="+")
        text_var.trace_add("write", on_text)
        value_var.trace_add("write", on_value)
        show()

    def _show_placeholder(self):
        """入力欄が空なら案内文を出す。"""
        if self._placeholder_on or self.input.get("1.0", "end-1c"):
            return
        self.input.insert(
            "1.0", "メッセージを入力  (Enter で送信 / Shift+Enter で改行 / Ctrl+V で画像も添付)",
            "placeholder",
        )
        self._placeholder_on = True

    def _hide_placeholder(self):
        if self._placeholder_on:
            self.input.delete("1.0", "end")
            self._placeholder_on = False

    def _input_text(self):
        """入力欄の文字列 (案内文は除く)。"""
        if self._placeholder_on:
            return ""
        return self.input.get("1.0", "end").strip()

    def _zoom(self, step):
        self._set_font_size(int(self.font_size_var.get()) + step)

    def _on_ctrl_wheel(self, event):
        self._zoom(1 if event.delta > 0 else -1)
        return "break"

    def _set_font_size(self, size):
        """チャット欄・入力欄の文字サイズを変える (名前付きフォントなので表示中の文字も追従する)。"""
        size = max(FONT_SIZE_MIN, min(FONT_SIZE_MAX, int(size)))
        self.font_size_var.set(size)
        self.fonts["chat"].configure(size=size)
        self.fonts["chat_bold"].configure(size=size)
        self.fonts["small"].configure(size=max(FONT_SIZE_MIN, size - 2))
        self.fonts["small_italic"].configure(size=max(FONT_SIZE_MIN, size - 2))

    def toggle_settings(self):
        self.show_settings_var.set(not self.show_settings_var.get())
        self._apply_settings_visibility()

    def _apply_settings_visibility(self):
        """詳細設定パネルの開閉。"""
        if self.show_settings_var.get():
            self.settings_panel.pack(fill="x", pady=(8, 0), before=self.chat_frame)
            self.settings_btn.config(text="設定 ▲")
        else:
            self.settings_panel.pack_forget()
            self.settings_btn.config(text="設定 ▼")

    def reset_sampling(self):
        """サンプリングを既定値 (Gemma 4 の推奨値) に戻す。"""
        self.temp_var.set(DEFAULT_TEMPERATURE)
        self.top_p_var.set(DEFAULT_TOP_P)
        self.top_k_var.set(DEFAULT_TOP_K)
        self.maxtok_var.set(DEFAULT_MAX_TOKENS)
        self.status_var.set("サンプリングを既定値に戻しました")

    def _set_model_state(self, text, kind):
        """ヘッダーのモデル状態表示。kind は ok / warn / error / muted。"""
        self.model_state_var.set(f"● {text}")
        self.model_state_label.config(style=f"State{kind.title()}.TLabel")

    def _set_busy(self, busy):
        """読み込み・生成中はステータスバーの進行表示を動かす。"""
        if busy:
            self.progress.pack(side="left", padx=(10, 0))
            self.progress.start(15)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def _show_welcome(self):
        if self.engine.model_name is None:
            self._append_system(
                "上の一覧からモデルを選んで「読み込み」を押すと会話を始められます"
                " (サンプリング等は右上の「設定」、文字の大きさは Ctrl + / Ctrl -)"
            )

    def _show_chat_menu(self, event):
        self.chat.focus_set()
        try:
            self.chat_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.chat_menu.grab_release()

    def copy_selection(self):
        try:
            text = self.chat.get("sel.first", "sel.last")
        except tk.TclError:
            self.status_var.set("コピーする範囲を選択してください")
            return
        self._to_clipboard(text, "選択範囲をコピーしました")

    def select_all_chat(self):
        self.chat.tag_add("sel", "1.0", "end-1c")

    def copy_last_answer(self):
        """直近の AI の回答をクリップボードへ。"""
        last = next((m for m in reversed(self.history) if m["role"] == "assistant"), None)
        if last is None:
            self.status_var.set("コピーできる回答がありません")
            return
        self._to_clipboard(plain_content(last["content"]), "最後の回答をコピーしました")

    def _to_clipboard(self, text, message):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set(message)

    # ---- アプリ設定 (新しいチャットの初期値) ----------------------------
    def _settings_fields(self):
        """{保存キー: (Var, 型変換)}。読み書きで同じ定義を使う。"""
        return {
            "system_prompt": (self.system_var, str),
            "thinking": (self.thinking_var, bool),
            "show_thought": (self.show_thought_var, bool),
            "pdf_as_image": (self.pdf_as_image_var, bool),
            "temperature": (self.temp_var, float),
            "top_p": (self.top_p_var, float),
            "top_k": (self.top_k_var, int),
            "max_tokens": (self.maxtok_var, int),
            "model": (self.model_var, str),
            "show_settings": (self.show_settings_var, bool),
            "font_size": (self.font_size_var, int),
            "save_dir": (self.save_dir_var, str),
            "skill": (self.skill_var, str),
            "rag_enabled": (self.rag_enabled_var, bool),
            "rag_docs_dir": (self.rag_docs_dir_var, str),
            "rag_model_dir": (self.rag_model_dir_var, str),
        }

    def _load_settings(self):
        """前回終了時の設定を復元する。無ければ既定値のまま。

        モデルフォルダだけは UI を組み立てる前に要るので、__init__ で先に反映済み。
        """
        data = load_settings_file()
        for key, (var, cast) in self._settings_fields().items():
            if key not in data:
                continue
            try:
                value = cast(data[key])
            except (TypeError, ValueError):
                continue
            # モデルは今あるものだけ復元する (前回の環境と違うことがあるため)
            if key == "model" and value not in (self.model_combo["values"] or ()):
                continue
            var.set(value)
        self._apply_thought_visibility()
        self._apply_settings_visibility()
        self._set_font_size(self.font_size_var.get())
        # 前回のスキル (消されていれば使わない)。出力形式などは保存済みの値のまま
        if self._current_skill() is None:
            self.skill_var.set(NO_SKILL)
        self._apply_skill(announce=False, template=False, modes=False)
        if data:
            log.info("[設定] 復元: %s", SETTINGS_PATH)

    def _save_settings(self):
        """現在の設定を次回起動用に保存する。"""
        data = {}
        for key, (var, cast) in self._settings_fields().items():
            try:
                data[key] = cast(var.get())
            except (TypeError, ValueError, tk.TclError):
                # Spinbox に数値でない文字が入っている等。その項目は保存しない
                log.warning("[設定] %s の値が不正なため保存しません", key)
        data["models_dir"] = str(MODELS_DIR)
        try:
            SETTINGS_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log.info("[設定] 保存: %s", SETTINGS_PATH)
        except Exception as e:
            log.warning("[設定] 保存に失敗しました (%s)", e)

    # ---- 会話履歴 (サイドバー) ------------------------------------------
    def _make_title(self):
        """最初のユーザー発話から会話タイトルを作る。"""
        for m in self.history:
            if m["role"] == "user":
                t = plain_content(m["content"], with_images=True).strip().replace("\n", " ")
                return (t[:TITLE_MAXLEN] + "…") if len(t) > TITLE_MAXLEN else t
        return "新しいチャット"

    def _save_current(self):
        """現在の会話を保存する (空、または前回保存から変わっていなければ何もしない)。

        開いただけの会話を保存し直すと更新日時が変わり、一覧の並びが入れ替わってしまう。
        """
        if not self.history or not self._dirty:
            return
        if self.current_id is None:
            self.current_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
            self.current_created = time.time()
        self.store.save({
            "id": self.current_id,
            "title": self._make_title(),
            "created": self.current_created,
            "updated": time.time(),
            "model": self.engine.model_name,
            "system_prompt": self.system_var.get(),
            "thinking": bool(self.thinking_var.get()),
            "skill": self.skill_var.get(),
            # 画像は base64 のまま保存すると JSON が肥大するため、本文だけを残す。
            # (会話を再開すると画像はモデルに渡らず、[添付画像: 名前] の記述だけが残る)
            "messages": [
                dict(m, content=plain_content(m["content"], with_images=True))
                for m in self.history
            ],
        })
        self._dirty = False

    def _refresh_sidebar(self):
        """保存済み会話の一覧を再描画する (検索欄の文字で絞り込む)。"""
        tree = self.session_tree
        tree.delete(*tree.get_children())
        query = self.search_var.get().strip().lower()
        for meta in self.store.list_meta():
            title = meta["title"] or "(無題)"
            if query and query not in title.lower():
                continue
            tree.insert(
                "", "end", iid=meta["id"], text=title,
                values=(format_updated(meta["updated"]),),
                tags=("current",) if meta["id"] == self.current_id else (),
            )
        # 開いている会話を選択状態にしておく (どれを見ているか分かるように)
        if self.current_id and tree.exists(self.current_id):
            tree.selection_set(self.current_id)
            tree.see(self.current_id)
        self._refresh_delete_button()

    def _refresh_delete_button(self):
        selected = self.session_tree.selection()
        self.delete_btn.config(state="normal" if selected else "disabled")
        self.delete_btn.config(
            text=f"選択した {len(selected)} 件を削除" if len(selected) > 1 else "選択した会話を削除"
        )

    def _on_session_select(self, event=None):
        """一覧で 1 件だけ選ばれたら、その会話を開く (複数選択は削除用)。"""
        self._refresh_delete_button()
        selected = self.session_tree.selection()
        if len(selected) == 1 and selected[0] != self.current_id:
            self.on_load_session(selected[0])

    def _open_selected_session(self):
        selected = self.session_tree.selection()
        if selected:
            self.on_load_session(selected[0])

    def _show_session_menu(self, event):
        row = self.session_tree.identify_row(event.y)
        if row and row not in self.session_tree.selection():
            self.session_tree.selection_set(row)
        try:
            self.session_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.session_menu.grab_release()

    def _start_new_session(self):
        self.history = []
        self.current_id = None
        self.current_created = None
        self._dirty = False
        self.attachments = []
        self._refresh_attachments()
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self._thumbnails = []
        self._refresh_regen_button()

    def _refresh_regen_button(self):
        """再生成・保存できる応答があるときだけボタンを有効にする。"""
        enabled = not self.generating and any(m["role"] == "assistant" for m in self.history)
        state = "normal" if enabled else "disabled"
        self.regen_btn.config(state=state)
        self.save_btn.config(state=state)

    # ---- 回答の保存 ------------------------------------------------------
    def _last_answer(self):
        last = next((m for m in reversed(self.history) if m["role"] == "assistant"), None)
        return plain_content(last["content"]) if last else None

    def on_save_answer(self):
        """直近の回答をファイルに保存する。

        候補は「回答全体 (.md / Word)」「コードブロック単体」「表 (Excel)」。
        ブロック単体なら言語名から拡張子を決める (```csv なら .csv)。
        書き込みはアプリが行い、場所は必ず人が選ぶ (モデルに任意の場所へ書かせない)。
        """
        if self.generating:
            return
        answer = self._last_answer()
        if not answer:
            self.status_var.set("保存できる回答がありません")
            return

        # 候補: [(見出し, プレビュー, 書き出す関数, 中身, 拡張子)]
        candidates = [
            ("回答全体 (.md)", "", save_text_file, answer.rstrip() + "\n", ".md"),
            ("回答全体 → Word (.docx)", "", write_docx, markdown_to_document(answer), ".docx"),
        ]
        tables = extract_tables(answer)
        document = candidates[1][3]
        has_markdown_table = any(b["type"] == "table" for b in document["blocks"])
        for i, table in enumerate(tables, 1):
            preview = " | ".join(table["columns"])[:60]
            label = f"表 {i} → Excel (.xlsx, {len(table['rows'])} 行 × {len(table['columns'])} 列)"
            candidates.append((label, preview, write_xlsx, [table], ".xlsx"))
        if len(tables) > 1:
            candidates.append((f"すべての表 → Excel (.xlsx, {len(tables)} シート)", "",
                               write_xlsx, tables, ".xlsx"))
        first_block = len(candidates)
        for i, (lang, body) in enumerate(extract_code_blocks(answer), 1):
            lines = body.splitlines()
            preview = "\n".join(line[:60] for line in lines[:SAVE_PREVIEW_LINES])
            if len(lines) > SAVE_PREVIEW_LINES:
                preview += "\n…"
            suffix = suffix_for_language(lang)
            label = f"コードブロック {i}: {lang or 'テキスト'} ({len(lines)} 行, {suffix})"
            candidates.append((label, preview, save_text_file, body, suffix))

        # 既定は「Markdown の表があれば Excel、コードブロックがあればそのブロック、
        # 無ければ回答全体」。```csv で頼まれた場合は CSV のまま保存したいことが多い
        if has_markdown_table:
            default = 2
        elif len(candidates) > first_block:
            default = first_block
        else:
            default = 0
        index = SaveChoiceDialog(
            self.root, [(c[0], c[1]) for c in candidates], default=default
        ).show()
        if index is None:
            return
        _, _, writer, payload, suffix = candidates[index]
        self._save_with_dialog(writer, payload, suffix, "answer")

    def _save_with_dialog(self, writer, payload, suffix, stem):
        """保存先を聞いて writer(path, payload) で書き出す。保存したパスを返す。"""
        initial_dir = self.save_dir_var.get()
        if not initial_dir or not Path(initial_dir).is_dir():
            initial_dir = str(Path.home())
        chosen = filedialog.asksaveasfilename(
            title="回答をファイルに保存",
            initialdir=initial_dir,
            initialfile=f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}{suffix}",
            defaultextension=suffix,
            filetypes=[(f"{suffix} ファイル", f"*{suffix}"), ("すべてのファイル", "*.*")],
        )
        if not chosen:
            return None
        try:
            path = writer(chosen, payload)
        except Exception as e:
            log.exception("[保存] 失敗: %s", chosen)
            self._append_system(f"保存できませんでした: {e}", error=True)
            return None
        self.save_dir_var.set(str(path.parent))
        log.info("[保存] %s", path)
        self._append_system(f"保存しました: {path}")
        self.status_var.set(f"保存しました: {path.name}")
        return path

    def _finish_structured(self, stopped):
        """Excel / Word 出力の JSON を読み、画面の表示を表・文書に置き換える。

        読めた場合は (種別, データ) を返し、_assistant_buf を Markdown に差し替える
        (会話履歴には JSON ではなく読める形で残す。後から「回答を保存」でも使える)。
        読めなければ JSON のまま残して理由を出し、None を返す。
        """
        kind = self._structured
        self._structured = None
        if stopped:
            self._append_system("途中で止めたため、ファイルは作りませんでした")
            return None
        try:
            data = parse_structured(kind, self._assistant_buf)
        except ValueError as e:
            log.warning("[GEN] 構造化出力を読めません: %s", e)
            hint = ""
            if self.engine.last_finish_reason == "length":
                hint = " (max_tokens に達して途中で切れています。詳細設定で増やしてください)"
            self._append_system(f"表・文書として読めませんでした: {e}{hint}", error=True)
            return None
        markdown = structured_to_markdown(kind, data)
        self.chat.config(state="normal")
        self.chat.delete("answer_start", "end-1c")
        self.chat.insert("end", markdown, "assistant")
        self.chat.config(state="disabled")
        self.chat.see("end")
        self._assistant_buf = markdown
        return kind, data

    def _save_structured(self, kind, data):
        """Excel / Word 出力の結果を保存する (保存先は人が選ぶ)。"""
        stem = safe_filename(data.get("title"))
        if kind == "table":
            self._save_with_dialog(write_xlsx, [data], ".xlsx", stem)
        else:
            self._save_with_dialog(write_docx, data, ".docx", stem)

    def on_new_chat(self):
        if self.generating:
            return
        self._save_current()          # 開いていた会話を保存
        self._start_new_session()
        self._refresh_sidebar()
        self._show_welcome()
        self.status_var.set("新しいチャット")
        self.input.focus_set()
        log.info("[UI] 新規チャット")

    def on_load_session(self, session_id):
        """サイドバーの会話をクリック -> 再開。"""
        if self.generating:
            self.status_var.set("生成中は会話を切り替えられません")
            self._refresh_sidebar()       # 選択表示を今の会話に戻す
            return
        self._save_current()          # 今の会話を保存してから切り替え
        data = self.store.load(session_id)
        self.history = data.get("messages", [])
        self.current_id = data["id"]
        self.current_created = data.get("created", time.time())
        self._dirty = False
        self.system_var.set(data.get("system_prompt", ""))
        # 会話で使っていたスキルに戻す (出力形式もスキルに合わせる)。思考モードは会話の値を優先
        skill_name = data.get("skill") or NO_SKILL
        self.skill_var.set(skill_name if self._skill_by_name(skill_name) else NO_SKILL)
        self._apply_skill(announce=False, template=False)
        self.thinking_var.set(bool(data.get("thinking", False)))
        self._repaint_chat()
        self._refresh_sidebar()
        self._refresh_regen_button()
        self.status_var.set(f"会話を再開: {data.get('title', '')}")
        log.info("[UI] 会話を再開: %s (%d発話)", session_id, len(self.history))

    def on_delete_selected(self):
        """選択された会話をまとめて削除。"""
        if self.generating:
            return
        ids = list(self.session_tree.selection())
        if not ids:
            self.status_var.set("削除する会話を一覧から選んでください")
            return
        titles = [self.session_tree.item(sid, "text") for sid in ids[:5]]
        more = f"\nほか {len(ids) - 5} 件" if len(ids) > 5 else ""
        if not messagebox.askyesno(
            "会話の削除",
            f"{len(ids)} 件の会話を削除します。元に戻せません。\n\n"
            + "\n".join(f"・{t}" for t in titles) + more,
            icon="warning",
        ):
            return
        for sid in ids:
            self.store.delete(sid)
            if sid == self.current_id:
                self._start_new_session()
        self._refresh_sidebar()
        self.status_var.set(f"{len(ids)} 件の会話を削除しました")
        log.info("[UI] %d 件の履歴を削除", len(ids))

    # ---- ハンドラ --------------------------------------------------------
    def on_load(self):
        name = self.model_var.get()
        if not name:
            return
        self.load_btn.config(state="disabled")
        self.status_var.set(f"読み込み中: {name} ...")
        self._set_model_state("読み込み中", "muted")
        self._set_busy(True)
        self._append_system(f"モデル読み込み中: {name}")
        log.info("[UI] 読み込みボタン押下 -> %s", name)
        threading.Thread(
            target=self._load_worker, args=(name,), name="loader", daemon=True
        ).start()

    def on_choose_models_dir(self):
        """モデル (.gguf) を置いているフォルダを選び直す。設定ファイルにも保存する。"""
        if self.generating:
            return
        chosen = filedialog.askdirectory(
            title="モデルフォルダを選択", initialdir=str(MODELS_DIR)
        )
        if not chosen:
            return
        set_models_dir(chosen)
        models = discover_models()
        self.model_combo["values"] = models
        if models:
            self.model_combo.current(0)
        else:
            self.model_var.set("")
        self._append_system(f"モデルフォルダ: {MODELS_DIR.resolve()} ({len(models)} 件)")
        self.status_var.set(f"モデル {len(models)} 件")
        self._save_settings()
        if MODELS_DIR_FROM_ENV:
            self._append_system(
                "環境変数 LLM_MODELS_DIR が設定されているため、"
                "次回起動時はそちらが優先されます"
            )

    def _load_worker(self, name):
        t0 = time.time()
        try:
            ok = self.engine.load(name)
        except Exception as e:
            # ここで握らないとローダースレッドごと落ち、
            # UI が「読み込み中 ...」のまま読み込みボタンも無効のまま固まる。
            log.exception("[LOAD] 読み込み中にエラー")
            self.token_queue.put(("load_error", f"{name}: {e}"))
            return
        tag = "ロード完了" if ok else "モックモード (実推論なし)"
        vision = "画像入力 可" if self.engine.vision else "画像入力 不可 (mmproj 未検出)"
        self.token_queue.put(("loaded", f"{tag}: {name} ({time.time() - t0:.1f}s) / {vision}"))

    # ---- 添付ファイル ----------------------------------------------------
    def on_attach(self):
        """画像 / 文書ファイルを選び、次の送信に添付する。"""
        if self.generating:
            return
        paths = filedialog.askopenfilenames(
            title="添付するファイルを選択",
            filetypes=[
                ("対応ファイル", " ".join(f"*{s}" for s in sorted(SUPPORTED_SUFFIXES))),
                ("画像", " ".join(f"*{s}" for s in sorted(IMAGE_SUFFIXES))),
                ("すべてのファイル", "*.*"),
            ],
        )
        self._add_paths(Path(raw) for raw in paths)

    def _add_paths(self, paths):
        """ファイルを添付に加える。読めないものは理由を表示して飛ばす。追加した件数を返す。"""
        added = 0
        for path in paths:
            try:
                if path.is_dir():
                    raise RuntimeError("フォルダは添付できません")
                entries = self._make_attachments(path)
            except Exception as e:
                log.warning("[添付] 追加できません: %s (%s)", path.name, e)
                self._append_system(f"添付できません: {path.name} ({e})")
                continue
            self.attachments.extend(entries)
            added += len(entries)
        self._refresh_attachments()
        return added

    # ---- クリップボードからの添付 ----------------------------------------
    def _clipboard_text(self):
        try:
            return self.root.clipboard_get()
        except tk.TclError:             # 文字が入っていない
            return ""

    def _attach_clipboard_image(self, data):
        name = time.strftime("clipboard_%Y%m%d_%H%M%S.png")
        if not self.engine.vision:
            self._append_system(
                "このモデル構成では画像を渡せません (mmproj 未検出)。添付はできますが送信時に外されます"
            )
        self.attachments.append(image_attachment(name, data))
        self._refresh_attachments()
        log.info("[添付] クリップボードの画像: %s (%.1f KB)", name, len(data) / 1024)
        self.status_var.set(f"クリップボードの画像を添付しました: {name}")

    def on_paste_attach(self):
        """「貼り付け」ボタン: クリップボードの中身を添付する。

        優先順: コピーしたファイル > 文字 (テキストとして添付) > 画像。
        Excel のセルをコピーすると文字と画像の両方が入るため、文字を画像より優先する。
        """
        if self.generating:
            return
        try:
            kind, payload = grab_clipboard()
        except RuntimeError as e:
            kind, payload, error = None, None, str(e)
        else:
            error = None
        if kind == "files":
            added = self._add_paths(payload)
            if added:
                self.status_var.set(f"クリップボードのファイルを添付しました ({added} 件)")
            return
        text = self._clipboard_text().strip()
        if text:
            if len(text) > MAX_DOC_CHARS:
                omitted = len(text) - MAX_DOC_CHARS
                text = text[:MAX_DOC_CHARS] + f"\n…(以降 {omitted} 文字を省略)"
            name = time.strftime("clipboard_%H%M%S.txt")
            self.attachments.append({"kind": "document", "name": name, "text": text})
            self._refresh_attachments()
            log.info("[添付] クリップボードの文字: %d 文字", len(text))
            self.status_var.set(f"クリップボードの文字を添付しました ({len(text)} 文字)")
            return
        if kind == "image":
            self._attach_clipboard_image(payload)
            return
        self._append_system(error or "クリップボードに添付できるものがありません", error=bool(error))

    def _on_input_paste(self, event=None):
        """入力欄の Ctrl+V: 文字があれば普通に貼り付け、無ければ画像・ファイルを添付する。"""
        if self.generating or self._clipboard_text():
            return None                 # Tk の既定の貼り付けに任せる
        try:
            kind, payload = grab_clipboard()
        except RuntimeError as e:
            log.debug("[添付] クリップボードを読めません: %s", e)
            return None
        if kind == "files":
            self._add_paths(payload)
            return "break"
        if kind == "image":
            self._attach_clipboard_image(payload)
            return "break"
        return None

    def _make_attachments(self, path):
        """パスから添付エントリの一覧を作る。読めない場合は例外を送出。

        PDF はページごとの画像になるため、1 ファイルから複数エントリができる。
        """
        if not path.exists():
            raise RuntimeError("ファイルが見つかりません")

        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            data = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            log.info("[添付] 画像: %s (%.1f KB)", path.name, len(data) / 1024)
            return [image_attachment(path.name, data, mime)]

        if suffix == ".pdf" and self.pdf_as_image_var.get():
            return self._pdf_page_attachments(path)

        try:
            text = extract_document_text(path)
        except RuntimeError as e:
            if suffix != ".pdf":
                raise
            # 文字を持たない PDF (スキャン画像など) は画像として渡す
            log.info("[添付] %s はテキストを抽出できないため画像化します (%s)", path.name, e)
            return self._pdf_page_attachments(path, reason=str(e))

        log.info("[添付] 文書: %s (%d 文字)", path.name, len(text))
        return [{"kind": "document", "name": path.name, "text": text}]

    def _pdf_page_attachments(self, path, reason=None):
        """PDF をページ画像の添付エントリに変換する。

        reason: テキストとして読めずに画像へ回ってきた場合の理由 (エラー文言に含める)。
        """
        if not self.engine.vision:
            if reason:
                raise RuntimeError(f"テキストとして読めず ({reason})、画像化には mmproj が必要です")
            raise RuntimeError("画像として渡すには mmproj が必要です")
        images = render_pdf_pages(path)
        if not images:
            raise RuntimeError("ページがありません")
        log.info("[添付] PDF を画像化: %s (%d ページ)", path.name, len(images))
        return [image_attachment(f"{path.name} p.{page}", data) for page, data in images]

    def on_camera(self):
        """カメラ小窓を開く。撮影した画像はそのまま添付に入る。"""
        if self.generating:
            return
        if getattr(self, "camera_window", None) is not None:
            # すでに開いていれば前面に出すだけ
            try:
                self.camera_window.window.lift()
                return
            except Exception:
                self.camera_window = None
        if not self.engine.vision:
            self._append_system(
                "このモデル構成では画像を渡せません (mmproj 未検出)。撮影はできますが送信時に外されます"
            )
        try:
            self.camera_window = CameraWindow(self.root, self._on_camera_capture)
        except Exception as e:
            log.warning("[カメラ] 起動できません (%s)", e)
            self._append_system(f"カメラを開けません: {e}")
            self.camera_window = None
            return
        self.camera_window.window.bind("<Destroy>", self._on_camera_closed)

    def _on_camera_capture(self, attachment):
        self.attachments.append(attachment)
        self._refresh_attachments()

    def _on_camera_closed(self, event):
        # Toplevel の子ウィジェットの Destroy でも呼ばれるため、本体か確認する
        if self.camera_window is not None and event.widget is self.camera_window.window:
            self.camera_window = None

    def on_clear_attachments(self):
        if self.generating or not self.attachments:
            return
        log.info("[添付] %d 件を解除", len(self.attachments))
        self.attachments = []
        self._refresh_attachments()

    def _refresh_attachments(self):
        """添付一覧の行を更新する (添付が無ければ隠す)。"""
        if not self.attachments:
            self.attach_var.set("")
            self.attach_row.pack_forget()
            return
        names = ", ".join(a["name"] for a in self.attachments)
        if len(names) > ATTACH_LABEL_MAXLEN:
            names = names[:ATTACH_LABEL_MAXLEN] + "…"
        self.attach_var.set(f"添付 {len(self.attachments)} 件: {names}")
        self.attach_row.pack(fill="x", pady=(6, 0), before=self.entry_row)

    def _build_content(self, text):
        """入力文と添付から送信用の content を組み立てる。

        画像が無ければ従来どおり str を返す (モックモードや履歴表示をそのまま使うため)。
        画像がある場合のみ OpenAI 形式のパート配列にする。
        """
        images = [a for a in self.attachments if a["kind"] == "image"]
        documents = [a for a in self.attachments if a["kind"] == "document"]

        if images and not self.engine.vision:
            self._append_system(
                "このモデル構成では画像を渡せません (mmproj 未検出)。テキストのみ送信します"
            )
            log.warning("[添付] mmproj 未検出のため画像 %d 件を除外", len(images))
            images = []

        blocks = [text] if text else []
        for doc in documents:
            blocks.append(f"--- 添付ファイル: {doc['name']} ---\n{doc['text']}\n--- ここまで ---")
        merged = "\n\n".join(blocks)

        if not images:
            return merged
        # 画像は「本文で言及 + 画像そのもの」にすると 2 枚あると解釈されるため、
        # 本文には書かず画像パートだけを渡す。名前は保存・表示用にパートへ持たせる。
        parts = [{"type": "text", "text": merged}] if merged else []
        parts.extend(
            {"type": "image_url", "image_url": {"url": img["data_uri"]}, "name": img["name"]}
            for img in images
        )
        return parts

    # ---- 送信 ------------------------------------------------------------
    def on_send(self):
        if self.generating:
            log.debug("[UI] 生成中のため送信を無視")
            return "break"
        text = self._input_text()
        if not text and not self.attachments:
            return "break"
        if self.engine.model_name is None:
            self._append_system("先にモデルを読み込んでください (上の「読み込み」ボタン)")
            self.load_btn.focus_set()
            log.warning("[UI] モデル未読み込みで送信されました")
            return "break"

        content = self._build_content(text)
        if isinstance(content, str) and not content.strip():
            # 添付が全て除外された (画像を渡せない構成で画像だけ添付した等)
            self._append_system("送信できる内容がありません")
            return "break"

        self.input.delete("1.0", "end")
        self.input.edit_reset()         # 送信済みの文を Ctrl+Z で戻さない
        self.attachments = []
        self._refresh_attachments()
        self.history.append({"role": "user", "content": content})
        self._dirty = True
        self._append_message("user", plain_content(content), content_images(content))
        log.info(
            "[UI] 送信: %d 文字 / 画像 %d 枚 / 履歴 %d 件",
            len(plain_content(content)),
            0 if isinstance(content, str) else sum(1 for p in content if p["type"] == "image_url"),
            len(self.history),
        )

        self._start_generation()
        return "break"

    def on_regenerate(self):
        """直前の応答を捨てて、同じ入力で生成し直す。"""
        if self.generating or self.engine.model_name is None:
            return
        last = next(
            (i for i in range(len(self.history) - 1, -1, -1)
             if self.history[i]["role"] == "assistant"),
            None,
        )
        if last is None:
            self._append_system("再生成できる応答がありません")
            return
        self.history = self.history[:last]
        self._dirty = True
        self._repaint_chat()
        log.info("[UI] 再生成 (履歴 %d 件から)", len(self.history))
        self._start_generation()

    def on_stop(self):
        """生成中のワーカーに停止を伝える。そこまでの応答は残す。"""
        if not self.generating:
            return
        self._stop_event.set()
        self.send_btn.config(state="disabled")
        self.status_var.set("停止中 ...")
        log.info("[UI] 停止要求")

    # ---- スキル ----------------------------------------------------------
    def _skill_by_name(self, name):
        return next((s for s in self.skills if s["name"] == name), None)

    def _current_skill(self):
        return self._skill_by_name(self.skill_var.get())

    def _refresh_skill_choices(self):
        self.skill_combo["values"] = [NO_SKILL] + [s["name"] for s in self.skills]

    def _apply_skill(self, announce=True, template=True, modes=True):
        """選んだスキルを反映する。

        modes: 出力形式と思考モードもスキルに合わせるか (起動時の復元では保存値を優先)
        template: 入力欄が空ならひな形を入れるか
        """
        skill = self._current_skill()
        self.skill_desc_var.set(skill["description"] if skill else "")
        if skill is None:
            if modes and announce:
                self.output_mode_var.set("チャット")
                self._append_system("スキルを使いません")
            return
        if modes:
            self.output_mode_var.set(skill["output"])
            if skill["thinking"] != "そのまま":
                self.thinking_var.set(skill["thinking"] == "オン")
        if template and skill["template"] and not self._input_text():
            self._hide_placeholder()
            self.input.insert("1.0", skill["template"])
            self.input.mark_set("insert", "end-1c")
        if announce:
            details = [f"出力: {self.output_mode_var.get()}"]
            if skill["thinking"] != "そのまま":
                details.append(f"思考モード: {skill['thinking']}")
            self._append_system(f"スキル「{skill['name']}」を使います ({'、'.join(details)})")
            self.input.focus_set()
        log.info("[スキル] 選択: %s", skill["name"])

    def on_edit_skills(self):
        """スキルの設定画面を開く。保存したらファイルへ書いて、一覧を更新する。"""
        if self.generating:
            return
        result = SkillEditor(self.root, self.skills, selected=self.skill_var.get()).show()
        if result is None:
            return
        try:
            save_skills(result)
        except Exception as e:
            log.exception("[スキル] 保存に失敗")
            self._append_system(f"スキルを保存できませんでした: {e}", error=True)
            return
        self.skills = result
        self._refresh_skill_choices()
        if self._current_skill() is None:
            self.skill_var.set(NO_SKILL)
        # 選んでいるスキルの内容が変わった可能性があるので、説明と出力形式を合わせ直す
        self._apply_skill(announce=False, template=False)
        self.status_var.set(f"スキルを保存しました ({len(result)} 件)")

    # ---- 資料検索 (RAG) --------------------------------------------------
    def _load_rag_index(self):
        """保存済みの索引を読む (rag.py と numpy が無ければ資料検索を使えないだけ)。"""
        try:
            import rag

            index = rag.RagIndex()
            index.load()
        except Exception as e:
            log.warning("[RAG] 資料検索を使えません (%s)", e)
            index = None
        self.rag_index = index
        self._refresh_rag_status()

    def _refresh_rag_status(self):
        index = self.rag_index
        if index is None:
            self.rag_status_var.set("使えません (numpy が必要です)")
        elif not index.ready:
            self.rag_status_var.set("索引がありません (「設定」から作成)")
        else:
            self.rag_status_var.set(
                f"{len(index.meta.get('files', {}))} ファイル / {len(index.chunks)} チャンク"
            )

    def _on_rag_toggle(self):
        if not self.rag_enabled_var.get():
            self._append_system("資料検索を使いません")
            return
        if self.rag_index is None or not self.rag_index.ready:
            self._append_system("資料検索の索引がありません。「設定」から作成してください")
        elif not self.rag_model_dir_var.get().strip():
            self._append_system("埋め込みモデルが未設定です。「設定」から選んでください")
        else:
            self._append_system("資料検索を使います (回答の後に出典を表示します)")

    def on_rag_settings(self):
        if self.generating:
            return
        RagWindow(self)

    def _rag_embedder(self, model_dir):
        """埋め込みモデル (一度読んだら使い回す。フォルダが変わったら読み直す)。"""
        import rag

        with self._embed_lock:
            if self._embedder is None or self._embedder_key != model_dir:
                self._embedder = rag.Embedder(model_dir)
                self._embedder_key = model_dir
            return self._embedder

    def _rag_query(self):
        """資料検索を使う場合の (検索文, 埋め込みモデルのフォルダ)。使わないなら None。

        Tk の変数はワーカースレッドから読めないので、ここ (UI スレッド) で値を取り出しておく。
        """
        if not self.rag_enabled_var.get():
            return None
        if self.rag_index is None or not self.rag_index.ready:
            self._append_system("資料検索の索引が無いため、資料を使わずに答えます")
            return None
        if not self.rag_model_dir_var.get().strip():
            self._append_system("埋め込みモデルが未設定のため、資料を使わずに答えます")
            return None
        last = self.history[-1] if self.history else None
        if not last or last["role"] != "user":
            return None
        query = plain_content(last["content"]).strip()
        return (query, self.rag_model_dir_var.get().strip()) if query else None

    def _rag_augment(self, messages, query, model_dir):
        """(ワーカースレッド) 資料を検索し、直近の発話に抜粋と出典を差し込む。

        差し込むのは今回モデルへ送る messages だけで、画面と会話履歴の発話はそのまま
        (毎ターン抜粋が履歴に積み上がって、コンテキストを圧迫しないように)。
        """
        import rag

        self.token_queue.put(("status", "資料を検索中 ..."))
        t0 = time.time()
        # 索引を作成中なら待たずに、資料なしで答える
        if not self._rag_lock.acquire(blocking=False):
            self.token_queue.put(("rag_error", "索引を作成中のため、今回は資料を使いません"))
            return messages
        try:
            embedder = self._rag_embedder(model_dir)
            hits = self.rag_index.search(query, embedder)
        except Exception as e:
            log.exception("[RAG] 検索に失敗")
            self.token_queue.put(("rag_error", str(e)))
            return messages
        finally:
            self._rag_lock.release()
        log.info("[RAG] 検索 %.2f 秒: %s", time.time() - t0,
                 [(rag_source_label(h), round(h["score"], 3)) for h in hits])
        if not hits:
            return messages
        self.token_queue.put(("sources", hits))
        self.token_queue.put(("status", "生成中 ...  (Esc で停止)"))
        text = rag.format_context(hits, query)
        last = dict(messages[-1])
        content = last["content"]
        if isinstance(content, list):
            # 画像付きの発話: 文のパートを置き換える (無ければ先頭に足す)
            parts = [p for p in content if p.get("type") != "text"]
            last["content"] = [{"type": "text", "text": text}] + parts
        else:
            last["content"] = text
        return messages[:-1] + [last]

    def _append_sources(self, sources):
        """回答の下に出典を出す。クリックするとそのファイルを開く。"""
        self.chat.config(state="normal")
        self._ensure_newline()
        self.chat.insert("end", "出典: ", "source")
        for i, source in enumerate(sources, 1):
            self._link_count += 1
            tag = f"srclink{self._link_count}"
            self.chat.insert("end", f"[{i}] {source['label']}", ("source", "srclink", tag))
            self.chat.tag_bind(tag, "<Button-1>",
                               lambda e, p=source["path"]: self._open_source(p))
            if i < len(sources):
                self.chat.insert("end", "  ", "source")
        self.chat.insert("end", "\n", "source")
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _open_source(self, path):
        if not Path(path).exists():
            self._append_system(f"ファイルが見つかりません: {path}", error=True)
            return
        try:
            open_path(path)
        except Exception as e:
            self._append_system(f"ファイルを開けません: {e}", error=True)

    def _output_kind(self):
        """今の出力形式 ("table" / "document"、通常のチャットなら None)。"""
        return OUTPUT_MODES.get(self.output_mode_var.get())

    def _on_output_mode(self):
        kind = self._output_kind()
        if kind:
            self._append_system(
                f"出力形式: {self.output_mode_var.get()}。次の回答は"
                + ("表" if kind == "table" else "文書")
                + "として作り、終わったら保存先を聞きます (思考モードは使いません)"
            )
        self.input.focus_set()

    def _system_message(self):
        """システムプロンプト (思考モードなら先頭に <|think|>) を組み立てる。

        Excel / Word 出力のときは形式の説明を足し、思考モードは使わない
        (出力を JSON の文法で縛るため、思考チャネルの記号を出せなくなる)。
        """
        skill = self._current_skill()
        parts = [
            self.system_var.get().strip(),
            skill["instructions"] if skill else "",
            STRUCTURED_INSTRUCTIONS.get(self._output_kind(), ""),
        ]
        text = "\n\n".join(p for p in parts if p)
        kind = self._output_kind()
        if not kind and self.thinking_var.get():
            text = f"{THINK_TOKEN}\n{text}" if text else THINK_TOKEN
        return {"role": "system", "content": text} if text else None

    def _start_generation(self):
        """現在の履歴で生成ワーカーを起動する (送信・再生成の共通処理)。"""
        usage = self.engine.context_usage()
        if usage and usage[0] > usage[1] * 0.85:
            self._append_system(
                f"コンテキストの残りが少なくなっています ({usage[0]}/{usage[1]})。"
                "「＋ 新規チャット」で始め直すか、LLM_N_CTX を大きくしてください"
            )
        # 出典などアプリ用のキーはモデルへ渡さない (チャットテンプレートが想定していない)
        messages = [{"role": m["role"], "content": m["content"]} for m in self.history]
        system = self._system_message()
        if system:
            messages.insert(0, system)
        rag_query = self._rag_query()
        max_tokens = int(self.maxtok_var.get())
        structured = self._output_kind()
        if self.thinking_var.get() and not structured:
            # 思考は回答と同じ予算を消費するので、その分を上乗せする
            max_tokens += THINKING_EXTRA_TOKENS

        # 履歴が伸びてもシステムプロンプトが押し出されないよう、予算内に収める
        budget = self.engine.context_budget(max_tokens)
        if rag_query:
            budget -= RAG_RESERVED_TOKENS       # 検索した抜粋を差し込むぶんを空けておく
        messages, dropped, stripped = fit_to_budget(
            messages, budget, self.engine.count_tokens
        )
        if dropped or stripped:
            log.info(
                "[CTX] 予算 %d トークンに調整: 古い発話 %d 件を除外 / 画像 %d 件を除外",
                budget, dropped, stripped,
            )
            notes = []
            if dropped:
                notes.append(f"古い発話 {dropped} 件")
            if stripped:
                notes.append(f"過去の画像 {stripped} 件")
            self._append_system(
                f"コンテキストに収めるため、{'と'.join(notes)}を今回の送信から外しました"
                " (画面と保存済みの履歴はそのまま残ります)"
            )

        params = {
            "max_tokens": max_tokens,
            "temperature": float(self.temp_var.get()),
            "top_p": float(self.top_p_var.get()),
            "top_k": int(self.top_k_var.get()),
        }
        if structured:
            params["response_format"] = {
                "type": "json_object", "schema": STRUCTURED_SCHEMAS[structured],
            }

        # 準備が済んでから UI を生成中の状態にする (通知が "AI:" の後に出ないように)
        self.generating = True
        self._stop_event.clear()
        self._in_thought = False        # 思考チャネルを表示中か
        self._answer_started = False    # 回答の最初のトークンを出したか
        self._structured = structured   # この生成が Excel / Word 出力か
        self._pending_sources = None
        # 送信ボタンは生成中だけ「停止」になる (Esc でも止められる)
        self.send_btn.config(text="■ 停止", command=self.on_stop, state="normal")
        self.regen_btn.config(state="disabled")
        self.status_var.set(
            {"table": "表を生成中 ...", "document": "文書を生成中 ..."}.get(structured, "生成中 ...")
            + "  (Esc で停止)"
        )
        self._set_busy(True)
        self._append_message("assistant", "")  # "AI" の見出しだけ先に表示
        # 生成された JSON を、完成後に表・文書の表示へ置き換えるための目印
        self.chat.mark_set("answer_start", "end-1c")
        self.chat.mark_gravity("answer_start", "left")

        threading.Thread(
            target=self._gen_worker, args=(messages, params, rag_query), name="gen", daemon=True,
        ).start()

    def _gen_worker(self, messages, params, rag_query=None):
        log.info(
            "[GEN] 生成開始 (max_tokens=%(max_tokens)d, temperature=%(temperature).2f,"
            " top_p=%(top_p).2f, top_k=%(top_k)d)", params,
        )
        if params.get("response_format"):
            log.info("[GEN] 出力を JSON スキーマで制約 (%s)", self._structured)
        t0 = time.time()
        if rag_query:
            messages = self._rag_augment(messages, *rag_query)
        first = None
        n = 0
        splitter = ThoughtSplitter()
        stopped = False
        try:
            for piece in self.engine.stream(messages, **params):
                if self._stop_event.is_set():
                    stopped = True
                    log.info("[GEN] 停止要求により打ち切り (%d tokens)", n)
                    break
                if first is None:
                    first = time.time()
                    log.info("[GEN] 初トークンまで %.2fs", first - t0)
                n += 1
                for kind, text in splitter.feed(piece):
                    self.token_queue.put((kind, text))
                if n % 20 == 0:
                    dt = time.time() - (first or t0)
                    log.debug("[GEN] 経過 %d tokens, %.1f tok/s", n, n / dt if dt > 0 else 0.0)
            for kind, text in splitter.flush():
                self.token_queue.put((kind, text))
            dt = time.time() - (first or t0)
            speed = n / dt if dt > 0 else 0.0
            finish = getattr(self.engine, "last_finish_reason", None)
            log.info(
                "[GEN] %s: %d tokens, 総 %.2fs, %.1f tok/s (finish_reason=%s)",
                "停止" if stopped else "完了", n, time.time() - t0, speed, finish,
            )
            self.token_queue.put((
                "stopped" if stopped else "end",
                {"tokens": n, "speed": speed, "finish": finish,
                 # 入力の読み込みにかかった時間 (送信してから最初の文字が出るまで)
                 "wait": (first - t0) if first else None},
            ))
        except Exception as e:  # 推論中の例外もターミナルに出す
            log.exception("[GEN] 生成中にエラー")
            self.token_queue.put(("error", str(e)))

    def _insert_newline(self, event):
        """Shift+Enter: 送信せず改行を挿入する。"""
        self.input.insert("insert", "\n")
        return "break"

    def on_close(self):
        """ウィンドウを閉じる前に現在の会話と設定を保存。"""
        if self.camera_window is not None:
            # カメラを掴んだままだと他アプリから使えなくなるので必ず離す
            try:
                self.camera_window.close()
            except Exception:
                log.exception("[カメラ] 終了処理に失敗")
            self.camera_window = None
        try:
            self._save_current()
            self._save_settings()
        finally:
            log.info("=== 終了 ===")
            self.root.destroy()

    # ---- メインスレッドでの描画更新 -------------------------------------
    def _poll_queue(self):
        """ワーカースレッドからの出力をメインスレッドで反映する。"""
        try:
            while True:
                kind, payload = self.token_queue.get_nowait()
                if kind == "answer":
                    if not self._answer_started:
                        # 思考の直後に回答が続くので、区切りを入れてから始める。
                        # 区切りは思考タグを付ける -> 折りたたみ時に一緒に隠れる
                        if self._in_thought:
                            self._stream_token("\n", "thought")
                            self._in_thought = False
                        payload = payload.lstrip()   # 回答先頭の余分な改行を落とす
                        if not payload:
                            continue
                        self._answer_started = True
                    self._assistant_buf += payload
                    self._stream_token(payload, "json_raw" if self._structured else "assistant")
                elif kind == "thought":
                    # 思考は表示するだけで履歴には残さない (次のターンへは渡さない)
                    if not self._in_thought:
                        self._in_thought = True
                        self._stream_token("思考\n", ("thought", "thought_head"))
                        payload = payload.lstrip("\n")
                    self._stream_token(payload, "thought")
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "sources":
                    self._pending_sources = payload
                elif kind == "rag_error":
                    self._append_system(f"資料検索を使えませんでした: {payload}", error=True)
                elif kind in ("end", "stopped"):
                    structured_result = None
                    if self._structured and self._assistant_buf:
                        structured_result = self._finish_structured(kind == "stopped")
                    sources = self._pending_sources
                    self._pending_sources = None
                    if self._assistant_buf:
                        message = {"role": "assistant", "content": self._assistant_buf}
                        if sources:
                            # 出典は会話と一緒に保存し、開き直したときも表示する
                            message["sources"] = [
                                {"label": rag_source_label(h), "path": h["path"]} for h in sources
                            ]
                        self.history.append(message)
                        self._dirty = True
                    self._assistant_buf = ""
                    truncated = payload.get("finish") == "length"
                    if kind == "stopped":
                        label = "停止"
                    elif truncated:
                        label = "打ち切り"
                        self._append_system(
                            "max_tokens に達したため回答が途中で終わりました。"
                            "max_tokens を増やすか、「再生成」でやり直してください"
                        )
                    else:
                        label = "完了"
                    wait = payload.get("wait")
                    self._finish_generation(
                        f"{label} ({payload['tokens']} tokens, {payload['speed']:.1f} tok/s"
                        + (f", 応答開始まで {wait:.1f} 秒" if wait is not None else "") + ")"
                    )
                    if sources and self.history and self.history[-1].get("sources"):
                        self._append_sources(self.history[-1]["sources"])
                    self._save_current()        # 1往復ごとに自動保存
                    self._refresh_sidebar()
                    if structured_result:
                        # 画面の更新が済んでから保存先を聞く
                        self.root.after(50, lambda r=structured_result: self._save_structured(*r))
                elif kind == "error":
                    self._append_system(f"エラー: {payload}", error=True)
                    if "context window" in payload.lower():
                        self._append_system(
                            "入力と max_tokens の合計がコンテキスト長を超えています。"
                            "max_tokens を減らすか、LLM_N_CTX を大きくして起動し直してください"
                        )
                    self._assistant_buf = ""
                    self._structured = None
                    self._pending_sources = None
                    self._finish_generation("エラー")
                elif kind == "loaded":
                    self._set_busy(False)
                    self.status_var.set(payload)
                    self._append_system(payload)
                    self.load_btn.config(state="normal")
                    if self.engine.llm is not None:
                        self._set_model_state("読み込み済み", "ok")
                    else:
                        self._set_model_state("モックモード", "warn")
                    self._refresh_context_usage()
                    self.input.focus_set()
                elif kind == "load_error":
                    self._set_busy(False)
                    self.status_var.set("読み込み失敗")
                    self._set_model_state("読み込み失敗", "error")
                    self._append_system(f"モデル読み込み失敗: {payload}", error=True)
                    self.load_btn.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(POLL_INTERVAL_MS, self._poll_queue)

    def _repaint_chat(self):
        """history の内容をチャット表示に描き直す (会話再開時)。"""
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self._thumbnails = []          # 描き直すので古いサムネイルは捨てる
        for m in self.history:
            self._append_message(
                m["role"], plain_content(m["content"]), content_images(m["content"])
            )
            if m.get("sources"):
                self._append_sources(m["sources"])

    def _ensure_newline(self):
        """直前の出力が行の途中で終わっていれば改行する (見出しを行頭から始めるため)。"""
        if self.chat.get("1.0", "end-1c") and self.chat.get("end-2c", "end-1c") != "\n":
            self.chat.insert("end", "\n")

    def _append_message(self, role, text, images=()):
        label = {"user": "あなた", "assistant": "AI"}.get(role, role)
        tag = role if role in ("user", "assistant") else "assistant"
        self.chat.config(state="normal")
        self._ensure_newline()
        self.chat.insert("end", label + "\n", f"{tag}_head")
        if text:
            self.chat.insert("end", text, tag)
        for data in images:
            self._insert_thumbnail(data, tag)
        if role == "user":
            # 行末の改行まで背景色のタグに含めると、右端まで塗られて吹き出しのように見える
            self._ensure_newline()
            self.chat.tag_add(tag, "end-2c", "end-1c")
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _insert_thumbnail(self, data, tag="user"):
        """チャット欄に画像を小さく貼る。作れなければ何もしない。"""
        thumbnail = thumbnail_png(data)
        if thumbnail is None:
            return
        try:
            photo = tk.PhotoImage(data=base64.b64encode(thumbnail).decode("ascii"))
        except Exception:
            log.debug("[表示] サムネイルを表示できませんでした", exc_info=True)
            return
        # PhotoImage は参照が切れると表示が消えるため、アプリ側で持ち続ける
        self._thumbnails.append(photo)
        self._ensure_newline()
        self.chat.image_create("end", image=photo, padx=10, pady=4)
        self.chat.insert("end", "\n", tag)

    def _stream_token(self, piece, tag="assistant"):
        self.chat.config(state="normal")
        self.chat.insert("end", piece, tag)
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _apply_thought_visibility(self):
        """「思考を表示」に合わせて、思考タグの折りたたみを切り替える。"""
        self.chat.tag_config("thought", elide=not self.show_thought_var.get())

    def _append_system(self, text, error=False):
        """案内・警告をチャット欄に小さく出す (会話の履歴には含めない)。"""
        self.chat.config(state="normal")
        self._ensure_newline()
        self.chat.insert("end", f"{text}\n", "error" if error else "system")
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _finish_generation(self, status):
        self.generating = False
        self._stop_event.clear()
        self._set_busy(False)
        self.send_btn.config(text="送信", command=self.on_send, state="normal")
        self._refresh_regen_button()
        self.status_var.set(status)
        self._refresh_context_usage()
        self.chat.config(state="normal")
        self.chat.insert("end", "\n")
        self.chat.config(state="disabled")

    def _refresh_context_usage(self):
        """ステータスバー右端のコンテキスト使用量。"""
        usage = self.engine.context_usage()
        if not usage:
            self.ctx_var.set("")
            return
        used, total = usage
        percent = used * 100 // total if total else 0
        self.ctx_var.set(f"コンテキスト {used:,} / {total:,} ({percent}%)")
        self.ctx_label.config(style="StateWarn.TLabel" if percent >= 85 else "Muted.TLabel")


def main():
    log.info("=== ローカルLLM チャット 起動 ===")
    log.info("llama-cpp-python: %s", "利用可能" if HAS_LLAMA else "未導入 (モックモード)")
    root = tk.Tk()
    ChatApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
