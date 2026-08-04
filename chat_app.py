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
      * クリックで過去の会話を再開
      * チェックを入れて選択した会話をまとめて削除
  - 会話は JSON ファイルとして自動保存 (chat_sessions/ フォルダ)
  - ファイル添付 (画像 / PDF / Word / Excel / テキスト系)
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
  # モデルの置き場所を指定する場合:
  LLM_MODELS_DIR=/path/to/models python chat_app.py
  # mmproj を明示指定する場合:
  LLM_MMPROJ=/path/to/mmproj-gemma-4-E4B.gguf python chat_app.py
"""

import os
import sys
import json
import time
import queue
import base64
import logging
import mimetypes
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------
# モデル (.gguf) を置いているフォルダ。環境変数 LLM_MODELS_DIR で上書き可。
MODELS_DIR = Path(os.environ.get("LLM_MODELS_DIR", ".")).expanduser()

# 会話履歴 (JSON) の保存先。スクリプトと同じ場所の chat_sessions/ 。
SESSIONS_DIR = Path(__file__).resolve().parent / "chat_sessions"

# アプリ設定 (システムプロンプト・サンプリング値など) の保存先。
# 会話ごとの設定は各会話の JSON 側に持つ。こちらは「新しいチャットの初期値」。
SETTINGS_PATH = Path(__file__).resolve().parent / "chat_settings.json"

# マルチモーダル投影ファイル。未指定ならモデルフォルダから mmproj*.gguf を探す。
MMPROJ_OVERRIDE = os.environ.get("LLM_MMPROJ")

# 既定の推論パラメータ。
# 添付ファイルや思考モードで入力・出力とも長くなるため、環境変数で調整できる。
DEFAULT_N_CTX = int(os.environ.get("LLM_N_CTX", 16384))
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", 2048))
DEFAULT_N_THREADS = int(os.environ.get("LLM_N_THREADS", 0)) or os.cpu_count() or 4

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


def extract_document_text(path):
    """文書ファイルからテキストを取り出す。読めない場合は RuntimeError。"""
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
    if len(text) > MAX_DOC_CHARS:
        omitted = len(text) - MAX_DOC_CHARS
        log.warning("[添付] %s が長いため %d 文字を省略しました", path.name, omitted)
        text = text[:MAX_DOC_CHARS] + f"\n…(以降 {omitted} 文字を省略)"
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


def image_attachment(name, data, mime="image/png"):
    """画像バイト列を data URI 形式の添付エントリにする。"""
    uri = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    return {"kind": "image", "name": name, "data_uri": uri}


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


def plain_content(content):
    """content (str または OpenAI 形式のパート配列) を表示・保存用の文字列にする。"""
    if isinstance(content, str):
        return content
    texts = [
        part.get("text", "")
        for part in (content or [])
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    return "\n".join(t for t in texts if t)


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

        log.info("モデル読み込み開始: %s (n_ctx=%d, n_threads=%d)", model_name, n_ctx, n_threads)
        t0 = time.time()
        chat_handler = self._build_chat_handler()
        try:
            llm = Llama(
                model_path=str(path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                chat_handler=chat_handler,
                verbose=False,
            )
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
        log.info(
            "モデル読み込み完了: %.1f 秒 (画像入力 %s)",
            time.time() - t0, "有効" if self.vision else "無効",
        )
        return True

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

    def stream(self, messages, max_tokens, temperature, top_p=DEFAULT_TOP_P, top_k=DEFAULT_TOP_K):
        """応答トークンを順次 yield するジェネレータ。"""
        self.last_finish_reason = None
        if self.llm is None:
            if self.load_error:
                raise RuntimeError(f"モデルが読み込まれていません: {self.load_error}")
            yield from self._mock_stream(messages)
            return
        completion = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop_words_for(self.model_name),
            stream=True,
        )
        for chunk in completion:
            choice = chunk["choices"][0]
            # "length" なら max_tokens 到達で打ち切られている (回答が途中で切れる原因)
            self.last_finish_reason = choice.get("finish_reason") or self.last_finish_reason
            piece = choice["delta"].get("content")
            if piece:
                yield piece

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
    def _mock_stream(messages):
        """UI 確認用のダミー応答 (1文字ずつ返す)。"""
        user = plain_content(messages[-1]["content"]) if messages else ""
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
class ChatApp:
    def __init__(self, root):
        self.root = root
        self.engine = LLMEngine()
        self.store = SessionStore(SESSIONS_DIR)

        self.history = []              # [{"role": ..., "content": ...}]
        self.attachments = []          # 次の送信に添付するファイル [{kind, name, ...}]
        self.current_id = None         # 現在の会話 ID (未保存なら None)
        self.current_created = None    # 現在の会話の作成時刻
        self.session_rows = []         # サイドバー行 [(BooleanVar, meta), ...]

        self.token_queue = queue.Queue()
        self.generating = False
        self._stop_event = threading.Event()   # 生成の打ち切り要求
        self._assistant_buf = ""       # ストリーミング中のアシスタント発話バッファ

        self._build_ui()
        self._load_settings()
        self._refresh_sidebar()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(POLL_INTERVAL_MS, self._poll_queue)
        log.info(
            "アプリ起動完了 (CPUスレッド=%d, モデルフォルダ=%s)",
            DEFAULT_N_THREADS, MODELS_DIR.resolve(),
        )

    # ---- UI 構築 ---------------------------------------------------------
    def _build_ui(self):
        self.root.title("ローカルLLM チャット (CPU / オフライン)")
        self.root.geometry("1000x640")
        self.root.minsize(720, 460)

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        # ===== 左サイドバー: 会話履歴 =====
        left = ttk.Frame(main, width=240)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        ttk.Button(left, text="＋ 新規チャット", command=self.on_new_chat).pack(
            fill="x", padx=6, pady=(8, 4)
        )
        ttk.Label(left, text="会話履歴", foreground="#666666").pack(anchor="w", padx=8)

        # スクロール可能なリスト領域 (Canvas + 内部 Frame)
        list_wrap = ttk.Frame(left)
        list_wrap.pack(fill="both", expand=True, padx=4, pady=4)
        self.list_canvas = tk.Canvas(list_wrap, highlightthickness=0, width=224)
        vsb = ttk.Scrollbar(list_wrap, orient="vertical", command=self.list_canvas.yview)
        self.list_frame = ttk.Frame(self.list_canvas)
        self.list_frame.bind(
            "<Configure>",
            lambda e: self.list_canvas.configure(scrollregion=self.list_canvas.bbox("all")),
        )
        self.list_canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        self.list_canvas.configure(yscrollcommand=vsb.set)
        self.list_canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        ttk.Button(left, text="選択した履歴を削除", command=self.on_delete_selected).pack(
            fill="x", padx=6, pady=(4, 8)
        )

        # ===== 右側: チャット本体 =====
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        # 上段: モデル選択 + 読み込み + ステータス
        top = ttk.Frame(right, padding=(8, 6))
        top.pack(fill="x")
        ttk.Label(top, text="モデル:").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            top, textvariable=self.model_var, state="readonly", width=42
        )
        self.model_combo["values"] = discover_models()
        if self.model_combo["values"]:
            self.model_combo.current(0)
        self.model_combo.pack(side="left", padx=4)
        self.load_btn = ttk.Button(top, text="読み込み", command=self.on_load)
        self.load_btn.pack(side="left", padx=4)
        self.status_var = tk.StringVar(value="未読み込み")
        ttk.Label(top, textvariable=self.status_var, foreground="#0066cc").pack(
            side="left", padx=8
        )

        # オプション段 1: サンプリング (既定値は Gemma 4 の推奨値)
        opt = ttk.Frame(right, padding=(8, 0))
        opt.pack(fill="x")
        ttk.Label(opt, text="temperature:").pack(side="left")
        self.temp_var = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        ttk.Spinbox(
            opt, from_=0.0, to=2.0, increment=0.1, width=5, textvariable=self.temp_var
        ).pack(side="left", padx=(2, 10))
        ttk.Label(opt, text="top_p:").pack(side="left")
        self.top_p_var = tk.DoubleVar(value=DEFAULT_TOP_P)
        ttk.Spinbox(
            opt, from_=0.0, to=1.0, increment=0.05, width=5, textvariable=self.top_p_var
        ).pack(side="left", padx=(2, 10))
        ttk.Label(opt, text="top_k:").pack(side="left")
        self.top_k_var = tk.IntVar(value=DEFAULT_TOP_K)
        ttk.Spinbox(
            opt, from_=1, to=200, increment=1, width=5, textvariable=self.top_k_var
        ).pack(side="left", padx=(2, 10))
        ttk.Label(opt, text="max_tokens:").pack(side="left")
        self.maxtok_var = tk.IntVar(value=DEFAULT_MAX_TOKENS)
        ttk.Spinbox(
            opt, from_=16, to=8192, increment=16, width=6, textvariable=self.maxtok_var
        ).pack(side="left", padx=2)

        # オプション段 2: システムプロンプト / 思考モード
        sysrow = ttk.Frame(right, padding=(8, 4))
        sysrow.pack(fill="x")
        ttk.Label(sysrow, text="システムプロンプト:").pack(side="left")
        self.system_var = tk.StringVar(value="")
        ttk.Entry(sysrow, textvariable=self.system_var).pack(
            side="left", fill="x", expand=True, padx=(4, 8)
        )
        # 思考モード: システムプロンプト先頭に <|think|> を付けて有効化する
        self.thinking_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sysrow, text="思考モード", variable=self.thinking_var).pack(side="left")
        self.show_thought_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            sysrow, text="思考を表示", variable=self.show_thought_var,
            command=self._apply_thought_visibility,
        ).pack(side="left", padx=(4, 0))

        # 中段: チャット履歴表示
        self.chat = scrolledtext.ScrolledText(
            right, wrap="word", state="disabled", font=("", 11)
        )
        self.chat.pack(fill="both", expand=True, padx=8, pady=6)
        self.chat.tag_config("user", foreground="#1a7f37", font=("", 11, "bold"))
        self.chat.tag_config("assistant", foreground="#0a3069")
        self.chat.tag_config("system", foreground="#999999", font=("", 9, "italic"))
        self.chat.tag_config("thought", foreground="#8250df", font=("", 9, "italic"))
        self._apply_thought_visibility()

        # 下段: 添付行 + 入力欄 + 送信
        bottom = ttk.Frame(right, padding=(8, 6))
        bottom.pack(fill="x")

        attach_row = ttk.Frame(bottom)
        attach_row.pack(fill="x", pady=(0, 4))
        self.attach_btn = ttk.Button(attach_row, text="ファイル添付", command=self.on_attach)
        self.attach_btn.pack(side="left")
        self.attach_clear_btn = ttk.Button(
            attach_row, text="解除", command=self.on_clear_attachments, state="disabled"
        )
        self.attach_clear_btn.pack(side="left", padx=4)
        # PDF は既定でテキスト優先 (抽出できなければ自動で画像化)。
        # チェックすると常にページ画像として渡す (図表やレイアウトを見せたいとき)。
        self.pdf_as_image_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            attach_row, text="PDFを画像として読む", variable=self.pdf_as_image_var
        ).pack(side="left", padx=(8, 0))
        self.attach_var = tk.StringVar(value="添付なし")
        ttk.Label(attach_row, textvariable=self.attach_var, foreground="#666666").pack(
            side="left", padx=6
        )
        # 生成の制御は右寄せ (生成中のみ停止、生成後のみ再生成が押せる)
        self.stop_btn = ttk.Button(
            attach_row, text="停止", command=self.on_stop, state="disabled"
        )
        self.stop_btn.pack(side="right")
        self.regen_btn = ttk.Button(
            attach_row, text="再生成", command=self.on_regenerate, state="disabled"
        )
        self.regen_btn.pack(side="right", padx=4)

        entry_row = ttk.Frame(bottom)
        entry_row.pack(fill="x")
        self.input = tk.Text(entry_row, height=3, wrap="word", font=("", 11))
        self.input.pack(side="left", fill="x", expand=True)
        # Enter で送信 / Shift+Enter で改行
        self.input.bind("<Return>", lambda e: self.on_send())
        self.input.bind("<Shift-Return>", self._insert_newline)
        self.send_btn = ttk.Button(entry_row, text="送信\n(Enter)", command=self.on_send)
        self.send_btn.pack(side="left", padx=4, fill="y")

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
        }

    def _load_settings(self):
        """前回終了時の設定を復元する。無ければ既定値のまま。"""
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            log.info("[設定] 保存済みの設定はありません (既定値で起動)")
            return
        except Exception as e:
            log.warning("[設定] 読み込みに失敗しました (%s) -> 既定値で起動", e)
            return
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
        log.info("[設定] 復元: %s", SETTINGS_PATH)

    def _save_settings(self):
        """現在の設定を次回起動用に保存する。"""
        data = {key: cast(var.get()) for key, (var, cast) in self._settings_fields().items()}
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
                t = plain_content(m["content"]).strip().replace("\n", " ")
                return (t[:TITLE_MAXLEN] + "…") if len(t) > TITLE_MAXLEN else t
        return "新しいチャット"

    def _save_current(self):
        """現在の会話を保存する (空なら何もしない)。"""
        if not self.history:
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
            # 画像は base64 のまま保存すると JSON が肥大するため、本文だけを残す。
            # (会話を再開すると画像はモデルに渡らず、[添付画像: 名前] の記述だけが残る)
            "messages": [dict(m, content=plain_content(m["content"])) for m in self.history],
        })

    def _refresh_sidebar(self):
        """保存済み会話の一覧を再描画する。"""
        for child in self.list_frame.winfo_children():
            child.destroy()
        self.session_rows = []
        for meta in self.store.list_meta():
            row = ttk.Frame(self.list_frame)
            row.pack(fill="x", pady=1)
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(row, variable=var).pack(side="left")
            title = meta["title"] or "(無題)"
            if meta["id"] == self.current_id:
                title = "▶ " + title          # 現在開いている会話に印
            ttk.Button(
                row, text=title, width=22,
                command=lambda sid=meta["id"]: self.on_load_session(sid),
            ).pack(side="left", fill="x", expand=True)
            self.session_rows.append((var, meta))

    def _start_new_session(self):
        self.history = []
        self.current_id = None
        self.current_created = None
        self.attachments = []
        self._refresh_attachments()
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self._refresh_regen_button()

    def _refresh_regen_button(self):
        """再生成できる応答があるときだけボタンを有効にする。"""
        enabled = not self.generating and any(m["role"] == "assistant" for m in self.history)
        self.regen_btn.config(state="normal" if enabled else "disabled")

    def on_new_chat(self):
        if self.generating:
            return
        self._save_current()          # 開いていた会話を保存
        self._start_new_session()
        self._refresh_sidebar()
        self._append_system("新しいチャットを開始しました")
        log.info("[UI] 新規チャット")

    def on_load_session(self, session_id):
        """サイドバーの会話をクリック -> 再開。"""
        if self.generating:
            return
        self._save_current()          # 今の会話を保存してから切り替え
        data = self.store.load(session_id)
        self.history = data.get("messages", [])
        self.current_id = data["id"]
        self.current_created = data.get("created", time.time())
        self.system_var.set(data.get("system_prompt", ""))
        self.thinking_var.set(bool(data.get("thinking", False)))
        self._repaint_chat()
        self._refresh_sidebar()
        self._refresh_regen_button()
        self.status_var.set(f"会話を再開: {data.get('title', '')}")
        log.info("[UI] 会話を再開: %s (%d発話)", session_id, len(self.history))

    def on_delete_selected(self):
        """チェックされた会話をまとめて削除。"""
        if self.generating:
            return
        ids = [meta["id"] for var, meta in self.session_rows if var.get()]
        if not ids:
            self._append_system("削除する履歴にチェックを入れてください")
            return
        if not messagebox.askyesno("確認", f"{len(ids)} 件の会話を削除します。よろしいですか?"):
            return
        for sid in ids:
            self.store.delete(sid)
            if sid == self.current_id:
                self._start_new_session()
        self._refresh_sidebar()
        log.info("[UI] %d 件の履歴を削除", len(ids))

    # ---- ハンドラ --------------------------------------------------------
    def on_load(self):
        name = self.model_var.get()
        if not name:
            return
        self.load_btn.config(state="disabled")
        self.status_var.set(f"読み込み中: {name} ...")
        self._append_system(f"モデル読み込み中: {name}")
        log.info("[UI] 読み込みボタン押下 -> %s", name)
        threading.Thread(
            target=self._load_worker, args=(name,), name="loader", daemon=True
        ).start()

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
        for raw in paths:
            path = Path(raw)
            try:
                self.attachments.extend(self._make_attachments(path))
            except Exception as e:
                log.warning("[添付] 追加できません: %s (%s)", path.name, e)
                self._append_system(f"添付できません: {path.name} ({e})")
        self._refresh_attachments()

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

    def on_clear_attachments(self):
        if self.generating or not self.attachments:
            return
        log.info("[添付] %d 件を解除", len(self.attachments))
        self.attachments = []
        self._refresh_attachments()

    def _refresh_attachments(self):
        """添付一覧のラベルと解除ボタンの状態を更新する。"""
        if not self.attachments:
            self.attach_var.set("添付なし")
            self.attach_clear_btn.config(state="disabled")
            return
        names = ", ".join(a["name"] for a in self.attachments)
        if len(names) > ATTACH_LABEL_MAXLEN:
            names = names[:ATTACH_LABEL_MAXLEN] + "…"
        self.attach_var.set(f"添付 {len(self.attachments)} 件: {names}")
        self.attach_clear_btn.config(state="normal")

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
        blocks.extend(f"[添付画像: {img['name']}]" for img in images)
        merged = "\n\n".join(blocks)

        if not images:
            return merged
        parts = [{"type": "text", "text": merged}]
        parts.extend(
            {"type": "image_url", "image_url": {"url": img["data_uri"]}} for img in images
        )
        return parts

    # ---- 送信 ------------------------------------------------------------
    def on_send(self):
        if self.generating:
            log.debug("[UI] 生成中のため送信を無視")
            return "break"
        text = self.input.get("1.0", "end").strip()
        if not text and not self.attachments:
            return "break"
        if self.engine.model_name is None:
            self._append_system("先にモデルを読み込んでください")
            log.warning("[UI] モデル未読み込みで送信されました")
            return "break"

        content = self._build_content(text)
        if isinstance(content, str) and not content.strip():
            # 添付が全て除外された (画像を渡せない構成で画像だけ添付した等)
            self._append_system("送信できる内容がありません")
            return "break"

        self.input.delete("1.0", "end")
        self.attachments = []
        self._refresh_attachments()
        self.history.append({"role": "user", "content": content})
        self._append_message("user", plain_content(content))
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
        self._repaint_chat()
        log.info("[UI] 再生成 (履歴 %d 件から)", len(self.history))
        self._start_generation()

    def on_stop(self):
        """生成中のワーカーに停止を伝える。そこまでの応答は残す。"""
        if not self.generating:
            return
        self._stop_event.set()
        self.stop_btn.config(state="disabled")
        self.status_var.set("停止中 ...")
        log.info("[UI] 停止要求")

    def _system_message(self):
        """システムプロンプト (思考モードなら先頭に <|think|>) を組み立てる。"""
        text = self.system_var.get().strip()
        if self.thinking_var.get():
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
        self.generating = True
        self._stop_event.clear()
        self.send_btn.config(state="disabled")
        self.regen_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("生成中 ...")
        self._append_message("assistant", "")  # "AI: " の見出しだけ先に表示

        messages = list(self.history)
        system = self._system_message()
        if system:
            messages.insert(0, system)
        max_tokens = int(self.maxtok_var.get())
        if self.thinking_var.get():
            # 思考は回答と同じ予算を消費するので、その分を上乗せする
            max_tokens += THINKING_EXTRA_TOKENS
        params = {
            "max_tokens": max_tokens,
            "temperature": float(self.temp_var.get()),
            "top_p": float(self.top_p_var.get()),
            "top_k": int(self.top_k_var.get()),
        }
        threading.Thread(
            target=self._gen_worker, args=(messages, params), name="gen", daemon=True,
        ).start()

    def _gen_worker(self, messages, params):
        log.info(
            "[GEN] 生成開始 (max_tokens=%(max_tokens)d, temperature=%(temperature).2f,"
            " top_p=%(top_p).2f, top_k=%(top_k)d)", params,
        )
        t0 = time.time()
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
                {"tokens": n, "speed": speed, "finish": finish},
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
                    self._assistant_buf += payload
                    self._stream_token(payload, "assistant")
                elif kind == "thought":
                    # 思考は表示するだけで履歴には残さない (次のターンへは渡さない)
                    self._stream_token(payload, "thought")
                elif kind in ("end", "stopped"):
                    if self._assistant_buf:
                        self.history.append(
                            {"role": "assistant", "content": self._assistant_buf}
                        )
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
                    self._finish_generation(
                        f"{label} ({payload['tokens']} tokens, {payload['speed']:.1f} tok/s)"
                    )
                    self._save_current()        # 1往復ごとに自動保存
                    self._refresh_sidebar()
                elif kind == "error":
                    self._append_system(f"エラー: {payload}")
                    if "context window" in payload.lower():
                        self._append_system(
                            "入力と max_tokens の合計がコンテキスト長を超えています。"
                            "max_tokens を減らすか、LLM_N_CTX を大きくして起動し直してください"
                        )
                    self._assistant_buf = ""
                    self._finish_generation("エラー")
                elif kind == "loaded":
                    self.status_var.set(payload)
                    self._append_system(payload)
                    self.load_btn.config(state="normal")
                elif kind == "load_error":
                    self.status_var.set("読み込み失敗")
                    self._append_system(f"モデル読み込み失敗: {payload}")
                    self.load_btn.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(POLL_INTERVAL_MS, self._poll_queue)

    def _repaint_chat(self):
        """history の内容をチャット表示に描き直す (会話再開時)。"""
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        for m in self.history:
            self._append_message(m["role"], plain_content(m["content"]))

    def _append_message(self, role, text):
        label = {"user": "あなた", "assistant": "AI"}.get(role, role)
        self.chat.config(state="normal")
        self.chat.insert("end", f"\n{label}: ", role)
        if text:
            self.chat.insert("end", text, role)
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _stream_token(self, piece, tag="assistant"):
        self.chat.config(state="normal")
        self.chat.insert("end", piece, tag)
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _apply_thought_visibility(self):
        """「思考を表示」に合わせて、思考タグの折りたたみを切り替える。"""
        self.chat.tag_config("thought", elide=not self.show_thought_var.get())

    def _append_system(self, text):
        self.chat.config(state="normal")
        self.chat.insert("end", f"\n[システム] {text}\n", "system")
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _finish_generation(self, status):
        self.generating = False
        self._stop_event.clear()
        self.send_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._refresh_regen_button()
        usage = self.engine.context_usage()
        if usage:
            used, total = usage
            status = f"{status} / コンテキスト {used}/{total} ({used * 100 // total}%)"
        self.status_var.set(status)
        self.chat.config(state="normal")
        self.chat.insert("end", "\n")
        self.chat.config(state="disabled")


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
