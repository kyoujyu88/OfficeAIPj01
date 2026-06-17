#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ローカルLLM チャットアプリ (Tkinter / CPU・オフライン向け)

特徴:
  - Tkinter 製の GUI チャット (Python 標準ライブラリのみ。追加インストール不要)
  - llama-cpp-python で GGUF モデルを CPU 推論
  - 生成トークンをストリーミング表示 (生成中に逐次表示)
  - モデル選択 / temperature / max_tokens を UI から調整
  - 左サイドバーに会話履歴を一覧表示
      * クリックで過去の会話を再開
      * チェックを入れて選択した会話をまとめて削除
  - 会話は JSON ファイルとして自動保存 (chat_sessions/ フォルダ)
  - ターミナルに動作状況 (状態遷移・性能) をデバッグ出力

備考:
  llama-cpp-python が未導入、またはモデルファイルが見つからない場合は
  「モックモード」で起動し、UI と挙動の確認だけは行えます (実推論は行いません)。

実行:
  python chat_app.py
  # モデルの置き場所を指定する場合:
  LLM_MODELS_DIR=/path/to/models python chat_app.py
"""

import os
import sys
import json
import time
import queue
import logging
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------
# モデル (.gguf) を置いているフォルダ。環境変数 LLM_MODELS_DIR で上書き可。
MODELS_DIR = Path(os.environ.get("LLM_MODELS_DIR", ".")).expanduser()

# 会話履歴 (JSON) の保存先。スクリプトと同じ場所の chat_sessions/ 。
SESSIONS_DIR = Path(__file__).resolve().parent / "chat_sessions"

# 既定の推論パラメータ (CPU 向けの控えめな値)
DEFAULT_N_CTX = 4096
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_N_THREADS = os.cpu_count() or 4

# サイドバーのタイトル表示の最大文字数
TITLE_MAXLEN = 20

# 停止語 (これが現れたら生成を打ち切る)。
# モデルが「あなた:」等と続きを勝手に生成する“一人芝居”を防ぐ。
STOP_WORDS = [
    "<|im_end|>", "<|im_start|>",   # ChatML (Yi-Coder 等)
    "\nあなた:", "\nUser:", "\nuser:",
    "あなた:", "User:",
]

# UI がワーカースレッドからの出力を取りに行く間隔 (ミリ秒)
POLL_INTERVAL_MS = 40

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

    def load(self, model_name, n_ctx=DEFAULT_N_CTX, n_threads=DEFAULT_N_THREADS):
        """モデルを読み込む。成功で True、モックで False を返す。"""
        path = MODELS_DIR / model_name
        if not HAS_LLAMA or not path.exists():
            self.llm = None
            self.model_name = model_name
            reason = "llama-cpp-python 未導入" if not HAS_LLAMA else "ファイル未検出"
            log.warning("モックモードでロード: %s (%s)", model_name, reason)
            return False

        log.info("モデル読み込み開始: %s (n_ctx=%d, n_threads=%d)", model_name, n_ctx, n_threads)
        t0 = time.time()
        self.llm = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        self.model_name = model_name
        log.info("モデル読み込み完了: %.1f 秒", time.time() - t0)
        return True

    def stream(self, messages, max_tokens, temperature):
        """応答トークンを順次 yield するジェネレータ。"""
        if self.llm is None:
            yield from self._mock_stream(messages)
            return
        completion = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=STOP_WORDS,
            stream=True,
        )
        for chunk in completion:
            delta = chunk["choices"][0]["delta"]
            piece = delta.get("content")
            if piece:
                yield piece

    @staticmethod
    def _mock_stream(messages):
        """UI 確認用のダミー応答 (1文字ずつ返す)。"""
        user = messages[-1]["content"] if messages else ""
        text = (
            f"[モック応答] 受け取りました:「{user}」\n"
            "これは UI 動作確認用のダミー応答です。"
            "実モデルを読み込むと、ここに生成結果がストリーミング表示されます。"
        )
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
        self.current_id = None         # 現在の会話 ID (未保存なら None)
        self.current_created = None    # 現在の会話の作成時刻
        self.session_rows = []         # サイドバー行 [(BooleanVar, meta), ...]

        self.token_queue = queue.Queue()
        self.generating = False
        self._assistant_buf = ""       # ストリーミング中のアシスタント発話バッファ

        self._build_ui()
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

        # オプション段: temperature / max_tokens
        opt = ttk.Frame(right, padding=(8, 0))
        opt.pack(fill="x")
        ttk.Label(opt, text="temperature:").pack(side="left")
        self.temp_var = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        ttk.Spinbox(
            opt, from_=0.0, to=2.0, increment=0.1, width=5, textvariable=self.temp_var
        ).pack(side="left", padx=(2, 10))
        ttk.Label(opt, text="max_tokens:").pack(side="left")
        self.maxtok_var = tk.IntVar(value=DEFAULT_MAX_TOKENS)
        ttk.Spinbox(
            opt, from_=16, to=4096, increment=16, width=6, textvariable=self.maxtok_var
        ).pack(side="left", padx=2)

        # 中段: チャット履歴表示
        self.chat = scrolledtext.ScrolledText(
            right, wrap="word", state="disabled", font=("", 11)
        )
        self.chat.pack(fill="both", expand=True, padx=8, pady=6)
        self.chat.tag_config("user", foreground="#1a7f37", font=("", 11, "bold"))
        self.chat.tag_config("assistant", foreground="#0a3069")
        self.chat.tag_config("system", foreground="#999999", font=("", 9, "italic"))

        # 下段: 入力欄 + 送信
        bottom = ttk.Frame(right, padding=(8, 6))
        bottom.pack(fill="x")
        self.input = tk.Text(bottom, height=3, wrap="word", font=("", 11))
        self.input.pack(side="left", fill="x", expand=True)
        # Enter で送信 / Shift+Enter で改行
        self.input.bind("<Return>", lambda e: self.on_send())
        self.input.bind("<Shift-Return>", self._insert_newline)
        self.send_btn = ttk.Button(bottom, text="送信\n(Enter)", command=self.on_send)
        self.send_btn.pack(side="left", padx=4, fill="y")

    # ---- 会話履歴 (サイドバー) ------------------------------------------
    def _make_title(self):
        """最初のユーザー発話から会話タイトルを作る。"""
        for m in self.history:
            if m["role"] == "user":
                t = m["content"].strip().replace("\n", " ")
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
            "messages": self.history,
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
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")

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
        self._repaint_chat()
        self._refresh_sidebar()
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
        ok = self.engine.load(name)
        tag = "ロード完了" if ok else "モックモード (実推論なし)"
        self.token_queue.put(("loaded", f"{tag}: {name} ({time.time() - t0:.1f}s)"))

    def on_send(self):
        if self.generating:
            log.debug("[UI] 生成中のため送信を無視")
            return "break"
        text = self.input.get("1.0", "end").strip()
        if not text:
            return "break"
        if self.engine.model_name is None:
            self._append_system("先にモデルを読み込んでください")
            log.warning("[UI] モデル未読み込みで送信されました")
            return "break"

        self.input.delete("1.0", "end")
        self.history.append({"role": "user", "content": text})
        self._append_message("user", text)
        log.info("[UI] 送信: %d 文字 / 履歴 %d 件", len(text), len(self.history))

        self.generating = True
        self.send_btn.config(state="disabled")
        self.status_var.set("生成中 ...")
        self._append_message("assistant", "")  # "AI: " の見出しだけ先に表示

        temp = float(self.temp_var.get())
        max_tok = int(self.maxtok_var.get())
        threading.Thread(
            target=self._gen_worker,
            args=(list(self.history), max_tok, temp),
            name="gen",
            daemon=True,
        ).start()
        return "break"

    def _gen_worker(self, messages, max_tokens, temperature):
        log.info("[GEN] 生成開始 (max_tokens=%d, temperature=%.2f)", max_tokens, temperature)
        t0 = time.time()
        first = None
        n = 0
        try:
            for piece in self.engine.stream(messages, max_tokens, temperature):
                if first is None:
                    first = time.time()
                    log.info("[GEN] 初トークンまで %.2fs", first - t0)
                n += 1
                self.token_queue.put(("token", piece))
                if n % 20 == 0:
                    dt = time.time() - (first or t0)
                    log.debug("[GEN] 経過 %d tokens, %.1f tok/s", n, n / dt if dt > 0 else 0.0)
            dt = time.time() - (first or t0)
            speed = n / dt if dt > 0 else 0.0
            log.info("[GEN] 完了: %d tokens, 総 %.2fs, %.1f tok/s", n, time.time() - t0, speed)
            self.token_queue.put(("end", n))
        except Exception as e:  # 推論中の例外もターミナルに出す
            log.exception("[GEN] 生成中にエラー")
            self.token_queue.put(("error", str(e)))

    def _insert_newline(self, event):
        """Shift+Enter: 送信せず改行を挿入する。"""
        self.input.insert("insert", "\n")
        return "break"

    def on_close(self):
        """ウィンドウを閉じる前に現在の会話を保存。"""
        try:
            self._save_current()
        finally:
            log.info("=== 終了 ===")
            self.root.destroy()

    # ---- メインスレッドでの描画更新 -------------------------------------
    def _poll_queue(self):
        """ワーカースレッドからの出力をメインスレッドで反映する。"""
        try:
            while True:
                kind, payload = self.token_queue.get_nowait()
                if kind == "token":
                    self._assistant_buf += payload
                    self._stream_token(payload)
                elif kind == "end":
                    self.history.append({"role": "assistant", "content": self._assistant_buf})
                    self._assistant_buf = ""
                    self._finish_generation("完了")
                    self._save_current()        # 1往復ごとに自動保存
                    self._refresh_sidebar()
                elif kind == "error":
                    self._append_system(f"エラー: {payload}")
                    self._assistant_buf = ""
                    self._finish_generation("エラー")
                elif kind == "loaded":
                    self.status_var.set(payload)
                    self._append_system(payload)
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
            self._append_message(m["role"], m["content"])

    def _append_message(self, role, text):
        label = {"user": "あなた", "assistant": "AI"}.get(role, role)
        self.chat.config(state="normal")
        self.chat.insert("end", f"\n{label}: ", role)
        if text:
            self.chat.insert("end", text, role)
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _stream_token(self, piece):
        self.chat.config(state="normal")
        self.chat.insert("end", piece, "assistant")
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _append_system(self, text):
        self.chat.config(state="normal")
        self.chat.insert("end", f"\n[システム] {text}\n", "system")
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _finish_generation(self, status):
        self.generating = False
        self.send_btn.config(state="normal")
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
