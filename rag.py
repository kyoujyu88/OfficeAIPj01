#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料検索 (RAG) — 社内資料のフォルダから、質問に関係する部分を探して回答に使う。

chat_app.py から使うほか、単体でも索引の作成と検索の確認ができる。

  python rag.py build <資料フォルダ> --model <埋め込みモデルのフォルダ>
  python rag.py search "有給休暇の申請方法" [--top 5]
  python rag.py info

設計 (詳細は HANDOVER.md 9.1 / 6.12):
  - 埋め込みは sentence-transformers を直接使う (LangChain は使わない)
  - 検索は正規化済みベクトルの内積 (numpy)。数万チャンクまでなら一瞬で終わる
  - 回答に渡すのは上位 3 件 x 400 字、合計 1500 字まで。CPU では「取ってきた文を
    読ませる時間 (prefill)」が一番重いため、検索の精度より量を絞ることを優先する
  - 出典 (ファイル名・ページ) を必ず付ける。小型モデルは幻覚しやすいので、
    根拠を人が確かめられることが要
  - 索引は差分更新する (更新日時とサイズが変わったファイルだけ読み直す)

重いライブラリ (sentence-transformers / torch) は使うときにだけ読み込む。
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

log = logging.getLogger("chat_app.rag")

# 索引の保存先 (chat_app.py と同じ場所の rag_index/)
INDEX_DIR = Path(__file__).resolve().parent / "rag_index"

# チャンク (検索の単位) の大きさ。文の切れ目で区切るため、実際はこれより少し短くなる
CHUNK_CHARS = int(os.environ.get("LLM_RAG_CHUNK_CHARS", 400))
CHUNK_OVERLAP = 80                  # 前のチャンクと重ねる文字数 (文が途中で切れても拾えるように)
TOP_K = int(os.environ.get("LLM_RAG_TOP_K", 3))
MAX_CONTEXT_CHARS = int(os.environ.get("LLM_RAG_MAX_CHARS", 1500))
EMBED_BATCH = 16                    # 埋め込みを計算するときの一度の件数
# これより似ていない抜粋は使わない (関係の薄い資料が出典に並ばないように)。
# 値の目安はモデルで大きく違うため、既定では使わない。設定画面の「検索テスト」で
# 関係する資料と関係しない資料の点数を見てから決めること
MIN_SCORE = float(os.environ.get("LLM_RAG_MIN_SCORE", "-1"))

# 索引に入れる拡張子 (画像は除く。chat_app の添付と同じ読み取り関数を使う)
INDEXABLE_SUFFIXES = {
    ".pdf", ".docx", ".xlsx", ".xlsm",
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".log",
    ".yaml", ".yml", ".ini", ".xml", ".html", ".sql",
}

# 索引の形式。変えたら古い索引は作り直す
INDEX_VERSION = 1

# 文の切れ目 (ここで区切ると読みやすいチャンクになる)
_SENTENCE_ENDS = "。！？!?\n"


# --------------------------------------------------------------------------
# 埋め込みモデル
# --------------------------------------------------------------------------
def prefixes_for(model_name):
    """モデルごとの接頭辞 (検索文, 資料)。付け忘れると精度が大きく落ちる。

    - Ruri v3:            "検索クエリ: " / "検索文書: "
    - Ruri v1 / v2:       "クエリ: " / "文章: "
    - multilingual-e5 系: "query: " / "passage: "
    """
    name = (model_name or "").lower()
    if "ruri" in name:
        if "v3" in name:
            return "検索クエリ: ", "検索文書: "
        return "クエリ: ", "文章: "
    if "e5" in name:
        return "query: ", "passage: "
    return "", ""


class Embedder:
    """sentence-transformers の埋め込みモデル。フォルダ (または名前) を指定して読む。"""

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.name = Path(self.model_path).name or self.model_path
        self.query_prefix, self.doc_prefix = prefixes_for(self.name)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "資料検索には sentence-transformers が必要です (pip install sentence-transformers)"
            )
        log.info("[RAG] 埋め込みモデルを読み込み中: %s", self.model_path)
        t0 = time.time()
        try:
            self.model = SentenceTransformer(self.model_path, device="cpu")
        except Exception as e:
            hint = ""
            if "ruri" in self.name.lower() and "v3" in self.name.lower():
                hint = " (Ruri v3 には transformers 4.48 以降が必要です: pip install -U transformers)"
            raise RuntimeError(f"埋め込みモデルを読み込めません: {e}{hint}") from e
        log.info("[RAG] 埋め込みモデル読み込み完了: %.1f 秒", time.time() - t0)

    def _encode(self, texts):
        import numpy as np

        vectors = self.model.encode(
            list(texts), batch_size=EMBED_BATCH, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def encode_documents(self, texts):
        return self._encode(self.doc_prefix + t for t in texts)

    def encode_query(self, text):
        return self._encode([self.query_prefix + text])[0]


# --------------------------------------------------------------------------
# 資料の読み取りとチャンク分け
# --------------------------------------------------------------------------
def split_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """文章を size 文字程度のチャンクに分ける。できるだけ文の切れ目で区切る。"""
    text = "\n".join(line.strip() for line in (text or "").splitlines())
    text = "\n".join(filter(None, text.split("\n")))       # 空行を詰める
    if len(text) <= size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            # 後ろ半分にある最後の文末で切る (無ければ size で切る)
            cut = max(text.rfind(ch, start + size // 2, end) for ch in _SENTENCE_ENDS)
            if cut > start:
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def read_sections(path, extract_text):
    """ファイルを [(場所, 本文), ...] に分けて読む。場所は出典に使う ("p.3" / "シート: 集計" / "")。

    PDF はページごとに読み、ページ番号を出典に出せるようにする。
    それ以外は chat_app の extract_text (添付と同じ読み取り) を使う。
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("PDF を読むには pypdf が必要です (pip install pypdf)")
        reader = PdfReader(str(path))
        sections = [(f"p.{i}", page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
        sections = [(loc, text) for loc, text in sections if text.strip()]
        if not sections:
            raise RuntimeError("テキストを抽出できませんでした (画像だけの PDF などの可能性)")
        return sections
    text = extract_text(path)
    if path.suffix.lower() in (".xlsx", ".xlsm") and "# シート: " in text:
        sections = []
        for block in text.split("# シート: ")[1:]:
            name, _, body = block.partition("\n")
            sections.append((f"シート: {name.strip()}", body))
        return sections
    return [("", text)]


def iter_documents(docs_dir):
    """索引に入れるファイルを列挙する (サブフォルダも含む。隠しファイルと一時ファイルは除く)。"""
    docs_dir = Path(docs_dir)
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in INDEXABLE_SUFFIXES:
            continue
        if any(part.startswith(".") for part in path.relative_to(docs_dir).parts):
            continue
        if path.name.startswith("~$"):          # Office の一時ファイル
            continue
        yield path


# --------------------------------------------------------------------------
# 索引
# --------------------------------------------------------------------------
class RagIndex:
    """チャンクとベクトルを保存した索引。

    rag_index/
      meta.json     索引の情報 (資料フォルダ、モデル、ファイルごとの更新日時とサイズ)
      chunks.json   チャンク [{"file", "location", "text"}, ...]
      vectors.npy   チャンクのベクトル (正規化済み、chunks.json と同じ順)
    """

    def __init__(self, index_dir=None):
        self.dir = Path(index_dir or INDEX_DIR)
        self.meta = {}
        self.chunks = []
        self.vectors = None

    # ---- 読み書き ----------------------------------------------------------
    def load(self):
        """保存済みの索引を読む。無ければ False。"""
        try:
            import numpy as np

            self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
            self.chunks = json.loads((self.dir / "chunks.json").read_text(encoding="utf-8"))
            self.vectors = np.load(self.dir / "vectors.npy")
        except FileNotFoundError:
            self.meta, self.chunks, self.vectors = {}, [], None
            return False
        if self.meta.get("version") != INDEX_VERSION or len(self.chunks) != len(self.vectors):
            log.warning("[RAG] 索引の形式が古いか壊れているため、作り直しが必要です")
            self.meta, self.chunks, self.vectors = {}, [], None
            return False
        return True

    def save(self):
        """索引を書く。書きかけで落ちても前の索引が壊れないよう、一時ファイル経由で置き換える。"""
        import numpy as np

        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "vectors.npy.tmp"
        with open(tmp, "wb") as f:              # パスで渡すと np.save が .npy を付け足すため
            np.save(f, self.vectors)
        os.replace(tmp, self.dir / "vectors.npy")
        # meta.json は最後に書く (途中で落ちても件数の食い違いで検出できる)
        for name, data in (
            ("chunks.json", json.dumps(self.chunks, ensure_ascii=False)),
            ("meta.json", json.dumps(self.meta, ensure_ascii=False, indent=1)),
        ):
            tmp = self.dir / (name + ".tmp")
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, self.dir / name)

    @property
    def ready(self):
        return self.vectors is not None and len(self.chunks) > 0

    def summary(self):
        """画面表示用の要約。"""
        if not self.meta:
            return "索引はまだありません"
        files = self.meta.get("files", {})
        built = time.strftime("%Y/%m/%d %H:%M", time.localtime(self.meta.get("built", 0)))
        return (f"{len(files)} ファイル / {len(self.chunks)} チャンク "
                f"(モデル: {self.meta.get('model', '?')}, 更新: {built})")

    # ---- 作成・差分更新 --------------------------------------------------
    def build(self, docs_dir, embedder, extract_text, progress=None, cancel=None):
        """資料フォルダから索引を作る。前回から変わったファイルだけ読み直す。

        progress(処理済み数, 全体数, ファイル名) を呼ぶ。cancel (threading.Event) が
        立ったら途中でやめる (それまでの索引はそのまま残る)。
        戻り値: {"added", "updated", "removed", "unchanged", "failed": [(名前, 理由)], "cancelled"}
        """
        import numpy as np

        docs_dir = Path(docs_dir).resolve()
        if not docs_dir.is_dir():
            raise RuntimeError(f"資料フォルダが見つかりません: {docs_dir}")
        self.load()
        # 資料フォルダやモデルが変わったら、ベクトルを使い回せないので全部作り直す
        reusable = (
            self.ready
            and self.meta.get("docs_dir") == str(docs_dir)
            and self.meta.get("model") == embedder.name
        )
        old_files = self.meta.get("files", {}) if reusable else {}
        old_by_file = {}
        if reusable:
            for i, chunk in enumerate(self.chunks):
                old_by_file.setdefault(chunk["file"], []).append(i)

        paths = list(iter_documents(docs_dir))
        new_chunks, new_vectors, files = [], [], {}
        stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0, "failed": [],
                 "cancelled": False}
        for n, path in enumerate(paths, 1):
            if cancel is not None and cancel.is_set():
                stats["cancelled"] = True
                log.info("[RAG] 索引の作成を中止しました")
                return stats
            rel = path.relative_to(docs_dir).as_posix()
            if progress:
                progress(n, len(paths), rel)
            st = path.stat()
            signature = {"mtime": st.st_mtime_ns, "size": st.st_size}
            if old_files.get(rel, {}).get("signature") == signature and rel in old_by_file:
                ids = old_by_file[rel]
                new_chunks.extend(self.chunks[i] for i in ids)
                new_vectors.append(self.vectors[ids])
                files[rel] = old_files[rel]
                stats["unchanged"] += 1
                continue
            try:
                pieces = [
                    {"file": rel, "location": loc, "text": chunk}
                    for loc, text in read_sections(path, extract_text)
                    for chunk in split_text(text)
                ]
                if not pieces:
                    raise RuntimeError("本文がありません")
                vectors = embedder.encode_documents(p["text"] for p in pieces)
            except Exception as e:
                log.warning("[RAG] 読めないファイル: %s (%s)", rel, e)
                stats["failed"].append((rel, str(e)))
                continue
            new_chunks.extend(pieces)
            new_vectors.append(vectors)
            files[rel] = {"signature": signature, "chunks": len(pieces)}
            stats["updated" if rel in old_files else "added"] += 1
        stats["removed"] = len(set(old_files) - set(files))

        self.chunks = new_chunks
        self.vectors = (np.concatenate(new_vectors).astype("float32") if new_vectors
                        else np.zeros((0, 0), dtype="float32"))
        self.meta = {
            "version": INDEX_VERSION, "docs_dir": str(docs_dir), "model": embedder.name,
            "chunk_chars": CHUNK_CHARS, "built": time.time(), "files": files,
        }
        self.save()
        log.info("[RAG] 索引を保存: %s (%s)", self.dir, stats)
        return stats

    # ---- 検索 --------------------------------------------------------------
    def search(self, query, embedder, top_k=TOP_K, max_chars=MAX_CONTEXT_CHARS, min_score=None):
        """質問に近いチャンクを返す。合計が max_chars を超えない範囲で上位から取る。

        戻り値: [{"no", "file", "location", "text", "score", "path"}, ...]
        """
        import numpy as np

        if not self.ready or not (query or "").strip():
            return []
        if self.meta.get("model") != embedder.name:
            raise RuntimeError(
                f"索引は「{self.meta.get('model')}」で作られています。"
                f"「{embedder.name}」で使うには索引を作り直してください"
            )
        scores = self.vectors @ embedder.encode_query(query)
        order = np.argsort(-scores)
        hits, total, seen = [], 0, set()
        min_score = MIN_SCORE if min_score is None else min_score
        for i in order:
            if scores[int(i)] < min_score:
                break                           # 以降はもっと低い
            chunk = self.chunks[int(i)]
            key = (chunk["file"], chunk["text"])
            if key in seen:
                continue
            text = chunk["text"]
            if total + len(text) > max_chars:
                if hits:
                    break
                text = text[:max_chars]        # 1 件目だけは切り詰めてでも入れる
            seen.add(key)
            total += len(text)
            hits.append({
                "no": len(hits) + 1, "file": chunk["file"], "location": chunk["location"],
                "text": text, "score": float(scores[int(i)]),
                "path": str(Path(self.meta["docs_dir"]) / chunk["file"]),
            })
            if len(hits) >= top_k:
                break
        return hits


# --------------------------------------------------------------------------
# 回答への組み込み
# --------------------------------------------------------------------------
CONTEXT_INSTRUCTION = (
    "以下は資料から検索した抜粋です。回答はこの抜粋に基づいて行い、"
    "根拠にした抜粋の番号を [1] のように文中に示してください。"
    "抜粋に答えが書かれていない場合は、推測せずに「資料には見当たりません」と答えてください。"
)


def source_label(hit):
    """出典の表示 ("規程/就業規則.pdf p.3")。"""
    return f"{hit['file']} {hit['location']}".strip()


def format_context(hits, question):
    """検索結果と質問を、モデルに渡す 1 つの文にまとめる。"""
    blocks = [CONTEXT_INSTRUCTION]
    for hit in hits:
        blocks.append(f"[{hit['no']}] {source_label(hit)}\n{hit['text']}")
    blocks.append(f"質問: {question}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# コマンドライン (索引の作成と、検索品質の確認)
# --------------------------------------------------------------------------
def _settings():
    try:
        path = Path(__file__).resolve().parent / "chat_settings.json"
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main(argv=None):
    import argparse
    import threading

    parser = argparse.ArgumentParser(description="資料検索 (RAG) の索引の作成と検索")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="資料フォルダから索引を作る (差分更新)")
    b.add_argument("docs_dir", nargs="?", help="資料フォルダ (省略時は chat_settings.json の値)")
    b.add_argument("--model", help="埋め込みモデルのフォルダ (省略時は chat_settings.json の値)")
    s = sub.add_parser("search", help="検索して上位の抜粋を表示する")
    s.add_argument("query")
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--model")
    sub.add_parser("info", help="索引の情報を表示する")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    settings = _settings()
    index = RagIndex()

    if args.command == "info":
        index.load()
        print(index.summary())
        return 0

    model = args.model or settings.get("rag_model_dir")
    if not model:
        parser.error("埋め込みモデルを --model で指定してください")
    embedder = Embedder(model)

    if args.command == "build":
        docs_dir = args.docs_dir or settings.get("rag_docs_dir")
        if not docs_dir:
            parser.error("資料フォルダを指定してください")
        from chat_app import extract_document_text        # 添付と同じ読み取りを使う

        def extract(path):
            return extract_document_text(path, max_chars=None)

        def progress(n, total, name):
            print(f"\r[{n}/{total}] {name[:60]:<60}", end="", flush=True)

        t0 = time.time()
        stats = index.build(docs_dir, embedder, extract, progress=progress,
                            cancel=threading.Event())
        print()
        print(f"完了 ({time.time() - t0:.1f} 秒): 追加 {stats['added']} / 更新 {stats['updated']} / "
              f"削除 {stats['removed']} / 変更なし {stats['unchanged']} / 失敗 {len(stats['failed'])}")
        for name, reason in stats["failed"]:
            print(f"  読めなかったファイル: {name} ({reason})")
        print(index.summary())
        return 0

    if not index.load():
        print("索引がありません。先に build を実行してください")
        return 1
    t0 = time.time()
    hits = index.search(args.query, embedder, top_k=args.top, max_chars=10 ** 9)
    print(f"検索 {time.time() - t0:.2f} 秒")
    for hit in hits:
        print(f"\n[{hit['no']}] {hit['score']:.3f}  {source_label(hit)}")
        print("    " + hit["text"].replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
