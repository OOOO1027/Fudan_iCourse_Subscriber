"""Encrypted storage of board pages and their readings.

The store is a directory that scripts/board_branch.sh mirrors to boards/ on the
`boards` branch:

    index.enc    JSON, one entry per lecture: state and the reading of each board
    <name>.enc   one lecture's page images with their positions and times
    <name>.summary.enc   the summary a lecture had before it was rewritten
                         with boards (JSON), kept so that it can be put back
    *.rev        the same two, encrypted to scripts/board_review_key.pub

.enc files use the database's password (src/data/crypto_box.py), so whoever can
open the database can open the boards.  <name> is a keyed hash of the lecture
id: the branch is public and shows only how many lectures have boards.

The directory must hold the `.ready` marker that board_branch.sh writes after
it has fetched the branch.  Without it the index on the branch is unknown, and
writing a new one would forget every lecture already stored.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes

from src.data import crypto_box

MAGIC = b"ICB1"
READY = ".ready"


class StoreUnavailable(RuntimeError):
    pass


def pack(manifest: dict, images: list[bytes]) -> bytes:
    """MAGIC, 4-byte length, manifest JSON, then the images end to end.  The
    manifest's `images` lists each image's length in order."""
    head = json.dumps({**manifest, "images": [len(i) for i in images]},
                      ensure_ascii=False, separators=(",", ":")).encode()
    return MAGIC + len(head).to_bytes(4, "big") + head + b"".join(images)


def unpack(blob: bytes) -> tuple[dict, list[bytes]]:
    if blob[:4] != MAGIC:
        raise ValueError("not a board bundle")
    size = int.from_bytes(blob[4:8], "big")
    manifest = json.loads(blob[8:8 + size])
    images, at = [], 8 + size
    for length in manifest["images"]:
        images.append(blob[at:at + length])
        at += length
    return manifest, images


def seal_for_review(data: bytes, public_key: bytes) -> bytes:
    """RSA-OAEP(SHA-256) wrapped AES-256-GCM: b"BRD1", 2-byte length, wrapped
    key, 12-byte nonce, 16-byte tag, ciphertext."""
    session = get_random_bytes(32)
    wrapped = PKCS1_OAEP.new(RSA.import_key(public_key), hashAlgo=SHA256).encrypt(session)
    cipher = AES.new(session, AES.MODE_GCM, nonce=get_random_bytes(12))
    body, tag = cipher.encrypt_and_digest(data)
    return b"BRD1" + len(wrapped).to_bytes(2, "big") + wrapped + cipher.nonce + tag + body


class BoardStore:
    def __init__(self, directory, password: str, review_key: bytes | None = None):
        self.dir = Path(directory)
        self.password = password
        self.review_key = review_key
        self.dirty = False
        if not (self.dir / READY).exists():
            raise StoreUnavailable(f"{self.dir} was not fetched from the boards branch")
        index = self.dir / "index.enc"
        self.lectures: dict[str, dict] = {}
        if index.exists():
            plain = crypto_box.decrypt(index.read_bytes(), password)
            if not crypto_box.is_json_obj(plain):
                raise StoreUnavailable("board index did not decrypt to JSON")
            self.lectures = json.loads(plain)["lectures"]

    def get(self, sub_id) -> dict | None:
        return self.lectures.get(str(sub_id))

    def set(self, sub_id, entry: dict) -> None:
        self.lectures[str(sub_id)] = entry
        self.dirty = True

    def name(self, sub_id) -> str:
        return hashlib.sha256(f"ICS-board:{self.password}:{sub_id}".encode()).hexdigest()[:32]

    def _write(self, stem: str, data: bytes) -> None:
        (self.dir / f"{stem}.enc").write_bytes(crypto_box.encrypt(data, self.password))
        if self.review_key:
            (self.dir / f"{stem}.rev").write_bytes(seal_for_review(data, self.review_key))

    def put_pages(self, sub_id, manifest: dict, images: list[bytes]) -> str:
        name = self.name(sub_id)
        self._write(name, pack(manifest, images))
        return name

    def put_summary(self, sub_id, summary: str, model: str | None) -> None:
        self._write(f"{self.name(sub_id)}.summary", json.dumps(
            {"sub_id": str(sub_id), "summary": summary, "summary_model": model},
            ensure_ascii=False).encode())

    def _read(self, stem: str) -> bytes | None:
        """A stored file, decrypted: from this directory if it was written in
        this run, else from the branch through board_branch.sh's scratch
        repository, which fetches the one file on demand."""
        path = self.dir / f"{stem}.enc"
        try:
            if path.exists():
                blob = path.read_bytes()
            else:
                repo = os.environ.get("BOARD_GIT_DIR") or os.path.join(
                    os.environ.get("RUNNER_TEMP", "/tmp"), "boards.git")
                blob = subprocess.run(
                    ["git", "--git-dir", repo, "show", f"boards:boards/{stem}.enc"],
                    capture_output=True, timeout=300, check=True).stdout
            return crypto_box.decrypt(blob, self.password)
        except Exception:
            return None

    def load_pages(self, sub_id) -> tuple[dict, list[bytes]] | None:
        plain = self._read(self.name(sub_id))
        try:
            return unpack(plain) if plain else None
        except Exception:
            return None

    def load_summary(self, sub_id) -> str | None:
        """The summary the lecture had before it was rewritten with boards."""
        plain = self._read(f"{self.name(sub_id)}.summary")
        try:
            return json.loads(plain)["summary"] if plain else None
        except Exception:
            return None

    def save(self) -> None:
        if not self.dirty:
            return
        self._write("index", json.dumps({"version": 1, "lectures": self.lectures},
                                        ensure_ascii=False, separators=(",", ":")).encode())
        self.dirty = False
