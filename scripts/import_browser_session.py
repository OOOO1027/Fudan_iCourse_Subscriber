"""Receive encrypted session data without persisting or logging plaintext cookies."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api.session_import import SessionImportRelay, verify_and_save


def main():
    root = Path(os.environ['RUNNER_TEMP']) / 'icourse-import'
    root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def publish(challenge):
        # This artifact contains only a public key and a one-use target identifier.
        (root / 'connection.json').write_text(json.dumps(challenge))
        print('[SessionImport] Waiting for owner-approved encrypted session.', flush=True)

    relay = SessionImportRelay(publish)
    try:
        message = relay.receive('session', timeout=1200)
        verify_and_save(message['cookies'])
        message.clear()
    finally:
        relay.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Exception bodies can include server redirects or credential-bearing URLs.
        print('[SessionImport] Failed safely (' + type(error).__name__ + '); no session cached.', flush=True)
        sys.exit(1)
