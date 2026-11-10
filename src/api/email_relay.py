"""One-use encrypted messages for a live, owner-initiated MFA session."""
import base64
import io
import json
import os
import re
import secrets
import time
import zipfile

import requests
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA

from src.api.webvpn import AuthenticationActionRequired


class EmailRelay:
    def __init__(self, publish):
        self.key = RSA.generate(2048)
        self.target = '{}-{}-{}'.format(os.environ['GITHUB_RUN_ID'],
                                       os.environ['GITHUB_RUN_ATTEMPT'], secrets.token_hex(12))
        self.created = time.time()
        self.seen = set()
        self.api = 'https://api.github.com/repos/' + os.environ['GITHUB_REPOSITORY']
        self.owner = os.environ['GITHUB_REPOSITORY'].split('/')[0].lower()
        self.http = requests.Session()
        self.http.headers.update({'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
                                  'Accept': 'application/vnd.github+json'})
        self.challenge = {'target': self.target, 'expires': int(self.created + 1800),
                          'public_key': self.key.public_key().export_key().decode()}
        publish(self.challenge)

    def decode(self, ciphertext):
        try:
            if not isinstance(ciphertext, str) or len(ciphertext) > 512:
                raise ValueError()
            raw = base64.b64decode(ciphertext, validate=True)
            clear = PKCS1_OAEP.new(self.key, hashAlgo=SHA256,
                                  label=self.target.encode()).decrypt(raw)
            value = json.loads(clear)
            if not isinstance(value, dict) or value.get('target') != self.target:
                raise ValueError()
            if value.get('action') not in ('send', 'code'):
                raise ValueError()
            if value['action'] == 'code' and not re.fullmatch(r'[0-9]{4,10}', value.get('code', '')):
                raise ValueError()
            return value
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise ValueError('Invalid encrypted verification message') from None

    def _json(self, path):
        reply = self.http.get(self.api + path, timeout=(10, 25))
        reply.raise_for_status()
        return reply.json()

    def _trusted(self, run):
        return (run.get('path') == '.github/workflows/email_relay.yml'
                and run.get('event') == 'workflow_dispatch'
                and run.get('head_branch') == 'main'
                and (run.get('actor') or {}).get('login', '').lower() == self.owner)

    def receive(self, action, timeout=600):
        deadline = min(time.monotonic() + timeout,
                       time.monotonic() + max(0, self.created + 1800 - time.time()))
        while time.monotonic() < deadline:
            try:
                artifacts = self._json('/actions/artifacts?per_page=100').get('artifacts', [])
                for artifact in artifacts:
                    identifier = artifact['id']
                    if identifier in self.seen or not artifact['name'].startswith('email-response-' + self.target + '-'):
                        continue
                    if artifact.get('expired') or artifact.get('size_in_bytes', 0) > 16384:
                        self.seen.add(identifier)
                        continue
                    run_id = artifact.get('workflow_run', {}).get('id')
                    if not run_id or not self._trusted(self._json('/actions/runs/' + str(run_id))):
                        self.seen.add(identifier)
                        continue
                    response = self.http.get(self.api + '/actions/artifacts/' + str(identifier) + '/zip',
                                             timeout=(10, 25))
                    response.raise_for_status()
                    if len(response.content) > 16384:
                        self.seen.add(identifier)
                        continue
                    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                        entries = archive.infolist()
                        if len(entries) != 1 or entries[0].filename != 'response.enc' or entries[0].file_size > 512:
                            self.seen.add(identifier)
                            continue
                        message = self.decode(archive.read(entries[0]).decode('ascii').strip())
                    self.seen.add(identifier)
                    if message['action'] == action:
                        return message
            except (requests.RequestException, ValueError, KeyError, zipfile.BadZipFile):
                # Neither API errors nor rejected ciphertext may disclose verification data.
                pass
            time.sleep(4)
        raise AuthenticationActionRequired('Email verification input timed out; start a new recovery')

    def close(self):
        self.http.close()
        self.key = None
