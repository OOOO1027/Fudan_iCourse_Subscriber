import base64
import json
import time
import unittest
from unittest.mock import patch, Mock

from Crypto.PublicKey import RSA
from src.api.email_relay import encrypt_for_viewer
from src.api.session_import import SessionImportRelay, normalize_cookies, verify_and_save


class SessionImportTests(unittest.TestCase):
    def setUp(self):
        self.cookie = dict(name='test', value='not-a-real-cookie',
                           domain='webvpn.fudan.edu.cn', path='/', secure=True, expires=-1)
        self.relay = SessionImportRelay.__new__(SessionImportRelay)
        self.relay.key = RSA.generate(2048)
        self.relay.target = 'test-target'

    def envelope(self, **changes):
        value = dict(target=self.relay.target, action='session', created_at=time.time(), cookies=[self.cookie])
        value.update(changes)
        public = base64.b64encode(self.relay.key.public_key().export_key(format='DER')).decode()
        return json.dumps(encrypt_for_viewer(public, value, self.relay.target))

    def test_encrypted_round_trip(self):
        decoded = self.relay.decode(self.envelope())
        self.assertEqual(decoded['cookies'][0]['value'], self.cookie['value'])
        self.assertIsNone(decoded['cookies'][0]['expires'])

    def test_tampered_ciphertext(self):
        value = json.loads(self.envelope())
        value['tag'] = base64.b64encode(b'x' * 16).decode()
        with self.assertRaises(ValueError):
            self.relay.decode(json.dumps(value))

    def test_stale_or_wrong_target(self):
        for changes in ({'created_at': time.time() - 601}, {'target': 'other'}, {'action': 'code'}):
            with self.assertRaises(ValueError):
                self.relay.decode(self.envelope(**changes))

    def test_scope_is_closed(self):
        for domain in ('github.com', 'evilwebvpn.fudan.edu.cn', '.fudan.edu.cn'):
            with self.assertRaises(ValueError):
                normalize_cookies([{**self.cookie, 'domain': domain}])

    @patch('src.api.session_import.save_session')
    @patch('src.api.session_import.WebVPNSession')
    def test_rejected_api_never_cached(self, factory, save):
        factory.return_value._validate_icourse_response.side_effect = RuntimeError('unauthenticated')
        with self.assertRaises(RuntimeError):
            verify_and_save([self.cookie])
        save.assert_not_called()
        factory.return_value.session.close.assert_called_once()

    @patch('src.api.session_import.save_session', return_value=True)
    @patch('src.api.session_import.WebVPNSession')
    def test_verified_api_cached(self, factory, save):
        verify_and_save([self.cookie])
        save.assert_called_once_with(factory.return_value)
        self.assertTrue(factory.return_value.logged_in)

    def test_trusted_workflow(self):
        self.relay.owner = 'owner'
        good = dict(path=self.relay.workflow_path, event='workflow_dispatch', head_branch='main',
                    actor={'login': 'owner'})
        self.assertTrue(self.relay._trusted(good))
        self.assertFalse(self.relay._trusted({**good, 'path': '.github/workflows/email_relay.yml'}))


if __name__ == '__main__':
    unittest.main()
