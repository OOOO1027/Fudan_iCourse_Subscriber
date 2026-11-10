import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests
from src.api import session_cache as cache
from src.api.webvpn import WebVPNSession


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'auth' / 'session.enc'
        self.env = patch.dict(os.environ, {'RUNNER_TEMP': self.temp.name,
                              'ICOURSE_SESSION_FILE': str(self.path)})
        self.env.start()
        self.credentials = patch.multiple(cache.config, STUDENT_ID='test-user', PASSWORD='test-password')
        self.credentials.start()

    def tearDown(self):
        self.credentials.stop()
        self.env.stop()
        self.temp.cleanup()

    def save(self):
        vpn = WebVPNSession()
        vpn.session.cookies.set('session', 'test-private-cookie', domain='webvpn.fudan.edu.cn', secure=True)
        vpn.session.cookies.set('ignored', 'other-cookie', domain='example.test')
        cache.save_session(vpn)
        return vpn

    def test_cache_is_encrypted_and_revalidated(self):
        self.save()
        self.assertNotIn(b'test-private-cookie', self.path.read_bytes())
        self.assertTrue(self.path.with_suffix('.verified').is_file())
        fresh = WebVPNSession()
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"code":0}'
        fresh.session.get = Mock(return_value=response)
        with patch.object(cache, 'WebVPNSession', return_value=fresh):
            self.assertIs(cache.load_verified_session(), fresh)
        self.assertEqual(len(fresh.session.cookies), 1)
        self.assertEqual(next(iter(fresh.session.cookies)).value, 'test-private-cookie')
        fresh.session.get.assert_called_once()

    def test_expired_authorization_is_not_reused(self):
        self.save()
        fresh = WebVPNSession()
        response = requests.Response()
        response.status_code = 401
        fresh.session.get = Mock(return_value=response)
        with patch.object(cache, 'WebVPNSession', return_value=fresh):
            self.assertIsNone(cache.load_verified_session())

    def test_wrong_key_or_tampering_cannot_restore(self):
        self.save()
        with patch.object(cache.config, 'PASSWORD', 'wrong'):
            self.assertIsNone(cache.load_verified_session())

    def test_cache_cannot_be_written_in_checkout(self):
        with patch.dict(os.environ, {'ICOURSE_SESSION_FILE': '/outside-runner/session.enc'}):
            with self.assertRaises(ValueError):
                cache.save_session(Mock())

    def test_transport_failure_is_not_treated_as_expiration(self):
        self.save()
        fresh = WebVPNSession()
        fresh.session.get = Mock(side_effect=requests.exceptions.ReadTimeout())
        with patch.object(cache, 'WebVPNSession', return_value=fresh):
            with self.assertRaises(requests.exceptions.ReadTimeout):
                cache.load_verified_session()


if __name__ == '__main__':
    unittest.main()
