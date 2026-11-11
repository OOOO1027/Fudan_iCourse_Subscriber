import base64
import unittest
from unittest.mock import Mock, patch

import requests
from src.api.webvpn import WebVPNSession, AuthenticationActionRequired, totp_code


class TotpTests(unittest.TestCase):
    # Public RFC 6238 SHA1 test key, never an account credential.
    secret = base64.b32encode(b'12345678901234567890').decode()

    def setUp(self):
        self.vpn = WebVPNSession()
        self.addCleanup(self.vpn.session.close)
        self.challenge = dict(code=200, second=True, moduleCodes=['userAndOtp'],
                              requestNumber='challenge-number', authChainCode='otp-chain', userName='test-user')
        self.payload = dict(entityId='https://icourse.fudan.edu.cn', requestType='chain_type',
                            lck='test-context', authPara={'loginName': 'test-user'})

    def complete(self, result, url='https://id.fudan.edu.cn/idp/authn/authExecute'):
        self.vpn.session.post = Mock(return_value=Mock(json=Mock(return_value=result)))
        with patch('src.api.webvpn.config.TOTP_SECRET', self.secret), patch('src.api.webvpn.time.time', return_value=59):
            return self.vpn._complete_auth(self.challenge, self.payload, url, {'Origin': 'test-origin'})

    def test_rfc_vectors(self):
        for timestamp, expected in [(59, '287082'), (1111111109, '081804'),
                                    (1111111111, '050471'), (1234567890, '005924'),
                                    (2000000000, '279037'), (20000000000, '353130')]:
            self.assertEqual(totp_code(self.secret, timestamp), expected)

    def test_normalize_binding_key(self):
        self.assertEqual(totp_code(' ' + self.secret.lower() + '\n', 59), '287082')

    def test_invalid_secret_is_sanitized(self):
        for secret in ('', 'not-a-binding-secret!'):
            with self.assertRaises(AuthenticationActionRequired) as error:
                totp_code(secret, 59)
            if secret:
                self.assertNotIn(secret, str(error.exception))

    def test_direct_and_proxied_context_preserved(self):
        for url in ('https://id.fudan.edu.cn/idp/authn/authExecute',
                    'https://webvpn.fudan.edu.cn/https/proxied-idp/idp/authn/authExecute'):
            self.assertEqual(self.complete({'code': 200, 'loginToken': 'test-token'}, url), 'test-token')
            args, kwargs = self.vpn.session.post.call_args
            self.assertEqual(args, (url,))
            body = kwargs['json']
            self.assertEqual(body['authModuleCode'], 'userAndOtp')
            self.assertEqual(body['requestNumber'], 'challenge-number')
            self.assertEqual(body['authChainCode'], 'otp-chain')
            self.assertEqual(body['lck'], 'test-context')
            self.assertEqual(body['authPara']['otpCode'], '287082')
            self.assertNotIn('password', body['authPara'])

    def test_incomplete_and_rejected_result_fail_closed(self):
        for value in ({'code': 200, 'second': True, 'loginToken': 'test-token'},
                      {'code': 200}, {'code': 400, 'message': 'private-response'}, []):
            with self.assertRaises(AuthenticationActionRequired) as error:
                self.complete(value)
            self.assertNotIn('private-response', str(error.exception))

    def test_unsupported_or_missing_context_does_not_post(self):
        baseline = self.challenge.copy()
        for change in ({'moduleCodes': ['userAndMail']}, {'requestNumber': ''}, {'authChainCode': ''}):
            self.challenge = dict(baseline, **change)
            with self.assertRaises(AuthenticationActionRequired):
                self.complete({'code': 200, 'loginToken': 'test-token'})
            self.vpn.session.post.assert_not_called()

    def test_timeout_does_not_retry_or_disclose(self):
        self.vpn.session.post = Mock(side_effect=requests.Timeout('private-response'))
        with patch('src.api.webvpn.config.TOTP_SECRET', self.secret):
            with self.assertRaises(AuthenticationActionRequired) as error:
                self.vpn._complete_auth(self.challenge, self.payload, 'https://id.fudan.edu.cn/idp/authn/authExecute', {})
        self.vpn.session.post.assert_called_once()
        self.assertNotIn('private-response', str(error.exception))

    def test_no_secret_requires_configuration(self):
        with patch('src.api.webvpn.config.TOTP_SECRET', ''):
            with self.assertRaisesRegex(AuthenticationActionRequired, 'UIS_TOTP_SECRET'):
                self.vpn._complete_auth(self.challenge, self.payload, 'unused', {})


if __name__ == '__main__':
    unittest.main()
