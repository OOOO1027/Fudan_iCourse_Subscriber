import base64
import unittest
from unittest.mock import Mock

from src.api.qr_verification import QRVerification, seal, unseal
from src.api.webvpn import AuthenticationActionRequired, WebVPNSession


class QRTests(unittest.TestCase):
    def test_encryption_round_trip_and_tamper_rejection(self):
        blob = seal(b'private-image', 'test-user', 'test-password', 'qr-image')
        self.assertNotIn(b'private-image', blob)
        self.assertEqual(unseal(blob, 'test-user', 'test-password', 'qr-image'), b'private-image')
        for value, password, purpose in (
            (blob, 'wrong', 'qr-image'), (blob, 'test-password', 'session'),
            (blob[:-1] + bytes([blob[-1] ^ 1]), 'test-password', 'qr-image'),
        ):
            with self.assertRaises(ValueError):
                unseal(value, 'test-user', password, purpose)

    def test_official_second_factor_context_is_preserved(self):
        qr, complete = Mock(), Mock()
        qr.json.return_value = {'code': 200, 'data': {
            'qrToken': 'qr-test', 'qrMsg': 'data:image/png;base64,' + base64.b64encode(b'image').decode()}}
        complete.json.return_value = {'code': 200, 'loginToken': 'completed'}
        session, publish = Mock(), Mock()
        session.post.side_effect = [qr, complete]
        previous = {'entityId': 'https://icourse.fudan.edu.cn', 'lck': 'context', 'requestType': 'chain_type'}
        challenge = {'moduleCodes': ['qr'], 'authChainCode': 'next-chain',
                     'requestNumber': 'next-request', 'userName': 'test-user'}
        result = QRVerification(publish)(session, 'https://id.fudan.edu.cn/idp/authn/authExecute', {}, previous, challenge)
        publish.assert_called_once_with(b'image')
        body = session.post.call_args_list[1].kwargs['json']
        self.assertEqual(body['authModuleCode'], 'qr')
        self.assertEqual(body['requestNumber'], 'next-request')
        self.assertEqual(body['lck'], 'context')
        self.assertEqual(body['authPara'], {'loginName': 'test-user', 'qrToken': 'qr-test'})
        self.assertEqual(result['loginToken'], 'completed')

    def test_cannot_use_qr_when_not_offered(self):
        session = Mock()
        with self.assertRaises(AuthenticationActionRequired):
            QRVerification(Mock())(session, '', {}, {}, {'moduleCodes': ['userAndOtp']})
        session.post.assert_not_called()

    def test_interactive_hook_cannot_accept_another_incomplete_step(self):
        vpn = WebVPNSession(verification_handler=Mock(return_value={'code': 200, 'second': True}))
        with self.assertRaises(AuthenticationActionRequired):
            vpn._complete_auth({'second': True}, {}, '', {})


if __name__ == '__main__':
    unittest.main()
