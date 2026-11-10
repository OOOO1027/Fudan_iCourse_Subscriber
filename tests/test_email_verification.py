import base64
import copy
import json
import os
import unittest
from unittest.mock import Mock, patch

from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA

from src.api.email_relay import EmailRelay
from src.api.email_verification import EmailVerification
from src.api.webvpn import AuthenticationActionRequired


class EmailTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1',
                                          'GITHUB_REPOSITORY': 'owner/repo', 'GH_TOKEN': 'test-only'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def make_relay(self):
        relay = EmailRelay(Mock())
        self.addCleanup(relay.close)
        return relay

    def encode(self, relay, message, label=None):
        cipher = PKCS1_OAEP.new(relay.key.public_key(), hashAlgo=SHA256,
                               label=(label or relay.target).encode())
        return base64.b64encode(cipher.encrypt(json.dumps(message).encode())).decode()

    def test_roundtrip_and_public_challenge(self):
        relay = self.make_relay()
        message = {'target': relay.target, 'action': 'code', 'code': '123456'}
        self.assertEqual(relay.decode(self.encode(relay, message)), message)
        self.assertNotIn('PRIVATE', json.dumps(relay.challenge))
        self.assertEqual(set(relay.challenge), {'target', 'expires', 'public_key'})

    def test_reject_wrong_target_label_and_plaintext(self):
        relay = self.make_relay()
        for message in ({'target': 'wrong', 'action': 'send'},
                        {'target': relay.target, 'action': 'delete'},
                        {'target': relay.target, 'action': 'code', 'code': '<bad>'}):
            with self.assertRaisesRegex(ValueError, '^Invalid encrypted verification message$'):
                relay.decode(self.encode(relay, message))
        for value in ('123456', self.encode(relay, {'target': relay.target, 'action': 'send'}, 'wrong')):
            with self.assertRaises(ValueError):
                relay.decode(value)

    def test_relay_provenance(self):
        relay = self.make_relay()
        run = {'path': '.github/workflows/email_relay.yml', 'event': 'workflow_dispatch',
               'head_branch': 'main', 'actor': {'login': 'OWNER'}}
        self.assertTrue(relay._trusted(run))
        for key, value in [('path', 'other.yml'), ('event', 'pull_request'),
                           ('head_branch', 'other'), ('actor', {'login': 'other'})]:
            self.assertFalse(relay._trusted(dict(run, **{key: value})))

    def perform(self, final=None, delivery=None, modules=None):
        session = Mock()
        relay = Mock()
        relay.receive.side_effect = [{'action': 'send', 'point_x': 100}, {'action': 'code', 'code': '123456'}]
        session.get.return_value.json.return_value = {'code': '200', 'data': {
            'originalImageBase64': 'test-image', 'jigsawImageBase64': 'test-piece', 'token': 'test-puzzle'}}
        sent = []
        results = [{'code': '200'} if delivery is None else delivery,
                   {'code': '200', 'loginToken': 'test-token'} if final is None else final]

        def post(url, **kwargs):
            sent.append((url, copy.deepcopy(kwargs)))
            response = Mock()
            response.json.return_value = results.pop(0)
            return response

        session.post.side_effect = post
        handler = EmailVerification(Mock(), lambda publish: relay)
        previous = {'entityId': 'service', 'requestType': 'cas', 'lck': 'context'}
        challenge = {'moduleCodes': ['userAndMail'] if modules is None else modules,
                     'authChainCode': 'chain2', 'requestNumber': 'request2', 'userName': 'test-user'}
        return handler, session, relay, sent, previous, challenge

    def test_official_delivery_and_completion(self):
        handler, session, relay, sent, previous, challenge = self.perform()
        result = handler(session, 'https://id.fudan.edu.cn/idp/authn/authExecute', {}, previous, challenge)
        self.assertEqual(result['loginToken'], 'test-token')
        self.assertTrue(sent[0][0].endswith('/authn/sendMailMsg'))
        self.assertEqual(sent[0][1]['json']['lck'], 'context')
        self.assertEqual(sent[0][1]['json']['token'], 'test-puzzle')
        self.assertEqual(json.loads(sent[0][1]['json']['pointJson']), {'x': 100, 'y': 5})
        self.assertEqual(sent[1][1]['json']['authPara']['mailCode'], '123456')
        self.assertEqual(sent[1][1]['json']['authModuleCode'], 'userAndMail')
        self.assertEqual([c.args[0] for c in relay.receive.call_args_list], ['send', 'code'])
        relay.close.assert_called_once()

    def test_fail_closed_on_missing_token_or_second_factor(self):
        for final in ({'code': '200'}, {'code': '200', 'second': True}, {'code': '4005'}):
            handler, session, relay, sent, previous, challenge = self.perform(final=final)
            with self.assertRaises(RuntimeError):
                handler(session, 'https://school/authn/authExecute', {}, previous, challenge)
            relay.close.assert_called_once()

    def test_delivery_rejection_stops_without_submitting(self):
        handler, session, relay, sent, previous, challenge = self.perform(delivery={'code': '4005'})
        with self.assertRaises(AuthenticationActionRequired):
            handler(session, 'https://school/authn/authExecute', {}, previous, challenge)
        self.assertEqual(len(sent), 1)
        self.assertEqual(relay.receive.call_count, 1)

    def test_email_not_offered(self):
        handler, session, relay, sent, previous, challenge = self.perform(modules=['qr'])
        with self.assertRaises(AuthenticationActionRequired):
            handler(session, 'https://school/authn/authExecute', {}, previous, challenge)
        session.post.assert_not_called()

    def test_puzzle_must_be_completed_before_email(self):
        handler, session, relay, sent, previous, challenge = self.perform()
        relay.receive.side_effect = [{'action': 'send'}]
        with self.assertRaises(AuthenticationActionRequired):
            handler(session, 'https://school/authn/authExecute', {}, previous, challenge)
        session.post.assert_not_called()

    def test_puzzle_token_not_published(self):
        handler, session, relay, sent, previous, challenge = self.perform()
        public = Mock()
        handler.publish = public
        def factory(publish):
            publish({'target': 'test'})
            return relay
        handler.relay_factory = factory
        key = RSA.generate(2048)
        os.environ['ICOURSE_VIEWER_PUBLIC_KEY'] = base64.b64encode(key.public_key().export_key(format='DER')).decode()
        handler(session, 'https://school/authn/authExecute', {}, previous, challenge)
        delivered = public.call_args.args[0]
        self.assertNotIn('puzzle', delivered)
        encrypted = {k: base64.b64decode(v) for k, v in delivered['puzzle_encrypted'].items()}
        secret = PKCS1_OAEP.new(key, hashAlgo=SHA256, label=b'test').decrypt(encrypted['key'])
        cipher = AES.new(secret, AES.MODE_GCM, nonce=encrypted['nonce'])
        cipher.update(b'test')
        images = json.loads(cipher.decrypt_and_verify(encrypted['ciphertext'], encrypted['tag']))
        self.assertNotIn('token', images)
        self.assertEqual(images['originalImageBase64'], 'test-image')


if __name__ == '__main__':
    unittest.main()
