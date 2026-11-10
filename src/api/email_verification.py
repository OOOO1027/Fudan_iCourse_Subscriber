"""Complete the school's official email second factor in the existing session."""
import os
from src.api.email_relay import EmailRelay, encrypt_for_viewer
from src.api.webvpn import AuthenticationActionRequired, require_login_token


def response_data(response, context):
    response.raise_for_status()
    try:
        result = response.json()
    except ValueError:
        raise AuthenticationActionRequired(context + ': unexpected response') from None
    if not isinstance(result, dict) or str(result.get('code')) != '200':
        code = str(result.get('code', 'unknown')) if isinstance(result, dict) else 'unknown'
        code = code if code.isdigit() and len(code) < 8 else 'unknown'
        raise AuthenticationActionRequired(context + ': rejected (code=' + code + ')')
    return result


class EmailVerification:
    def __init__(self, publish, relay_factory=EmailRelay):
        self.publish = publish
        self.relay_factory = relay_factory

    def __call__(self, session, url, headers, previous, challenge):
        if 'userAndMail' not in (challenge.get('moduleCodes') or []):
            raise AuthenticationActionRequired('Email verification is not offered by the school')
        # Display the official puzzle, but keep its token in this session only.
        puzzle_reply = session.get(url.rsplit('/authn/', 1)[0] + '/captcha/getBlockPuzzle',
                                   headers=headers, timeout=(15, 60))
        puzzle = response_data(puzzle_reply, 'Official email puzzle').get('data', {})
        if not isinstance(puzzle, dict) or not all(isinstance(puzzle.get(k), str) and puzzle[k]
                                                 for k in ('originalImageBase64', 'jigsawImageBase64', 'token')):
            raise AuthenticationActionRequired('Official email puzzle has an unsupported format')
        images = {k: puzzle[k] for k in ('originalImageBase64', 'jigsawImageBase64')}
        if any(len(value) > 1_000_000 for value in images.values()):
            raise AuthenticationActionRequired('Official email puzzle exceeded the image limit')
        def publish_encrypted(info):
            encrypted = encrypt_for_viewer(os.environ['ICOURSE_VIEWER_PUBLIC_KEY'], images, info['target'])
            self.publish(dict(info, puzzle_encrypted=encrypted))
        relay = self.relay_factory(publish_encrypted)
        try:
            print('[Email] Public encryption challenge ready; waiting for owner to request email.', flush=True)
            request = relay.receive('send', timeout=900)
            point_x = request.get('point_x')
            if type(point_x) is not int or not 0 <= point_x <= 4096:
                raise AuthenticationActionRequired('Complete the official puzzle before requesting email')
            reply = session.post(url.rsplit('/', 1)[0] + '/sendMailMsg',
                                 json={'loginName': challenge['userName'],
                                       'pointJson': '{"x":' + str(point_x) + ',"y":5}',
                                       'token': puzzle.pop('token'), 'lck': previous['lck']},
                                 headers=headers, timeout=(15, 60))
            response_data(reply, 'Official email delivery')
            print('[Email] School accepted email delivery; waiting for encrypted code.', flush=True)
            message = relay.receive('code', timeout=600)
            payload = {
                'authModuleCode': 'userAndMail', 'authChainCode': challenge['authChainCode'],
                'entityId': previous['entityId'], 'requestNumber': challenge['requestNumber'],
                'requestType': previous['requestType'], 'lck': previous['lck'],
                'authPara': {'loginName': challenge['userName'], 'mailCode': message.pop('code')},
            }
            try:
                reply = session.post(url, json=payload, headers=headers, timeout=(15, 60))
            finally:
                payload['authPara']['mailCode'] = ''
            result = response_data(reply, 'Official email verification')
            require_login_token(result, 'Official email verification')
            print('[Email] Official second factor completed.', flush=True)
            return result
        finally:
            relay.close()
