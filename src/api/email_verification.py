"""Complete the school's official email second factor in the existing session."""
from src.api.email_relay import EmailRelay
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
        relay = self.relay_factory(self.publish)
        try:
            print('[Email] Public encryption challenge ready; waiting for owner to request email.', flush=True)
            relay.receive('send', timeout=900)
            reply = session.post(url.rsplit('/', 1)[0] + '/sendMailMsg',
                                 json={'loginName': challenge['userName'], 'pointJson': None,
                                       'token': None, 'lck': previous['lck']},
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
