import ast
from pathlib import Path
import unittest
from unittest.mock import Mock

import requests

from src.api.webvpn import WebVPNSession, AuthenticationActionRequired, require_login_token


def response(url, status=200):
    result = requests.Response()
    result.status_code = status
    result.url = url
    return result


class WebVPNSessionTests(unittest.TestCase):
    def login_function(self, factory, clock):
        # Isolate the real orchestration function from ASR/model dependencies.
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'login_with_retry')
        namespace = {'WebVPNSession': factory, 'time': clock,
                     'AuthenticationActionRequired': AuthenticationActionRequired}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'main.py', 'exec'), namespace)
        return namespace['login_with_retry']

    def test_interactive_challenge_is_not_retried(self):
        vpn, clock = Mock(), Mock()
        vpn.authenticate_icourse.side_effect = AuthenticationActionRequired('verification needed')
        factory = Mock(return_value=vpn)
        with self.assertRaises(AuthenticationActionRequired):
            self.login_function(factory, clock)()
        self.assertEqual(factory.call_count, 1)
        clock.sleep.assert_not_called()
        vpn.session.close.assert_called_once()

    def test_timeout_still_retries_and_closes_failed_session(self):
        first, second, clock = Mock(), Mock(), Mock()
        first.login.side_effect = requests.exceptions.ReadTimeout()
        factory = Mock(side_effect=[first, second])
        self.assertIs(self.login_function(factory, clock)(max_attempts=2), second)
        first.session.close.assert_called_once()
        second.authenticate_icourse.assert_called_once()
        clock.sleep.assert_called_once_with(30)

    def test_second_step_is_not_success_even_with_code_200(self):
        with self.assertRaisesRegex(AuthenticationActionRequired, "additional verification"):
            require_login_token({'code': 200, 'second': True, 'loginToken': None,
                                 'moduleCodes': ['testModule']}, 'iCourse CAS')

    def test_second_step_cannot_be_overridden_by_token(self):
        with self.assertRaises(AuthenticationActionRequired):
            require_login_token({'code': 200, 'second': True, 'loginToken': 'fake'}, 'CAS')

    def test_auth_errors_do_not_include_personal_fields(self):
        payload = {'code': 200, 'second': True, 'userName': 'private-user',
                   'mobile': 'private-phone', 'message': 'private-message',
                   'moduleCodes': ['sms', 'https://example.test/?token=private']}
        with self.assertRaises(AuthenticationActionRequired) as caught:
            require_login_token(payload, 'CAS')
        self.assertNotIn('private', str(caught.exception))
        self.assertIn('sms', str(caught.exception))

    def test_valid_login_token(self):
        self.assertEqual(require_login_token({'code': 200, 'loginToken': 'test'}, 'CAS'), 'test')

    def test_missing_token_is_not_success(self):
        with self.assertRaises(RuntimeError):
            require_login_token({'code': 200}, 'CAS')

    def test_icourse_verification_fails_closed(self):
        for body in ('<html>login</html>', '{"code":401}', '[]', '{"code":false}'):
            result = response('https://webvpn.fudan.edu.cn/api')
            result._content = body.encode()
            with self.assertRaises(RuntimeError):
                WebVPNSession._validate_icourse_response(result)

    def test_icourse_verification_accepts_api_success(self):
        result = response('https://webvpn.fudan.edu.cn/api')
        result._content = b'{"code":0,"data":{"id":123}}'
        WebVPNSession._validate_icourse_response(result)

    def test_auth_starts_at_gateway_and_keeps_redirect_context(self):
        vpn = WebVPNSession()
        first = response("https://webvpn.fudan.edu.cn/login", 302)
        first.headers["Location"] = "https://id.fudan.edu.cn/start"
        second = response("https://id.fudan.edu.cn/start", 302)
        second.headers["Location"] = "/ac/?lck=test-context"
        vpn.session.get = Mock(side_effect=[first, second])
        self.assertEqual(vpn._get_auth_context(), (
            "test-context", "https://webvpn.fudan.edu.cn"))
        self.assertEqual(vpn.session.get.call_args_list[0].args[0],
                         "https://webvpn.fudan.edu.cn/login?cas_login=true")
        self.assertEqual(vpn.session.get.call_count, 2)

    def test_auth_html_gateway_falls_back_to_cas_in_same_session(self):
        vpn = WebVPNSession()
        redirect = response("https://id.fudan.edu.cn/authenticate", 302)
        redirect.headers["Location"] = "/ac/?lck=fallback-context"
        vpn.session.get = Mock(side_effect=[
            response("https://webvpn.fudan.edu.cn/login"), redirect,
        ])
        self.assertEqual(vpn._get_auth_context()[0], "fallback-context")
        self.assertTrue(vpn.session.get.call_args_list[1].args[0].startswith(
            "https://id.fudan.edu.cn/idp/authCenter/authenticate?"))

    def test_auth_redirect_loop_is_bounded(self):
        vpn = WebVPNSession()
        redirect = response("https://webvpn.fudan.edu.cn/login", 302)
        redirect.headers["Location"] = "/login"
        vpn.session.get = Mock(return_value=redirect)
        with self.assertRaisesRegex(RuntimeError, "redirect limit"):
            vpn._get_auth_context()
        self.assertEqual(vpn.session.get.call_count, 10)

    def test_auth_rejects_external_redirect(self):
        vpn = WebVPNSession()
        redirect = response("https://webvpn.fudan.edu.cn/login", 302)
        redirect.headers["Location"] = "https://untrusted.example/?lck=test"
        vpn.session.get = Mock(return_value=redirect)
        with self.assertRaisesRegex(RuntimeError, "Unexpected destination"):
            vpn._get_auth_context()

    def test_login_page_is_not_success(self):
        vpn = WebVPNSession()
        vpn.session.get = Mock(return_value=response(
            "https://webvpn.fudan.edu.cn:443/login"))
        with self.assertRaisesRegex(RuntimeError, "returned its login page"):
            vpn._establish_session("https://webvpn.fudan.edu.cn/cas")

    def test_successful_portal(self):
        vpn = WebVPNSession()
        vpn.session.get = Mock(return_value=response(
            "https://webvpn.fudan.edu.cn/"))
        vpn._establish_session("https://webvpn.fudan.edu.cn/cas")

    def test_server_error(self):
        with self.assertRaises(requests.HTTPError):
            WebVPNSession._validate_session_response(response(
                "https://webvpn.fudan.edu.cn/", 502))

    def test_anonymous_cookie_after_timeout_is_not_success(self):
        vpn = WebVPNSession()
        vpn.session.cookies.set("wengine_vpn_ticket", "anonymous")
        vpn.session.get = Mock(side_effect=[
            requests.exceptions.Timeout(),
            response("https://webvpn.fudan.edu.cn/login"),
        ])
        with self.assertRaisesRegex(RuntimeError, "returned its login page"):
            vpn._establish_session("https://webvpn.fudan.edu.cn/cas")

    def test_valid_session_after_timeout(self):
        vpn = WebVPNSession()
        vpn.session.cookies.set("wengine_vpn_ticket", "test-session")
        vpn.session.get = Mock(side_effect=[
            requests.exceptions.Timeout(),
            response("https://webvpn.fudan.edu.cn/https/protected/"),
        ])
        vpn._establish_session("https://webvpn.fudan.edu.cn/cas")


if __name__ == "__main__":
    unittest.main()
