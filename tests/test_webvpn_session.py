import unittest
from unittest.mock import Mock

import requests

from src.api.webvpn import WebVPNSession


def response(url, status=200):
    result = requests.Response()
    result.status_code = status
    result.url = url
    return result


class WebVPNSessionTests(unittest.TestCase):
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
