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
