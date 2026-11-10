"""Keep the verified session in memory and process subscribed courses once."""
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api.webvpn import WebVPNSession, AuthenticationActionRequired
from src.api.qr_verification import QRVerification, seal
from src.api.session_cache import save_session
from src.runtime import config


def publish(image):
    root = Path(os.environ['ICOURSE_PAIR_DIR'])
    encrypted = seal(image, config.STUDENT_ID, config.PASSWORD, 'qr-image')
    temporary = root / 'qr.pending'
    temporary.write_bytes(encrypted)
    temporary.replace(root / 'qr.enc')
    print('[QR] Encrypted challenge ready for the account owner.', flush=True)


def main():
    vpn = None
    for attempt in range(3):
        vpn = WebVPNSession(verification_handler=QRVerification(publish))
        try:
            vpn.login()
            vpn.authenticate_icourse()
            save_session(vpn)
            break
        except AuthenticationActionRequired:
            vpn.session.close()
            raise
        except Exception:
            vpn.session.close()
            if attempt == 2:
                raise
            print('[Pair] Transient login failure; retrying with a fresh session.', flush=True)
            time.sleep(20)
    import main as application
    original_login = application.login_with_retry
    available = [vpn]

    def first_login(*args, **kwargs):
        return available.pop() if available else original_login(*args, **kwargs)

    application.login_with_retry = first_login
    try:
        application.run()
    finally:
        vpn.session.close()


if __name__ == '__main__':
    main()
