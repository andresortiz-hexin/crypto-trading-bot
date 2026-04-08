"""Health Monitor - Auto-healing watchdog for the trading bot.
Monitors bot health via heartbeat file, detects paralysis,
auto-restarts on failure, and sends Telegram alerts.
"""
import os
import sys
import time
import signal
import logging
import subprocess
import requests
import traceback
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [health_monitor] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger('health_monitor')

# Config
HEARTBEAT_FILE = '/tmp/bot_heartbeat'
HEALTH_FILE = '/tmp/bot_health.json'
MAX_HEARTBEAT_AGE = 300  # 5 min without heartbeat = dead
MAX_CONSECUTIVE_ERRORS = 5
MAX_RESTARTS = 10  # Max restarts before giving up
RESTART_COOLDOWN = 30  # seconds between restarts
CHECK_INTERVAL = 30  # check every 30 seconds
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')


def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': msg,
            'parse_mode': 'HTML'
        }, timeout=10)
    except Exception:
        pass


def read_heartbeat():
    """Read last heartbeat timestamp from file."""
    try:
        if os.path.exists(HEARTBEAT_FILE):
            with open(HEARTBEAT_FILE, 'r') as f:
                return float(f.read().strip())
    except (ValueError, IOError):
        pass
    return None


def read_health():
    """Read health status JSON from bot."""
    import json
    try:
        if os.path.exists(HEALTH_FILE):
            with open(HEALTH_FILE, 'r') as f:
                return json.load(f)
    except (ValueError, IOError):
        pass
    return {}


class BotSupervisor:
    """Supervises bot.py process with auto-healing."""

    def __init__(self):
        self.process = None
        self.restart_count = 0
        self.total_restarts = 0
        self.last_restart_time = 0
        self.consecutive_failures = 0
        self.start_time = time.time()
        self.last_healthy_time = time.time()

    def start_bot(self):
        """Start or restart the bot process."""
        if self.process and self.process.poll() is None:
            log.warning('Killing stuck bot process...')
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            except Exception as e:
                log.error(f'Kill error: {e}')

        # Clear old heartbeat
        try:
            if os.path.exists(HEARTBEAT_FILE):
                os.remove(HEARTBEAT_FILE)
        except IOError:
            pass

        log.info(f'Starting bot (restart #{self.total_restarts})...')
        self.process = subprocess.Popen(
            [sys.executable, 'bot.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        self.last_restart_time = time.time()
        self.total_restarts += 1
        return True

    def stream_output(self):
        """Non-blocking read of bot stdout."""
        if not self.process or not self.process.stdout:
            return
        import select
        while True:
            try:
                ready, _, _ = select.select(
                    [self.process.stdout], [], [], 0
                )
                if not ready:
                    break
                line = self.process.stdout.readline()
                if not line:
                    break
                print(line.rstrip())
            except Exception:
                break

    def check_health(self):
        """Check if bot is healthy. Returns (healthy, reason)."""
        # Check if process is alive
        if self.process is None:
            return False, 'process_not_started'

        if self.process.poll() is not None:
            exit_code = self.process.returncode
            return False, f'process_exited (code={exit_code})'

        # Check heartbeat freshness
        hb = read_heartbeat()
        if hb is None:
            # Allow grace period after restart (2 min)
            elapsed = time.time() - self.last_restart_time
            if elapsed > 120:
                return False, 'no_heartbeat_after_startup'
            return True, 'starting_up'

        age = time.time() - hb
        if age > MAX_HEARTBEAT_AGE:
            return False, f'heartbeat_stale ({age:.0f}s old)'

        # Check health file for error counts
        health = read_health()
        errors = health.get('consecutive_errors', 0)
        if errors >= MAX_CONSECUTIVE_ERRORS:
            return False, f'too_many_errors ({errors})'

        self.last_healthy_time = time.time()
        return True, 'ok'

    def handle_failure(self, reason):
        """Handle a detected failure with auto-recovery."""
        self.consecutive_failures += 1
        log.error(
            f'Bot unhealthy: {reason} '
            f'(failure {self.consecutive_failures})'
        )

        if self.total_restarts >= MAX_RESTARTS:
            msg = (
                f'<b>CRITICAL: Bot exceeded max restarts '
                f'({MAX_RESTARTS})</b>\n'
                f'Reason: {reason}\n'
                f'Manual intervention required!'
            )
            log.critical(msg)
            send_telegram(msg)
            return False

        # Cooldown between restarts
        elapsed = time.time() - self.last_restart_time
        if elapsed < RESTART_COOLDOWN:
            wait = RESTART_COOLDOWN - elapsed
            log.info(f'Cooldown: waiting {wait:.0f}s...')
            time.sleep(wait)

        # Send alert
        uptime = time.time() - self.start_time
        msg = (
            f'<b>BOT AUTO-RESTART #{self.total_restarts}</b>\n'
            f'Reason: {reason}\n'
            f'Failures: {self.consecutive_failures}\n'
            f'Uptime was: {uptime/3600:.1f}h'
        )
        send_telegram(msg)

        # Restart
        self.start_bot()
        return True

    def run(self):
        """Main supervisor loop."""
        log.info('=== Health Monitor Started ===')
        send_telegram(
            '<b>Health Monitor Active</b>\n'
            'Auto-restart on failure\n'
            f'Heartbeat timeout: {MAX_HEARTBEAT_AGE}s\n'
            f'Max restarts: {MAX_RESTARTS}'
        )

        self.start_bot()

        while True:
            try:
                # Stream bot output to our stdout
                self.stream_output()

                # Check health
                healthy, reason = self.check_health()

                if healthy:
                    self.consecutive_failures = 0
                else:
                    can_continue = self.handle_failure(reason)
                    if not can_continue:
                        log.critical('Giving up after max restarts')
                        send_telegram(
                            '<b>CRITICAL: Monitor giving up</b>\n'
                            'Max restarts exceeded. '
                            'Manual intervention needed.'
                        )
                        sys.exit(1)

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                log.info('Monitor stopped by user')
                if self.process:
                    self.process.terminate()
                break
            except Exception as e:
                log.error(f'Monitor error: {e}')
                time.sleep(CHECK_INTERVAL)


if __name__ == '__main__':
    supervisor = BotSupervisor()
    supervisor.run()
