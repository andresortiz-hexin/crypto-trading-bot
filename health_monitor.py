"""Health Monitor - Auto-healing watchdog for the trading bot.
Monitors bot process health via stdout activity detection.
Auto-restarts on crash or paralysis. Sends Telegram alerts.
No modifications needed in bot.py - monitors subprocess externally.
"""
import os
import sys
import time
import threading
import logging
import subprocess
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [monitor] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger('monitor')

# Config
MAX_SILENCE = 300  # 5 min no output = stuck
MAX_RESTARTS = 15  # max auto-restarts
RESTART_COOLDOWN = 30  # seconds between restarts
CHECK_INTERVAL = 20  # health check every 20s
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


class BotSupervisor:
    """Monitors and auto-restarts bot.py on failure or paralysis."""

    def __init__(self):
        self.process = None
        self.total_restarts = 0
        self.consecutive_failures = 0
        self.last_restart = 0
        self.last_output = time.time()
        self.start_time = time.time()
        self.output_lock = threading.Lock()

    def start_bot(self):
        """Start or restart the bot process."""
        if self.process and self.process.poll() is None:
            log.warning('Terminating stuck bot...')
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            except Exception as e:
                log.error(f'Kill error: {e}')

        log.info(f'Starting bot (attempt #{self.total_restarts + 1})...')
        self.process = subprocess.Popen(
            [sys.executable, '-u', 'bot.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        with self.output_lock:
            self.last_output = time.time()
        self.last_restart = time.time()
        self.total_restarts += 1

        # Start output reader thread
        t = threading.Thread(target=self._read_output, daemon=True)
        t.start()

    def _read_output(self):
        """Read bot stdout in background thread."""
        try:
            for line in self.process.stdout:
                line = line.rstrip()
                if line:
                    print(line, flush=True)
                    with self.output_lock:
                        self.last_output = time.time()
        except Exception:
            pass

    def check_health(self):
        """Returns (healthy, reason)."""
        if self.process is None:
            return False, 'not_started'

        exit_code = self.process.poll()
        if exit_code is not None:
            return False, f'crashed (exit={exit_code})'

        # Check output activity
        with self.output_lock:
            silence = time.time() - self.last_output

        # Grace period after start (3 min)
        since_start = time.time() - self.last_restart
        if since_start < 180:
            return True, 'starting_up'

        if silence > MAX_SILENCE:
            return False, f'no_output ({silence:.0f}s)'

        return True, 'ok'

    def handle_failure(self, reason):
        """Auto-recover from failure."""
        self.consecutive_failures += 1
        log.error(f'UNHEALTHY: {reason} (failure #{self.consecutive_failures})')

        if self.total_restarts >= MAX_RESTARTS:
            msg = (
                f'<b>CRITICAL: Max restarts ({MAX_RESTARTS}) reached</b>\n'
                f'Reason: {reason}\n'
                'Manual intervention required!'
            )
            send_telegram(msg)
            log.critical(msg)
            return False

        # Cooldown
        elapsed = time.time() - self.last_restart
        if elapsed < RESTART_COOLDOWN:
            time.sleep(RESTART_COOLDOWN - elapsed)

        uptime = time.time() - self.start_time
        msg = (
            f'<b>BOT AUTO-RESTART #{self.total_restarts + 1}</b>\n'
            f'Reason: {reason}\n'
            f'Consecutive failures: {self.consecutive_failures}\n'
            f'Total uptime: {uptime/3600:.1f}h'
        )
        send_telegram(msg)
        log.info(f'Restarting bot... (reason: {reason})')

        self.start_bot()
        return True

    def run(self):
        """Main supervisor loop."""
        log.info('=== Health Monitor Started ===')
        send_telegram(
            '<b>Health Monitor Active</b>\n'
            f'Max silence: {MAX_SILENCE}s\n'
            f'Max restarts: {MAX_RESTARTS}\n'
            'Auto-restart on crash or hang'
        )

        self.start_bot()

        while True:
            try:
                healthy, reason = self.check_health()

                if healthy:
                    self.consecutive_failures = 0
                else:
                    ok = self.handle_failure(reason)
                    if not ok:
                        send_telegram(
                            '<b>CRITICAL: Monitor stopped</b>\n'
                            'Max restarts exceeded.'
                        )
                        sys.exit(1)

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                log.info('Stopped by user')
                if self.process and self.process.poll() is None:
                    self.process.terminate()
                break
            except Exception as e:
                log.error(f'Monitor error: {e}')
                time.sleep(CHECK_INTERVAL)


if __name__ == '__main__':
    BotSupervisor().run()
