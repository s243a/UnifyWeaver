"""Display module using tput for Termux-compatible terminal control."""

import os
import subprocess
import sys
import threading
import time
from typing import Optional


def _get_tput_sequence(cap: str) -> str:
    """Get escape sequence for a terminfo capability."""
    try:
        result = subprocess.run(
            ['tput', cap],
            capture_output=True,
            text=True,
            timeout=1
        )
        return result.stdout
    except Exception:
        return ''


_SEQUENCES = {
    'cr': '\r',
    'el': _get_tput_sequence('el'),
    'civis': _get_tput_sequence('civis'),
    'cnorm': _get_tput_sequence('cnorm'),
    'sc': _get_tput_sequence('sc'),
    'rc': _get_tput_sequence('rc'),
}


def tput_write(cap: str) -> None:
    """Write cached tput sequence to stdout."""
    seq = _SEQUENCES.get(cap, '')
    if seq:
        os.write(sys.stdout.fileno(), seq.encode())


class TerminalControl:
    """Terminal control using tput commands."""

    @staticmethod
    def cr() -> None:
        tput_write('cr')

    @staticmethod
    def el() -> None:
        tput_write('el')

    @staticmethod
    def clear_line() -> None:
        tput_write('cr')
        tput_write('el')

    @staticmethod
    def save_cursor() -> None:
        tput_write('sc')

    @staticmethod
    def restore_cursor() -> None:
        tput_write('rc')

    @staticmethod
    def hide_cursor() -> None:
        tput_write('civis')

    @staticmethod
    def show_cursor() -> None:
        tput_write('cnorm')

    @staticmethod
    def move_up(n: int = 1) -> None:
        os.write(sys.stdout.fileno(), f"\033[{n}A".encode())

    @staticmethod
    def move_down(n: int = 1) -> None:
        os.write(sys.stdout.fileno(), f"\033[{n}B".encode())

    @staticmethod
    def cols() -> int:
        try:
            result = subprocess.run(['tput', 'cols'], capture_output=True, text=True, timeout=1)
            return int(result.stdout.strip() or '80')
        except (ValueError, Exception):
            return 80

    @staticmethod
    def lines() -> int:
        try:
            result = subprocess.run(['tput', 'lines'], capture_output=True, text=True, timeout=1)
            return int(result.stdout.strip() or '24')
        except (ValueError, Exception):
            return 24


class Spinner:
    """Animated spinner that updates in place using tput."""

    FRAMES = ['\u280b', '\u2819', '\u2839', '\u2838', '\u283c', '\u2834', '\u2826', '\u2827', '\u2807', '\u280f']
    INTERVAL = 0.1

    def __init__(self, message: str = "Working"):
        self.message = message
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame = 0
        self._tc = TerminalControl()
        self._cols = self._tc.cols()
        self._lock = threading.Lock()
        self._start_time = 0.0

    def _truncate(self, text: str, width: int) -> str:
        max_len = width - 2
        if len(text) > max_len:
            return text[:max_len - 3] + '...'
        return text

    def _animate(self) -> None:
        self._tc.hide_cursor()
        try:
            while self._running:
                with self._lock:
                    elapsed = time.time() - self._start_time
                    elapsed_str = f" ({elapsed:.0f}s)" if elapsed >= 2 else ""
                    frame = self.FRAMES[self._frame % len(self.FRAMES)]
                    display = self._truncate(
                        self.message + elapsed_str, self._cols
                    )
                    self._tc.clear_line()
                    sys.stdout.write(f"{frame} {display}")
                    sys.stdout.flush()
                    self._frame += 1
                time.sleep(self.INTERVAL)
        finally:
            self._tc.show_cursor()

    def start(self) -> None:
        if self._running:
            return
        self._start_time = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self, final_message: Optional[str] = None) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None

        with self._lock:
            self._tc.clear_line()
            sys.stdout.write(f"  {self.message}\n")
            if final_message:
                sys.stdout.write(final_message + '\n')
            sys.stdout.flush()

    def update(self, message: str) -> None:
        with self._lock:
            if message != self.message:
                self._tc.clear_line()
                sys.stdout.write(f"  {self.message}\n")
                sys.stdout.flush()
                self.message = message

    def __enter__(self) -> 'Spinner':
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


class ProgressBar:
    """Progress bar for streaming that updates in place."""

    def __init__(self, total: int = 100, width: int = 30,
                 prefix: str = "", suffix: str = ""):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.suffix = suffix
        self.current = 0
        self._tc = TerminalControl()

    def update(self, current: Optional[int] = None,
               increment: int = 0, suffix: str = None) -> None:
        if current is not None:
            self.current = current
        else:
            self.current += increment

        if suffix is not None:
            self.suffix = suffix

        self._render()

    def _render(self) -> None:
        if self.total <= 0:
            pct = 0
        else:
            pct = min(100, int(100 * self.current / self.total))

        filled = int(self.width * pct / 100)
        bar = '\u2588' * filled + '\u2591' * (self.width - filled)

        self._tc.clear_line()
        sys.stdout.write(f"{self.prefix}[{bar}] {pct}% {self.suffix}")
        sys.stdout.flush()

    def finish(self, message: str = "") -> None:
        self.current = self.total
        self._render()
        sys.stdout.write('\n')
        if message:
            sys.stdout.write(message + '\n')
        sys.stdout.flush()


class StatusLine:
    """Status line at the bottom of the terminal."""

    def __init__(self):
        self._tc = TerminalControl()
        self._saved = False
        self._status = ""

    def show(self, status: str) -> None:
        self._status = status
        self._tc.clear_line()
        cols = self._tc.cols()
        if len(status) > cols - 1:
            status = status[:cols - 4] + '...'
        sys.stdout.write(status)
        sys.stdout.flush()

    def clear(self) -> None:
        self._tc.clear_line()
        sys.stdout.flush()
        self._status = ""

    def update(self, status: str) -> None:
        self.show(status)


class StreamDisplay:
    """Display for streaming output with progress indication."""

    def __init__(self, show_tokens: bool = True):
        self.show_tokens = show_tokens
        self._tc = TerminalControl()
        self._char_count = 0
        self._line_count = 0

    def start(self, label: str = "Streaming") -> None:
        sys.stdout.write(f"\n{label}:\n")
        sys.stdout.flush()
        self._char_count = 0
        self._line_count = 0

    def chunk(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self._char_count += len(text)
        self._line_count += text.count('\n')

    def finish(self, tokens: Optional[dict] = None) -> None:
        sys.stdout.write('\n')
        if self.show_tokens and tokens:
            input_t = tokens.get('input', 0)
            output_t = tokens.get('output', 0)
            sys.stdout.write(f"  [Tokens: input={input_t}, output={output_t}]\n")
        sys.stdout.flush()


class DisplayMode:
    """Factory for display components based on mode."""

    APPEND_ONLY = 'append'
    NCURSES = 'ncurses'

    @classmethod
    def supports_ncurses(cls) -> bool:
        return bool(_SEQUENCES.get('el'))

    @classmethod
    def get_spinner(cls, message: str, mode: str = None) -> 'Spinner':
        if mode is None:
            mode = cls.NCURSES if cls.supports_ncurses() else cls.APPEND_ONLY

        if mode == cls.APPEND_ONLY:
            return AppendOnlySpinner(message)
        return Spinner(message)

    @classmethod
    def get_progress(cls, total: int, mode: str = None, **kwargs) -> 'ProgressBar':
        if mode is None:
            mode = cls.NCURSES if cls.supports_ncurses() else cls.APPEND_ONLY

        if mode == cls.APPEND_ONLY:
            return AppendOnlyProgress(total, **kwargs)
        return ProgressBar(total, **kwargs)


class AppendOnlySpinner:
    """Fallback spinner for append-only mode."""

    def __init__(self, message: str = "Working"):
        self.message = message
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _animate(self) -> None:
        count = 0
        while self._running:
            if count % 10 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            count += 1
            time.sleep(0.1)

    def start(self) -> None:
        if self._running:
            return
        sys.stdout.write(f"[{self.message}]")
        sys.stdout.flush()
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self, final_message: Optional[str] = None) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None

        if final_message:
            sys.stdout.write(f" {final_message}\n")
        else:
            sys.stdout.write(" done\n")
        sys.stdout.flush()

    def update(self, message: str) -> None:
        pass

    def __enter__(self) -> 'AppendOnlySpinner':
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


class AppendOnlyProgress:
    """Fallback progress bar for append-only mode."""

    def __init__(self, total: int = 100, width: int = 30,
                 prefix: str = "", suffix: str = ""):
        self.total = total
        self.prefix = prefix
        self._last_pct = -1

    def update(self, current: Optional[int] = None,
               increment: int = 0, suffix: str = None) -> None:
        if current is None:
            current = increment

        if self.total <= 0:
            return

        pct = min(100, int(100 * current / self.total))
        milestone = (pct // 25) * 25
        if milestone > self._last_pct:
            sys.stdout.write(f"[{milestone}%]")
            sys.stdout.flush()
            self._last_pct = milestone

    def finish(self, message: str = "") -> None:
        if self._last_pct < 100:
            sys.stdout.write("[100%]")
        sys.stdout.write(f" {message}\n" if message else "\n")
        sys.stdout.flush()
