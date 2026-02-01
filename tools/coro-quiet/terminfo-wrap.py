#!/usr/bin/env python3
"""terminfo-wrap: Translate raw ANSI escapes to terminfo sequences.

Wraps any command and translates raw escape codes (like \r, \x1B[K) to
terminfo sequences (tput cr, tput el) for Termux compatibility.

Usage:
    ./terminfo-wrap.py coro "your task"
    ./terminfo-wrap.py coro  # interactive mode
"""

import os
import pty
import re
import subprocess
import sys
import select
import signal

# Cache terminfo sequences at startup (much faster than calling tput each time)
def get_terminfo_seq(cap):
    """Get terminfo sequence for a capability."""
    try:
        result = subprocess.run(['tput', cap], capture_output=True)
        return result.stdout
    except:
        return b''

# Pre-cache common terminfo sequences
TERMINFO = {
    'cr': get_terminfo_seq('cr'),      # carriage return
    'el': get_terminfo_seq('el'),      # clear to end of line
    'el1': get_terminfo_seq('el1'),    # clear to beginning of line
    'ed': get_terminfo_seq('ed'),      # clear to end of screen
    'cuu1': get_terminfo_seq('cuu1'),  # cursor up 1
    'cud1': get_terminfo_seq('cud1'),  # cursor down 1
    'cuf1': get_terminfo_seq('cuf1'),  # cursor forward 1
    'cub1': get_terminfo_seq('cub1'),  # cursor back 1
    'home': get_terminfo_seq('home'),  # cursor home
    'clear': get_terminfo_seq('clear'), # clear screen
}

# Escape sequence patterns and their terminfo replacements
# Order matters - longer sequences first
ESCAPE_TRANSLATIONS = [
    # Cursor movement
    (re.compile(rb'\x1b\[(\d+)A'), lambda m: TERMINFO['cuu1'] * int(m.group(1))),  # cursor up N
    (re.compile(rb'\x1b\[(\d+)B'), lambda m: TERMINFO['cud1'] * int(m.group(1))),  # cursor down N
    (re.compile(rb'\x1b\[(\d+)C'), lambda m: TERMINFO['cuf1'] * int(m.group(1))),  # cursor forward N
    (re.compile(rb'\x1b\[(\d+)D'), lambda m: TERMINFO['cub1'] * int(m.group(1))),  # cursor back N
    (re.compile(rb'\x1b\[1A'), lambda m: TERMINFO['cuu1']),   # cursor up 1
    (re.compile(rb'\x1b\[1B'), lambda m: TERMINFO['cud1']),   # cursor down 1
    (re.compile(rb'\x1b\[A'), lambda m: TERMINFO['cuu1']),    # cursor up 1 (no number)
    (re.compile(rb'\x1b\[B'), lambda m: TERMINFO['cud1']),    # cursor down 1 (no number)

    # Line clearing
    (re.compile(rb'\x1b\[2K'), lambda m: TERMINFO['el1'] + TERMINFO['el']),  # clear entire line
    (re.compile(rb'\x1b\[K'), lambda m: TERMINFO['el']),      # clear to end of line
    (re.compile(rb'\x1b\[0K'), lambda m: TERMINFO['el']),     # clear to end of line
    (re.compile(rb'\x1b\[1K'), lambda m: TERMINFO['el1']),    # clear to beginning of line

    # Screen clearing
    (re.compile(rb'\x1b\[2J'), lambda m: TERMINFO['clear']),  # clear screen
    (re.compile(rb'\x1b\[J'), lambda m: TERMINFO['ed']),      # clear to end of screen

    # Cursor position (simplified - just go home for [H with no args)
    (re.compile(rb'\x1b\[H'), lambda m: TERMINFO['home']),    # cursor home
    (re.compile(rb'\x1b\[(\d+);(\d+)H'), None),               # absolute position - pass through

    # Carriage return
    (re.compile(rb'\r(?!\n)'), lambda m: TERMINFO['cr']),     # CR not followed by LF
]

def translate_escapes(data):
    """Translate raw ANSI escapes to terminfo sequences."""
    for pattern, replacement in ESCAPE_TRANSLATIONS:
        if replacement is None:
            continue  # pass through
        if callable(replacement):
            data = pattern.sub(replacement, data)
        else:
            data = pattern.sub(replacement, data)
    return data

def run_with_pty(args):
    """Run command in a PTY, translating escape sequences."""

    # Create a pseudo-terminal
    master_fd, slave_fd = pty.openpty()

    # Fork and run the command
    pid = os.fork()

    if pid == 0:
        # Child process
        os.close(master_fd)
        os.setsid()

        # Set up slave as controlling terminal
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)

        if slave_fd > 2:
            os.close(slave_fd)

        # Execute the command
        os.execvp(args[0], args)

    # Parent process
    os.close(slave_fd)

    # Set up signal handler for child exit
    def handle_sigchld(signum, frame):
        pass
    signal.signal(signal.SIGCHLD, handle_sigchld)

    # Copy data between PTY and real terminal
    try:
        while True:
            # Wait for data from either PTY or stdin
            rlist, _, _ = select.select([master_fd, sys.stdin.fileno()], [], [], 0.1)

            for fd in rlist:
                if fd == master_fd:
                    # Data from child - translate and output
                    try:
                        data = os.read(master_fd, 4096)
                        if not data:
                            break
                        translated = translate_escapes(data)
                        os.write(sys.stdout.fileno(), translated)
                    except OSError:
                        break

                elif fd == sys.stdin.fileno():
                    # Data from user - pass to child
                    try:
                        data = os.read(sys.stdin.fileno(), 4096)
                        if data:
                            os.write(master_fd, data)
                    except OSError:
                        break

            # Check if child has exited
            result = os.waitpid(pid, os.WNOHANG)
            if result[0] != 0:
                # Drain remaining output
                try:
                    while True:
                        rlist, _, _ = select.select([master_fd], [], [], 0.1)
                        if not rlist:
                            break
                        data = os.read(master_fd, 4096)
                        if not data:
                            break
                        translated = translate_escapes(data)
                        os.write(sys.stdout.fileno(), translated)
                except:
                    pass
                break

    except KeyboardInterrupt:
        os.kill(pid, signal.SIGINT)
        os.waitpid(pid, 0)
        return 130

    finally:
        os.close(master_fd)

    # Get exit status (may already be reaped)
    try:
        _, status = os.waitpid(pid, 0)
        return os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1
    except ChildProcessError:
        return 0  # Already reaped

def main():
    if len(sys.argv) < 2:
        print("Usage: terminfo-wrap.py <command> [args...]")
        print()
        print("Examples:")
        print("  terminfo-wrap.py coro 'your task'")
        print("  terminfo-wrap.py coro  # interactive")
        print()
        print("Translates raw ANSI escapes to terminfo sequences")
        print("for Termux compatibility.")
        return 1

    # Check if terminfo works
    if not TERMINFO['cr']:
        print("Warning: tput cr returned empty - terminfo may not work", file=sys.stderr)

    args = sys.argv[1:]
    return run_with_pty(args)

if __name__ == '__main__':
    sys.exit(main())
