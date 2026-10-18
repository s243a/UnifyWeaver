#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Streaming result fidelity and bounded VM lifetime regressions."""
import json
from pathlib import Path
import subprocess
import sys

binary = Path(__file__).parent / 'diff' / 'diff_uwresolve'
proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)
checks = 0

def row(name='p', version=1, request='p'):
    return dict(id='driver-selftest', catalog=dict(packages=[[name,[version,0,0]]],
                depends=[],conflicts=[],base=[],installed=[],requested=[]),
                query='resolve',args=[request])

def send(value):
    proc.stdin.write(json.dumps(value) + '\n')
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line, 'driver exited before returning a row'
    return json.loads(line)

def check(cond, message):
    global checks
    assert cond, message
    checks += 1

try:
    check(send(row())['ok'] == [['p',[1,0,0]]], 'ordinary result')
    for value in (2147483648, -2147483649, 4294967296):
        check('crash' in send(row(version=value)), 'reject WAM integer overflow')
    check('crash' in send(row(version=9223372036854775808)), 'reject JSON integer overflow')
    check('crash' in send(row('p\0q')), 'reject NUL package identity')
    check('crash' in send(row('\ud800')), 'reject unpaired surrogate')
    check(send(row('\U0001f600',request='\U0001f600'))['ok'] == [['\U0001f600',[1,0,0]]],
          'surrogate pair preserves Unicode package identity')
    check(send(row(request='missing'))['fail'], 'logical failure remains a failure')
    check(send(row())['ok'] == [['p',[1,0,0]]], 'valid row after errors')

    # Keep stdin open and measure the same running process after warmup.
    # A 64-package row exposes retained heap/trail growth within 1000 rows.
    many = row('p0',request='p0')
    many['catalog']['packages'] = [['p'+str(i),[1,0,0]] for i in range(64)]
    def rss():
        status = Path('/proc') / str(proc.pid) / 'status'
        if not status.exists(): return None
        return next(int(line.split()[1]) for line in status.read_text().splitlines()
                    if line.startswith('VmRSS:'))
    for _ in range(20): send(many)
    baseline = rss()
    for _ in range(1000):
        check(send(many)['ok'] == [['p0',[1,0,0]]], 'repeated indexed catalog')
    final = rss()
    if baseline is not None and final is not None:
        check(final - baseline < 8192, 'VM memory retained across rows')
        print(f'RSS after warmup={baseline} KiB; after 1000 rows={final} KiB')
    proc.stdin.close()
    check(proc.wait(timeout=10) == 0, 'driver clean exit')
finally:
    if proc.poll() is None:
        proc.kill()
        proc.wait()
print(f'{checks} driver checks passed')
