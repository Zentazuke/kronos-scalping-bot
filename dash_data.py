import glob, json, os, re, shutil, sqlite3, time
folder_list = glob.glob('/sessions/*/mnt/Trading Bot')
out = {'source': None, 'trades': [], 'error': '', 'folder': folder_list[0] if folder_list else None, 'ts': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

def from_db(folder):
    src = os.path.join(folder, 'journal.db')
    dst = '/tmp/_dash_journal.db'
    shutil.copy(src, dst)
    c = sqlite3.connect('file:' + dst + '?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute('SELECT id, ts_open, ts_close, symbol, direction, amount, entry_price, tp_price, sl_price, exit_price, pnl, status, variant, confluence_votes, p_up, p_down, adx, rsi FROM trades ORDER BY id')]
    c.close()
    return rows

def from_log(folder):
    txt = open(os.path.join(folder, 'bot.log'), encoding='utf-8', errors='replace').read()
    brackets = {}
    for m in re.finditer(r'^([\d-]+ [\d:]+).*?([A-Z]+/[A-Z]+): bracket live — (STRAT_\w+) ([\d.]+) @ ([\d.]+) \| TP ([\d.]+) \| SL ([\d.]+)', txt, re.M):
        brackets[(m.group(2), m.group(4), m.group(5))] = m.groups()
    trades = {}
    for m in re.finditer(r'^([\d-]+ [\d:]+).*?([A-Z]+/[A-Z]+): trade #(\d+) journaled — (STRAT_\w+) ([\d.]+) @ ([\d.]+)', txt, re.M):
        ts, sym, tid, d, amt, entry = m.groups()
        b = brackets.get((sym, amt, entry))
        trades[int(tid)] = {'id': int(tid), 'ts_open': ts, 'ts_close': None, 'symbol': sym,
            'direction': d, 'amount': amt, 'entry_price': entry,
            'tp_price': b[5] if b else None, 'sl_price': b[6] if b else None,
            'exit_price': None, 'pnl': None, 'status': 'OPEN', 'variant': None,
            'confluence_votes': None, 'p_up': None, 'p_down': None, 'adx': None, 'rsi': None}
    for m in re.finditer(r'^([\d-]+ [\d:]+).*?([A-Z]+/[A-Z]+): trade #(\d+) closed (\w+) — entry ([\d.]+) exit ([\d.]+) pnl (-?[\d.]+)', txt, re.M):
        ts, sym, tid, st, entry, exitp, pnl = m.groups()
        t = trades.setdefault(int(tid), {'id': int(tid), 'ts_open': None, 'symbol': sym,
            'direction': '?', 'amount': None, 'entry_price': entry, 'tp_price': None,
            'sl_price': None, 'variant': None, 'confluence_votes': None,
            'p_up': None, 'p_down': None, 'adx': None, 'rsi': None})
        t.update({'status': st, 'exit_price': exitp, 'pnl': pnl, 'ts_close': ts})
    return [trades[k] for k in sorted(trades)]

if out['folder']:
    for attempt in range(3):
        try:
            out['trades'] = from_db(out['folder']); out['source'] = 'db'; break
        except Exception as e:
            out['error'] = 'db: ' + str(e); time.sleep(2)
    if out['source'] is None:
        try:
            out['trades'] = from_log(out['folder']); out['source'] = 'log'
        except Exception as e:
            out['error'] += ' | log: ' + str(e)
else:
    out['error'] = 'Trading Bot folder not mounted in this session'
print(json.dumps(out))
