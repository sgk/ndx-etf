#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NDX (^ndx) と TQQQ/QLD の日次OHLCを Stooq から取得し、SQLiteに追記。
対話シェルで「NDXの金額」を入れると、TQQQ/QLDの想定レンジと Trading Signal を表示。

使い方:
  python ndx_etf.py update
  python ndx_etf.py shell
  python ndx_etf.py update shell   # 更新してからシェルへ
"""

from __future__ import annotations

import sys
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from io import StringIO


# ====== 設定 ======
DB_PATH = "ndx_etf.sqlite3"

STOOQ_DAILY_CSV = "https://stooq.com/q/d/l/?s={symbol}&i=d"

SYMBOL_NDX = "^ndx"
SYMBOLS_ETF = {
    "TQQQ": "tqqq.us",
    "QLD":  "qld.us",
}

# 直近3か月(目安)の営業日数
LOOKBACK_BDAYS = 63

# シェルで表示する日中変動シナリオ（±%）
RANGE_SCENARIOS_PCT = [0.5, 1.0, 1.5, 2.0]

# signal閾値（Zスコア）
Z_OVERHEAT = 1.5
Z_CAUTION = 1.0
Z_DIP = -1.0
Z_STRONG_DIP = -1.5


# ====== SQLite ======
def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS prices (
        symbol TEXT NOT NULL,
        date   TEXT NOT NULL, -- YYYY-MM-DD
        open   REAL,
        high   REAL,
        low    REAL,
        close  REAL,
        volume REAL,
        PRIMARY KEY(symbol, date)
    )
    """)
    conn.commit()


def upsert_prices(conn: sqlite3.Connection, symbol: str, df: pd.DataFrame) -> int:
    rows = []
    for _, r in df.iterrows():
        rows.append((
            symbol,
            r["date"].strftime("%Y-%m-%d"),
            float(r["open"]) if pd.notna(r["open"]) else None,
            float(r["high"]) if pd.notna(r["high"]) else None,
            float(r["low"]) if pd.notna(r["low"]) else None,
            float(r["close"]) if pd.notna(r["close"]) else None,
            float(r["volume"]) if pd.notna(r["volume"]) else None,
        ))
    cur = conn.cursor()
    cur.executemany("""
        INSERT OR REPLACE INTO prices(symbol,date,open,high,low,close,volume)
        VALUES(?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    return cur.rowcount


def load_prices(conn: sqlite3.Connection, symbol: str, limit: int = 600) -> pd.DataFrame:
    df = pd.read_sql_query("""
        SELECT date, open, high, low, close, volume
        FROM prices
        WHERE symbol = ?
        ORDER BY date ASC
    """, conn, params=(symbol,))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df.tail(limit).reset_index(drop=True)


# ====== Download ======
def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    url = STOOQ_DAILY_CSV.format(symbol=symbol)
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.strip().lower() for c in df.columns]  # date, open, high, low, close, volume
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


# ====== Modeling ======
@dataclass
class ModelResult:
    name: str
    a: float
    b: float
    c: float
    sigma: float
    n: int
    median_range_pct: float  # NDX日中レンジの中央値（±%）
    latest_date: str
    latest_ndx_close: float
    latest_etf_close: float
    latest_etf_expected: float
    latest_z: float
    latest_signal: str


def signal_from_z(z: float) -> str:
    if z > Z_OVERHEAT:
        return "🔴過熱"
    if z > Z_CAUTION:
        return "🟠注意"
    if z < Z_STRONG_DIP:
        return "🔵強い押し目"
    if z < Z_DIP:
        return "🟢押し目"
    return "⚪中立"


def fit_model_on_log_price(
    ndx: pd.DataFrame,
    etf: pd.DataFrame,
    etf_name: str,
    lookback: int
) -> Tuple[ModelResult, pd.DataFrame]:
    """
    ln(ETF_close) = a + b*ln(NDX_close) + c*ln(NDX_high/NDX_low) + eps
    """
    if ndx.empty or etf.empty:
        raise RuntimeError("DBに価格データがありません。まず update を実行してください。")

    m = ndx.merge(
        etf[["date", "close"]].rename(columns={"close": "etf_close"}),
        on="date",
        how="inner"
    ).dropna(subset=["close", "high", "low", "etf_close"]).copy()

    if len(m) < 30:
        raise RuntimeError(f"{etf_name}: 共通日付が少なすぎます（{len(m)}行）。")

    m["ln_ndx_close"] = np.log(m["close"])
    m["ln_ndx_range"] = np.log(m["high"] / m["low"])
    m["ln_etf_close"] = np.log(m["etf_close"])

    fit = m.tail(lookback).dropna(subset=["ln_ndx_close", "ln_ndx_range", "ln_etf_close"]).copy()
    n = len(fit)
    if n < 30:
        raise RuntimeError(f"{etf_name}: 回帰に使える行が少なすぎます（{n}行）。")

    y = fit["ln_etf_close"].to_numpy()
    x1 = fit["ln_ndx_close"].to_numpy()
    x2 = fit["ln_ndx_range"].to_numpy()

    X = np.column_stack([np.ones(n), x1, x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = map(float, beta)

    resid = y - (X @ beta)
    sigma = float(np.std(resid, ddof=1))

    med_range_ln = float(np.median(fit["ln_ndx_range"].to_numpy()))
    med_pct = float(np.exp(med_range_ln / 2.0) - 1.0)  # ±%近似

    tail = m.tail(lookback).copy()
    tail["pred_ln_etf"] = a + b * np.log(tail["close"]) + c * np.log(tail["high"] / tail["low"])
    tail["pred_etf"] = np.exp(tail["pred_ln_etf"])
    tail["resid_ln"] = np.log(tail["etf_close"]) - tail["pred_ln_etf"]
    tail["z"] = tail["resid_ln"] / sigma

    last = tail.dropna(subset=["etf_close", "pred_etf", "z"]).tail(1).iloc[0]
    latest_date = last["date"].strftime("%Y-%m-%d")
    latest_ndx_close = float(last["close"])
    latest_etf_close = float(last["etf_close"])
    latest_etf_expected = float(last["pred_etf"])
    latest_z = float(last["z"])
    latest_signal = signal_from_z(latest_z)

    model = ModelResult(
        name=etf_name,
        a=a, b=b, c=c, sigma=sigma, n=n,
        median_range_pct=med_pct,
        latest_date=latest_date,
        latest_ndx_close=latest_ndx_close,
        latest_etf_close=latest_etf_close,
        latest_etf_expected=latest_etf_expected,
        latest_z=latest_z,
        latest_signal=latest_signal
    )
    return model, tail


def expected_etf_price(model: ModelResult, ndx_level: float, ndx_range_pct: float) -> Tuple[float, float, float, float, float]:
    """
    return expected, -1σ, +1σ, -1.5σ, +1.5σ（価格空間）
    """
    p = float(ndx_range_pct)
    if not (0.0 < p < 0.5):
        raise ValueError("ndx_range_pct は 0〜0.5 の範囲（例: 0.015=±1.5%）で指定してください。")

    range_ln = np.log((1.0 + p) / (1.0 - p))
    pred_ln = model.a + model.b * np.log(ndx_level) + model.c * range_ln

    exp_px = float(np.exp(pred_ln))
    m1 = float(np.exp(pred_ln - model.sigma))
    p1 = float(np.exp(pred_ln + model.sigma))
    m15 = float(np.exp(pred_ln - 1.5 * model.sigma))
    p15 = float(np.exp(pred_ln + 1.5 * model.sigma))
    return exp_px, m1, p1, m15, p15


# ====== Commands ======
@dataclass
class ShellState:
    models: Optional[Dict[str, ModelResult]] = None


SHELL_HELP_TEXT = """
Commands:
  update             価格データを更新
  ndx <value>        NDXを入力して想定レンジ表示（例: ndx 25653.9）
  latest             最新日のTrading Signalを再表示
  help               ヘルプ
  quit / exit / q    終了

Tips:
  ただ数値だけ入れた場合も NDX として扱います（例: 25653.9）
"""


def normalize_tokens(tokens: List[str]) -> List[str]:
    joined = " ".join(tokens)
    return joined.replace(",", " ").split()


def is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def parse_command_sequence(raw_tokens: List[str]) -> List[List[str]]:
    tokens = normalize_tokens(raw_tokens)
    commands: List[List[str]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i].lower()
        if token in {"q", "quit", "exit"}:
            commands.append(["quit"])
            i += 1
            continue
        if token in {"update", "shell", "latest", "help"}:
            commands.append([token])
            i += 1
            continue
        if token == "ndx":
            if i + 1 >= len(tokens):
                raise ValueError("ndx の値がありません。")
            commands.append(["ndx", tokens[i + 1]])
            i += 2
            continue
        if is_number(token):
            commands.append(["ndx", token])
            i += 1
            continue
        raise ValueError(f"不明なコマンド: {tokens[i]}")
    return commands


def parse_ndx_value(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError("入力例: ndx 25653.9  /  25653.9  / latest / update / quit") from exc
    if value <= 0:
        raise ValueError("入力例: ndx 25653.9  /  25653.9  / latest / update / quit")
    return value


def ensure_models(conn: sqlite3.Connection, state: ShellState) -> Dict[str, ModelResult]:
    if state.models is None:
        state.models = build_models(conn)
    return state.models


def cmd_update(conn: sqlite3.Connection) -> None:
    print("Downloading daily OHLC from Stooq ...")
    ndx = fetch_stooq_daily(SYMBOL_NDX)
    n_ndx = upsert_prices(conn, SYMBOL_NDX, ndx)

    print(f"  upsert {SYMBOL_NDX}: {n_ndx} rows")

    for name, sym in SYMBOLS_ETF.items():
        df = fetch_stooq_daily(sym)
        n = upsert_prices(conn, sym, df)
        print(f"  upsert {name}({sym}): {n} rows")


def build_models(conn: sqlite3.Connection) -> Dict[str, ModelResult]:
    ndx = load_prices(conn, SYMBOL_NDX)
    if ndx.empty:
        raise RuntimeError("NDXデータがありません。python ndx_etf.py update を先に実行してください。")

    models: Dict[str, ModelResult] = {}
    for name, sym in SYMBOLS_ETF.items():
        etf = load_prices(conn, sym)
        model, _ = fit_model_on_log_price(ndx, etf, name, LOOKBACK_BDAYS)
        models[name] = model
    return models


def print_latest(models: Dict[str, ModelResult]) -> None:
    # 共通日付でない可能性があるので、ETFごとに表示
    print("\n=== Latest Trading Signals (each ETF's latest common date with NDX) ===")
    for name, m in models.items():
        print(f"\n[{name}] date={m.latest_date}")
        print(f"  NDX close      = {m.latest_ndx_close:,.2f}")
        print(f"  {name} close    = {m.latest_etf_close:,.2f}")
        print(f"  {name} expected = {m.latest_etf_expected:,.2f}")
        print(f"  Z              = {m.latest_z:.2f}")
        print(f"  Signal         = {m.latest_signal}")


def print_mapping_for_ndx(models: Dict[str, ModelResult], ndx_level: float) -> None:
    print(f"\nNDX入力値: {ndx_level:,.2f}")
    for name, m in models.items():
        print(f"\n--- {name} 想定レンジ（NDX入力値ベース）---")
        print(f"model: n={m.n} sigma={m.sigma:.6f}  median_range≈±{m.median_range_pct*100:.2f}%")
        print(f"{'range':>8}  {'exp':>10}  {'-1σ':>10}  {'+1σ':>10}  {'-1.5σ':>10}  {'+1.5σ':>10}")
        for rpct in RANGE_SCENARIOS_PCT:
            exp_px, m1, p1, m15, p15 = expected_etf_price(m, ndx_level, rpct / 100.0)
            print(f"{('±'+str(rpct)+'%'):>8}  {exp_px:10.2f}  {m1:10.2f}  {p1:10.2f}  {m15:10.2f}  {p15:10.2f}")

        exp_px, m1, p1, *_ = expected_etf_price(m, ndx_level, m.median_range_pct)
        print(f"(参考) 中央値レンジ ±{m.median_range_pct*100:.2f}%  exp={exp_px:.2f}  -1σ={m1:.2f}  +1σ={p1:.2f}")


def run_command(conn: sqlite3.Connection, state: ShellState, cmd: str, args: List[str], *, interactive: bool) -> bool:
    if cmd == "quit":
        return False
    if cmd == "help":
        print(SHELL_HELP_TEXT.strip())
        return True
    if cmd == "update":
        cmd_update(conn)
        state.models = None
        return True
    if cmd == "latest":
        models = ensure_models(conn, state)
        print_latest(models)
        return True
    if cmd == "ndx":
        if not args:
            raise ValueError("ndx の値がありません。")
        ndx_level = parse_ndx_value(args[0])
        models = ensure_models(conn, state)
        print_mapping_for_ndx(models, ndx_level)
        return True
    if cmd == "shell":
        if interactive:
            print("すでにシェルです。")
            return True
        run_shell(conn, state)
        return True
    raise ValueError(f"不明なコマンド: {cmd}")


def run_shell(conn: sqlite3.Connection, state: ShellState) -> None:
    models = ensure_models(conn, state)
    print("\n=== Model Ready (last ~3 months) ===")
    for name, m in models.items():
        print(f"[{name}] n={m.n}  a={m.a:.6f}  b={m.b:.6f}  c={m.c:.6f}  sigma={m.sigma:.6f}")

    print_latest(models)
    print(SHELL_HELP_TEXT.strip())

    while True:
        try:
            s = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return
        if not s:
            continue
        try:
            commands = parse_command_sequence([s])
        except ValueError as exc:
            print(str(exc))
            continue

        for command in commands:
            try:
                should_continue = run_command(conn, state, command[0], command[1:], interactive=True)
            except ValueError as exc:
                print(str(exc))
                should_continue = True
            if not should_continue:
                return


def cmd_shell(conn: sqlite3.Connection) -> None:
    state = ShellState()
    run_shell(conn, state)


# ====== main ======
def main(argv: List[str]) -> int:
    try:
        commands = parse_command_sequence(argv[1:])
    except ValueError as exc:
        print(str(exc))
        return 1
    if not commands:
        commands = [["shell"]]

    with sqlite3.connect(DB_PATH) as conn:
        ensure_db(conn)

        state = ShellState()
        for command in commands:
            try:
                should_continue = run_command(conn, state, command[0], command[1:], interactive=False)
            except ValueError as exc:
                print(str(exc))
                return 1
            if not should_continue:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
