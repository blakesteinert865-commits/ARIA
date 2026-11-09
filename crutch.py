#!/usr/bin/env python3
"""
crutch.py - an offline "crutch" for small language models.

Idea: code decides first, the model second.

Every message goes through a router of small tools (timers, lists, schedule
maths, dates, unit conversion, calculator, fact tables). If a tool matches, it
computes the answer exactly and a fixed template words it - the model is never
involved, so it cannot hallucinate. Anything no tool matches goes to the model
(any OpenAI-compatible local server: LM Studio, llama-server, Ollama...).

Bundled Hassil 3.12.1 sentence parsing, with optional dependencies
for YAML loading/word-number ranges only. Works offline.

Usage
    python crutch.py                  chat loop (tools + model)
    python crutch.py "text"           one-shot
    python crutch.py --no-model       tools only, never call a model
    python crutch.py --verbose        show which tool answered
    python crutch.py --test           self-test of every tool (no model needed)
    python crutch.py --write-timetable  save the built-in school timetable as
                                      timetable.json so you can edit it
    python crutch.py --compare [file] raw model vs crutch, side by side
                                      (file: one prompt per line)

Environment variables
    CRUTCH_URL        model endpoint (default LM Studio:
                      http://localhost:1234/v1/chat/completions;
                      llama-server: http://localhost:8080/v1/chat/completions)
    CRUTCH_MODEL      model name sent to the server (default "local-model")
    CRUTCH_TEMP       temperature (default 0.2)
    CRUTCH_MAX_TOKENS reply cap for normal requests (default 120)
    CRUTCH_DETAIL_MAX_TOKENS reply cap for detailed requests (default 512)
    CRUTCH_MAX_SENTENCES normal response sentence cap (default 3)
    CRUTCH_NOTHINK=1  ask the server to disable "thinking" output
    CRUTCH_HOME       data folder (default ./data next to crutch.py): lists, notes,
                      facts.json (your own fact table), unmatched.log

Add your own tool: write a function  def tool_x(t): ...  that returns a reply
string, or None if the message isn't for it, then add it to TOOLS.
"""
from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

MODEL_URL = os.environ.get("CRUTCH_URL", "http://localhost:1234/v1/chat/completions")
MODEL_NAME = os.environ.get("CRUTCH_MODEL", "local-model")
TEMPERATURE = float(os.environ.get("CRUTCH_TEMP", "0.2"))
MAX_TOKENS = int(os.environ.get("CRUTCH_MAX_TOKENS", "120"))
DETAIL_MAX_TOKENS = int(os.environ.get("CRUTCH_DETAIL_MAX_TOKENS", "512"))
NO_THINK = os.environ.get("CRUTCH_NOTHINK", "0") == "1"
MAX_SENTENCES = int(os.environ.get("CRUTCH_MAX_SENTENCES", "3"))

SYSTEM_PROMPT = (
    "You are Aria, a voice assistant on smart glasses. Answer ONLY the last "
    "user message. Earlier turns are background, not extra requests. Reply in one or two short "
    "sentences of plain text. No lists, no symbols, no formatting. If you are "
    "not sure, say you don't know. Never guess numbers, times or dates."
)

DETAILED_SYSTEM_PROMPT = (
    "You are Aria, a local voice assistant. Answer ONLY the last user "
    "message; previous turns provide context, not new requests. The user specifically requested a "
    "detailed explanation for this message. Give a clear, informative, longer "
    "answer with enough sentences to fully explain the topic, typically 6 to 12 "
    "when appropriate. Include important steps or examples where useful. Use "
    "plain English and avoid unnecessary filler. If you do not know something, "
    "say so. Never make up facts, numbers, times, or dates."
)

# Response length is request-scoped; never infer it from older conversation
# messages bundled into the model's context by the GUI.
_DETAIL_REQUEST_RE = re.compile(
    r"\b(?:"
    r"in[\s-]+detail|in[\s-]+depth|in[\s-]+full|"
    r"(?:really|very|more|extra|highly)?[\s-]*detailed(?:ly)?|"
    r"thorough(?:ly)?|comprehensive(?:ly)?|"
    r"step[\s-]+by[\s-]+step|"
    r"(?:more|extra|greater|lots? of|plenty of)[\s-]+detail|"
    r"(?:a[\s-]+)?(?:full|long(?:er)?|complete)[\s-]+(?:answer|explanation|response|breakdown)|"
    r"(?:explain|describe|cover)[\s-]+(?:it[\s-]+)?fully|"
    r"(?:elaborate|expand)[\s-]+on|"
    r"(?:walk|talk)[\s-]+me[\s-]+through|"
    r"break[\s-]+(?:it|this|that)[\s-]+down|"
    r"(?:go|get)[\s-]+into[\s-]+(?:the[\s-]+)?(?:details?|specifics)|"
    r"(?:explain|describe)[\s-]+(?:it[\s-]+)?(?:more|everything)|"
    r"(?:more[\s-]+(?:information|explanation|depth))"
    r")\b", re.I
)


def wants_detailed_reply(user_text: str) -> bool:
    """Whether THIS user message requests a longer model response."""
    t = re.sub(r"\s+", " ", user_text.lower().replace("\u2019", "'")).strip()
    for match in _DETAIL_REQUEST_RE.finditer(t):
        # Negations should not turn short-answer requests into detailed ones.
        before = t[max(0, match.start() - 30):match.start()]
        if re.search(r"(?:don't|do not|dont|no need to|without|not|avoid)\s+(?:(?:be|go|explain|describe|answer|reply|write|a|an|any)\s*){0,2}$", before):
            continue
        return True
    return False

NOW_OVERRIDE = None  # tests set this to a fixed datetime


def now() -> datetime:
    return NOW_OVERRIDE or datetime.now()


def home_dir() -> Path:
    p = Path(os.environ.get("CRUTCH_HOME", str(Path(__file__).resolve().parent / "data")))
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------

def fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}" if abs(x) >= 10000 else str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".")


def plural(n: float, word: str) -> str:
    return f"{fmt(n)} {word}" + ("" if abs(n - 1) < 1e-9 else "s")


def fmt_time(minutes: float) -> str:
    m = int(round(minutes)) % 1440
    h, mi = divmod(m, 60)
    return f"{(h % 12) or 12}:{mi:02d} {'AM' if h < 12 else 'PM'}"


def fmt_dur(minutes: float) -> str:
    total = int(round(minutes))
    if total == 0:
        return "0 minutes"
    h, m = divmod(abs(total), 60)
    parts = []
    if h:
        parts.append(plural(h, "hour"))
    if m:
        parts.append(plural(m, "minute"))
    return " ".join(parts)


def fmt_secs(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    parts = []
    if h:
        parts.append(plural(h, "hour"))
    if m:
        parts.append(plural(m, "minute"))
    if sec or not parts:
        parts.append(plural(sec, "second"))
    return " ".join(parts)


def fmt_date(d: date) -> str:
    return f"{d:%A}, {d:%B} {d.day}, {d.year}"


UNIT_G = r"(hours?|hrs?|h|minutes?|mins?)(?![a-z])"


def to_minutes(n: str, unit: str) -> float:
    return float(n) * (60 if unit.lower().startswith("h") else 1)


# ----------------------------------------------------------------------------
# Time parsing
# ----------------------------------------------------------------------------

_TIME_RE = re.compile(
    r"(?<![\d:.])(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?![a-z\d])", re.I
)


def normalize(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\bnoon\b", "12:00 pm", t)
    t = re.sub(r"\bmidnight\b", "12:00 am", t)
    return t


def find_times(t: str):
    """Clock times in text. Needs a colon or am/pm so plain numbers are ignored."""
    out = []
    for m in _TIME_RE.finditer(t):
        h, mi, mer = int(m.group(1)), m.group(2), m.group(3)
        if mi is None and mer is None:
            continue
        mi = int(mi) if mi else 0
        if mi > 59:
            continue
        if mer:
            if not 1 <= h <= 12:
                continue
            minutes = (h % 12) * 60 + mi + (720 if mer.lower().startswith("p") else 0)
            amb = False
        else:
            if h > 23:
                continue
            minutes = h * 60 + mi
            amb = 1 <= h <= 12
        out.append({"start": m.start(), "end": m.end(), "min": minutes, "amb": amb})
    return out


# ----------------------------------------------------------------------------
# Tool: schedule helper (travel time, latest time to leave, arrival time)
# ----------------------------------------------------------------------------

_SPEED_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(km/h|kph|kmh|mph|mi/h|km per hour|kilometers per hour|"
    r"kilometres per hour|miles per hour)", re.I)
_DIST_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[- ]?\s*(kms?|kilometers?|kilometres?|miles?|mi)\b", re.I)
_TRAVEL_WORDS = re.compile(
    r"\b(trips?|travel|travelling|traveling|drives?|driving|commutes?|journeys?|walk|"
    r"walking|rides?|km|kms|kilomet\w+|miles?|mph|kph)\b")
_NOT_EVENT = {"arrive", "be", "there", "leave", "start", "begin", "am", "is", "are",
              "get", "here", "go", "be", "it", "that", "home", "back", "meet", "at"}
MILE = 1.609344


def _speed_kmh(t: str):
    m = _SPEED_RE.search(t)
    if not m:
        return None
    v = float(m.group(1))
    return v * MILE if m.group(2).lower() in ("mph", "mi/h", "miles per hour") else v


def _dist_km(t: str):
    m = _DIST_RE.search(_SPEED_RE.sub(" ", t))
    if not m:
        return None
    v = float(m.group(1))
    return v * MILE if m.group(2).lower().startswith(("mi",)) else v


def _travel_minutes(t: str):
    p = re.search(
        r"(?:takes?|trip|travel(?:l?ing)?(?: time)?|drive|driving|journey|commute|"
        r"walk(?:ing)?|ride)[^0-9.?!]{0,25}?(\d+(?:\.\d+)?)\s*" + UNIT_G,
        _SPEED_RE.sub(" ", t))
    if p:
        return to_minutes(p.group(1), p.group(2))
    p = re.search(r"(\d+(?:\.\d+)?)[- ]?" + UNIT_G + r"[- ]?(?:trip|drive|journey|commute|walk|ride)", t)
    if p:
        return to_minutes(p.group(1), p.group(2))
    d, s = _dist_km(t), _speed_kmh(t)
    if d is not None and s:
        return d / s * 60
    return None


def tool_schedule(text: str):
    t = normalize(text)
    if not _TRAVEL_WORDS.search(t):
        return None
    times = find_times(t)

    # ---- Plan mode: several events + a home deadline --------------------
    events, home = [], None
    for tm in times:
        prefix = t[max(0, tm["start"] - 45):tm["start"]]
        if re.search(r"\b(home|back)\b.*\b(by|before|at)\s*$", prefix):
            home = tm
            continue
        m = re.search(r"([a-z]+)\s+(?:at|@)\s*$", prefix)
        if m and m.group(1) not in _NOT_EVENT:
            dur = re.match(r"\s*,?\s*(?:for|lasting|lasts)\s+(\d+(?:\.\d+)?)\s*" + UNIT_G, t[tm["end"]:])
            events.append({"name": m.group(1), "t": tm,
                           "dur": to_minutes(dur.group(1), dur.group(2)) if dur else None})

    if len(events) >= 2:
        if any(e["t"]["amb"] for e in events) or (home and home["amb"]):
            return "Please include AM or PM with each time."
        default = re.search(r"(?:each|every)\b[^.?!]*?(\d+(?:\.\d+)?)\s*" + UNIT_G,
                            _SPEED_RE.sub(" ", t))
        default = to_minutes(default.group(1), default.group(2)) if default else None
        for e in events:
            if e["dur"] is None:
                e["dur"] = default
        if any(e["dur"] is None for e in events):
            return "How long does each event last?"
        travel = _travel_minutes(t)
        if travel is None:
            return "I need the trip length: a travel time, or a distance and a speed."
        events.sort(key=lambda e: e["t"]["min"])
        out = [f"Each trip takes {fmt_dur(travel)}."]
        for i, e in enumerate(events):
            end = e["t"]["min"] + e["dur"]
            if i + 1 < len(events):
                deadline = events[i + 1]["t"]["min"]
            elif home:
                deadline = home["min"]
            else:
                continue
            latest = deadline - travel
            slack = latest - end
            if slack < -0.5:
                out.append(f"The {e['name']} ends at {fmt_time(end)}, so you cannot "
                           f"make the deadline; you would need to leave {fmt_dur(-slack)} "
                           f"before it ends.")
            else:
                tail = " That is right when it ends." if abs(slack) < 0.5 else ""
                out.append(f"Leave the {e['name']} by {fmt_time(latest)}.{tail}")
        return " ".join(out)

    # ---- Single trip ----------------------------------------------------
    travel = _travel_minutes(t)
    if travel is None:
        if times and re.search(r"\b(leave|depart|arrive)\b", t):
            return "I need the trip length: a travel time, or a distance and a speed."
        return None
    if not times:
        return f"That trip takes {fmt_dur(travel)}."
    tm = times[0]
    if tm["amb"]:
        return "Please include AM or PM with the time."
    prefix = t[max(0, tm["start"] - 30):tm["start"]]
    if re.search(r"\b(leave|leaving|depart|set off|head out|start|go|leaves)\b[^.?!]{0,15}?\b(at|by|around|after)\s*$", prefix):
        return f"If you leave at {fmt_time(tm['min'])}, you arrive at {fmt_time(tm['min'] + travel)}."
    return (f"The trip takes {fmt_dur(travel)}, so leave by "
            f"{fmt_time(tm['min'] - travel)} to arrive by {fmt_time(tm['min'])}.")


# ----------------------------------------------------------------------------
# Tool: dates and times
# ----------------------------------------------------------------------------

_MONTHS = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
           "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
           "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9, "october": 10,
           "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAYS = {"monday": 0, "mon": 0, "tuesday": 1, "tues": 1, "tue": 1, "wednesday": 2,
             "wed": 2, "thursday": 3, "thurs": 3, "thur": 3, "thu": 3, "friday": 4,
             "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6}
_WEEKDAY_RE = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))


def find_date(t: str):
    t = t.lower()
    today = now().date()
    try:
        m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
        if m:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = (re.search(r"\b(" + _MONTH_RE + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b", t))
        if m:
            mo, d, y = _MONTHS[m.group(1)], int(m.group(2)), m.group(3)
        else:
            m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + _MONTH_RE + r")\.?(?:,?\s+(\d{4}))?\b", t)
            mo, d, y = (_MONTHS[m.group(2)], int(m.group(1)), m.group(3)) if m else (None, None, None)
        if mo:
            if y:
                return date(int(y), mo, d)
            cand = date(today.year, mo, d)
            return cand if cand >= today else date(today.year + 1, mo, d)
    except ValueError:
        return None
    if re.search(r"\btoday\b", t):
        return today
    if re.search(r"\btomorrow\b", t):
        return today + timedelta(days=1)
    if re.search(r"\byesterday\b", t):
        return today - timedelta(days=1)
    m = re.search(r"\b(next|this|on)?\s*(" + _WEEKDAY_RE + r")\b", t)
    if m:
        ahead = (_WEEKDAYS[m.group(2)] - today.weekday()) % 7
        if m.group(1) == "next" and ahead == 0:
            ahead = 7
        return today + timedelta(days=ahead)
    return None


_CITY_TZ = {
    "new york": "America/New_York", "los angeles": "America/Los_Angeles",
    "chicago": "America/Chicago", "toronto": "America/Toronto",
    "london": "Europe/London", "paris": "Europe/Paris", "berlin": "Europe/Berlin",
    "madrid": "Europe/Madrid", "rome": "Europe/Rome", "moscow": "Europe/Moscow",
    "dubai": "Asia/Dubai", "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata",
    "singapore": "Asia/Singapore", "hong kong": "Asia/Hong_Kong",
    "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai", "tokyo": "Asia/Tokyo",
    "seoul": "Asia/Seoul", "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne", "adelaide": "Australia/Adelaide",
    "perth": "Australia/Perth", "brisbane": "Australia/Brisbane",
    "auckland": "Pacific/Auckland", "cairo": "Africa/Cairo",
    "johannesburg": "Africa/Johannesburg", "sao paulo": "America/Sao_Paulo",
}


def tool_datetime(text: str):
    t = normalize(text).strip().rstrip("?!. ")
    t = re.sub(r"^(?:tell me|could you tell me|do you know) (?:what )?(?:the )?(?:current|local) time is$", "what time is it", t)
    t = re.sub(r"^(?:can i get|give me|tell me) (?:the )?date(?: for today)?$", "what is the date", t)
    t = re.sub(r"^(?:what is|what's) (?:the )?time (?:right )?now$", "what time is it", t)
    t = re.sub(r"^(?:what day|what date) will (?:it|today) be tomorrow$", "what is the date tomorrow", t)
    t = re.sub(r"^(?:how much (?:longer|time)|how many minutes) (?:is it )?(?:until|till) ", "how long until ", t)
    n = now()
    today = n.date()
    times = find_times(t)

    # --- time in another city
    m = re.search(r"\btime (?:is it )?in ([a-z ]+?)(?: right now| now| currently)?$", t)
    if m and m.group(1).strip() in _CITY_TZ:
        try:
            from zoneinfo import ZoneInfo
            local = datetime.now(ZoneInfo(_CITY_TZ[m.group(1).strip()]))
            return f"It is {fmt_time(local.hour * 60 + local.minute)} in {m.group(1).title()}."
        except Exception:
            return "I don't have time zone data on this device."

    # --- date of today / tomorrow / yesterday, and time + date together
    mm = (re.fullmatch(r"(?:what(?:'s| is)?|whats|tell me)\s+(?:the )?(?:date|day)(?: is it| is| will it be)?\s*(?:for |on )?(today|tomorrow|yesterday)", t)
          or re.fullmatch(r"(?:what(?:'s| is)?|whats|tell me)\s+(today|tomorrow|yesterday)'?s?\s+(?:date|day)", t)
          or re.fullmatch(r"what (?:date|day) (?:is it|is|will it be|was it|was)\s*(today|tomorrow|yesterday)", t))
    if mm:
        word = mm.group(1)
        d0 = today + timedelta(days={"today": 0, "tomorrow": 1, "yesterday": -1}[word])
        return f"{word.title()} is {fmt_date(d0)}."
    if re.fullmatch(r"(?:what(?:'s| is)|whats)(?: the)? (?:time|date) (?:and|&|\+) (?:the )?(?:time|date)(?: now| today| right now)?", t):
        return f"It is {fmt_time(n.hour * 60 + n.minute)} on {fmt_date(today)}."

    # --- now
    if re.fullmatch(r"(?:what(?:'s| is)? the time(?: now| right now)?|what time is it(?: now| right now)?|"
                    r"current time|time(?: now| please)?|got the time|tell me the time)", t):
        return f"It is {fmt_time(n.hour * 60 + n.minute)}."
    if re.fullmatch(r"(?:what(?:'s| is)? (?:the )?(?:date|day)(?: today)?|today'?s date|"
                    r"what day is it(?: today)?|what day of the week is it|what is today)", t):
        return f"Today is {fmt_date(today)}."

    # --- what time will it be in N minutes/hours
    m = re.search(r"\b(?:what time|time)\b.*\b(?:in|after)\s+(\d+(?:\.\d+)?)\s*" + UNIT_G, t)
    if m and not times:
        target = n + timedelta(minutes=to_minutes(m.group(1), m.group(2)))
        return f"It will be {fmt_time(target.hour * 60 + target.minute)}."

    # --- how long until a clock time
    m = re.search(r"\bhow (?:long|much time|many (?:hours|minutes))\b.*\b(?:until|till|to|before)\b", t)
    if m and len(times) == 1 and not times[0]["amb"] and not find_date(t.replace("today", "")):
        diff = (times[0]["min"] - (n.hour * 60 + n.minute)) % 1440
        return f"{fmt_dur(diff)} until {fmt_time(times[0]['min'])}."

    # --- time arithmetic on an explicit time
    if len(times) == 1:
        tm = times[0]
        arith = re.search(r"(plus|\+|add|minus|-|subtract)\s*(\d+(?:\.\d+)?)\s*" + UNIT_G, t)
        rel = re.search(r"(\d+(?:\.\d+)?)\s*" + UNIT_G + r"\s*(after|before|later than|earlier than|from)\b", t)
        if arith or rel:
            if tm["amb"]:
                return "Please include AM or PM with the time."
            if arith:
                delta = to_minutes(arith.group(2), arith.group(3))
                sign = -1 if arith.group(1) in ("minus", "-", "subtract") else 1
            else:
                delta = to_minutes(rel.group(1), rel.group(2))
                sign = -1 if rel.group(3) in ("before", "earlier than") else 1
            return f"That is {fmt_time(tm['min'] + sign * delta)}."

    # --- difference between two clock times
    if len(times) == 2 and re.search(r"\b(from|between)\b", t) and \
            re.search(r"\b(how long|how many|difference|how much time)\b", t):
        if times[0]["amb"] or times[1]["amb"]:
            return "Please include AM or PM with each time."
        diff = (times[1]["min"] - times[0]["min"]) % 1440
        return f"That is {fmt_dur(diff)}."

    # --- N days/weeks from now / ago
    m = re.search(r"(?:what (?:date|day)|date).*?\b(?:in\s+(\d+)\s*(days?|weeks?)|(\d+)\s*(days?|weeks?)\s*(from now|from today|ago))", t)
    if m:
        n_, unit, when = (m.group(1), m.group(2), None) if m.group(1) else (m.group(3), m.group(4), m.group(5))
        days = int(n_) * (7 if unit.startswith("w") else 1)
        d = today - timedelta(days=days) if when == "ago" else today + timedelta(days=days)
        return f"That is {fmt_date(d)}."

    d = find_date(t)
    # --- how many days until / since a date
    if d and re.search(r"\b(how (?:many|long)|days?|weeks?)\b", t) and \
            re.search(r"\b(until|till|to|before|since|after|from)\b", t) and not times:
        diff = (d - today).days
        if diff == 0:
            return "That is today."
        days = abs(diff)
        extra = ""
        if days >= 14:
            w, r = divmod(days, 7)
            extra = f", which is {plural(w, 'week')}" + (f" and {plural(r, 'day')}" if r else "")
        if diff > 0:
            return f"{plural(days, 'day')} until {fmt_date(d)}{extra}."
        return f"{plural(days, 'day')} since {fmt_date(d)}{extra}."

    # --- day of week for a date
    if d and re.search(r"\b(what|which) day\b|\bday of the week\b", t):
        verb = "is" if d >= today else "was"
        return f"{d:%B} {d.day}, {d.year} {verb} a {d:%A}."
    return None


# ----------------------------------------------------------------------------
# Tool: unit conversion
# ----------------------------------------------------------------------------

_UNITS = {
    "length": {"mm": (0.001, "millimeters", ["mm", "millimeter", "millimeters", "millimetre", "millimetres"]),
               "cm": (0.01, "centimeters", ["cm", "centimeter", "centimeters", "centimetre", "centimetres"]),
               "m": (1, "meters", ["m", "meter", "meters", "metre", "metres"]),
               "km": (1000, "kilometers", ["km", "kms", "kilometer", "kilometers", "kilometre", "kilometres"]),
               "in": (0.0254, "inches", ["in", "inch", "inches"]),
               "ft": (0.3048, "feet", ["ft", "foot", "feet"]),
               "yd": (0.9144, "yards", ["yd", "yds", "yard", "yards"]),
               "mi": (1609.344, "miles", ["mi", "mile", "miles"])},
    "mass": {"mg": (0.001, "milligrams", ["mg", "milligram", "milligrams"]),
             "g": (1, "grams", ["g", "gram", "grams"]),
             "kg": (1000, "kilograms", ["kg", "kgs", "kilo", "kilos", "kilogram", "kilograms"]),
             "oz": (28.349523125, "ounces", ["oz", "ounce", "ounces"]),
             "lb": (453.59237, "pounds", ["lb", "lbs", "pound", "pounds"])},
    "volume": {"ml": (1, "milliliters", ["ml", "milliliter", "milliliters", "millilitre", "millilitres"]),
               "l": (1000, "liters", ["l", "liter", "liters", "litre", "litres"]),
               "tsp": (4.92892159375, "teaspoons", ["tsp", "teaspoon", "teaspoons"]),
               "tbsp": (14.78676478125, "tablespoons", ["tbsp", "tablespoon", "tablespoons"]),
               "floz": (29.5735295625, "fluid ounces", ["fl oz", "floz", "fluid ounce", "fluid ounces"]),
               "cup": (236.5882365, "cups", ["cup", "cups"]),
               "pint": (473.176473, "pints", ["pint", "pints"]),
               "quart": (946.352946, "quarts", ["quart", "quarts"]),
               "gal": (3785.411784, "gallons", ["gal", "gallon", "gallons"])},
    "speed": {"kmh": (1 / 3.6, "kilometers per hour", ["km/h", "kmh", "kph", "kilometers per hour", "kilometres per hour"]),
              "mph": (0.44704, "miles per hour", ["mph", "mi/h", "miles per hour"]),
              "ms": (1, "meters per second", ["m/s", "mps", "meters per second", "metres per second"]),
              "knot": (0.514444, "knots", ["knot", "knots", "kt", "kts"])},
    "time": {
        "sec": (1, "seconds", ["second", "seconds", "sec", "secs"]),
        "min": (60, "minutes", ["minute", "minutes", "min", "mins"]),
        "hr": (3600, "hours", ["hour", "hours", "hr", "hrs"]),
        "day": (86400, "days", ["day", "days"]),
        "week": (604800, "weeks", ["week", "weeks"]),
    },
    "area": {
        "sqcm": (0.0001, "square centimeters", ["square centimeters", "square centimetres", "sq cm", "cm2"]),
        "sqm": (1, "square meters", ["square meters", "square metres", "sq m", "m2"]),
        "sqkm": (1000000, "square kilometers", ["square kilometers", "square kilometres", "sq km", "km2"]),
        "sqft": (0.09290304, "square feet", ["square feet", "sq ft", "ft2"]),
        "acre": (4046.8564224, "acres", ["acre", "acres"]),
        "ha": (10000, "hectares", ["hectare", "hectares", "ha"]),
    },
    "temp": {"c": (None, "degrees Celsius", ["c", "celsius", "centigrade"]),
             "f": (None, "degrees Fahrenheit", ["f", "fahrenheit"]),
             "k": (None, "kelvin", ["k", "kelvin"])},
}
_ALIAS = {}
for _cat, _defs in _UNITS.items():
    for _key, (_factor, _name, _aliases) in _defs.items():
        for _a in _aliases:
            _ALIAS[_a] = (_cat, _key)
_US_NAMED = {"floz", "cup", "pint", "quart", "gal"}


def _lookup_unit(s: str):
    s = s.strip().lower()
    s = re.sub(r"^(?:a|an|one|the)\s+", "", s)
    return _ALIAS.get(s) or _ALIAS.get(s.rstrip("s")) if s else None


def _convert(v: float, a, b):
    (cat, ka), (_, kb) = a, b
    if cat == "temp":
        c = v if ka == "c" else (v - 32) * 5 / 9 if ka == "f" else v - 273.15
        return c if kb == "c" else c * 9 / 5 + 32 if kb == "f" else c + 273.15
    return v * _UNITS[cat][ka][0] / _UNITS[cat][kb][0]


def tool_units(text: str):
    t = text.lower().strip().rstrip("?!. ")
    t = re.sub(r"[°º]|degrees?\b", "", t)
    t = re.sub(r"^(?:please |hey )?(?:convert|change|turn|what is|what's|whats|how much is|how many is)\s+", "", t)
    t = re.sub(r"^how many ([a-z/ ]+?) (?:do you get |would that be )?in (\d+(?:\.\d+)?) ([a-z/ ]+)$", r"how many \1 in \2 \3", t)
    t = re.sub(r"^([\d.]+) ([a-z/ ]+) (?:equals|equal to|is how many|in terms of) ([a-z/ ]+)$", r"\1 \2 to \3", t)
    num = r"(-?\d+(?:\.\d+)?)"
    m = re.match(r"how (?:many|much) ([a-z0-9/ ]+?) (?:are |is )?(?:there )?in\s+(?:" + num + r"\s*)?([a-z0-9/ ]+)$", t)
    if m:
        to_u, v, from_u = m.group(1), float(m.group(2)) if m.group(2) else 1.0, m.group(3)
    else:
        m = re.match(num + r"\s*([a-z0-9/ ]+?)\s+(?:to|in|into|as)\s+(?:a |an )?([a-z0-9/ ]+)$", t)
        if not m:
            return None
        v, from_u, to_u = float(m.group(1)), m.group(2), m.group(3)
    a, b = _lookup_unit(from_u), _lookup_unit(to_u)
    if not a or not b:
        return None
    # "ounces" next to a volume unit means fluid ounces
    if a[0] != b[0] and a[1] == "oz" and b[0] == "volume":
        a = ("volume", "floz")
    elif a[0] != b[0] and b[1] == "oz" and a[0] == "volume":
        b = ("volume", "floz")
    if a[0] != b[0]:
        return f"I can't convert {_UNITS[a[0]][a[1]][1]} to {_UNITS[b[0]][b[1]][1]}."
    res = _convert(v, a, b)
    from_name, to_name = _UNITS[a[0]][a[1]][1], _UNITS[b[0]][b[1]][1]
    if a[1] in _US_NAMED:
        from_name = "US " + from_name
    if b[1] in _US_NAMED:
        to_name = "US " + to_name
    if abs(v - 1) < 1e-9:
        from_name = {"inches": "inch", "feet": "foot", "miles": "mile", "pounds": "pound",
                     "ounces": "ounce", "cups": "cup", "knots": "knot", "US cups": "US cup"}.get(from_name, from_name.rstrip("s") if from_name.endswith("s") and "per" not in from_name and from_name not in ("kelvin",) else from_name)
    return f"{fmt(v)} {from_name} is {fmt(res)} {to_name}."


# ----------------------------------------------------------------------------
# Tool: percentages, splitting bills, calculator, running totals
# ----------------------------------------------------------------------------

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 1000:
                raise ValueError("exponent too large")
            if isinstance(n.op, ast.Pow) and abs(a) > 10 ** 100:
                raise ValueError("base too large")
            result = _BIN[type(n.op)](a, b)
            if not isinstance(result, (int, float)) or abs(result) > 1e100:
                raise ValueError("result too large")
            return result
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        raise ValueError("unsupported")
    return ev(ast.parse(expr, mode="eval"))


def tool_percent(text: str):
    t = text.lower().strip().rstrip("?!. ").replace(",", "")
    n = r"\$?(\d+(?:\.\d+)?)"
    m = re.fullmatch(r"(?:what(?:'s| is)? |calculate |work out )?(?:the )?(?:percentage|percent) (?:increase|decrease|change) from " + n + r" to " + n, t)
    if m:
        a, b = map(float, m.groups())
        if not a:
            return "A percentage change from zero is undefined."
        return f"That is a {fmt(abs((b-a)/a*100))} percent {'increase' if b >= a else 'decrease'}."
    m = re.fullmatch(r"(?:increase|decrease|raise|reduce) " + n + r" by " + n + r"\s*(?:%|percent)", t)
    if m:
        a, pct = map(float, m.groups())
        factor = -1 if re.match(r"^(?:decrease|reduce)", t) else 1
        return f"The result is {fmt(a * (1 + factor * pct / 100))}."
    m = re.fullmatch(r"(?:what(?:'s| is)? |how much is )?" + n + r"\s*(?:%|percent) (?:of|from) " + n, t)
    if m:
        pct, base = map(float, m.groups())
        return f"{fmt(pct)} percent of {fmt(base)} is {fmt(pct*base/100)}."
    m = re.search(n + r"\s*(?:%|percent)\s*off\s*(?:of\s*)?" + n, t)
    if m:
        pct, price = float(m.group(1)), float(m.group(2))
        return f"{fmt(price * (1 - pct / 100))} after the discount. You save {fmt(price * pct / 100)}."
    m = re.search(n + r"\s*(?:%|percent)\s*(tip\s+)?(?:of|on)\s*" + n, t)
    if m:
        pct, base = float(m.group(1)), float(m.group(3))
        part = base * pct / 100
        if m.group(2):
            return f"A {fmt(pct)} percent tip on {fmt(base)} is {fmt(part)}, for a total of {fmt(base + part)}."
        return f"{fmt(pct)} percent of {fmt(base)} is {fmt(part)}."
    m = re.search(r"what\s*(?:'s|is)?\s*(?:the\s+)?percent(?:age)?\s+(?:is\s+)?" + n + r"\s+of\s+" + n, t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return f"{fmt(a)} is {fmt(a / b * 100)} percent of {fmt(b)}." if b else None
    m = re.search(r"split\s+" + n + r"\s+(?:between|among|by)\s+(\d+)(?:\s+(?:people|ways))?", t) or \
        re.search(n + r"\s+split\s+(\d+)\s+ways", t)
    if m:
        total, k = float(m.group(1)), int(m.group(2))
        return f"Each person pays {fmt(total / k)}." if k else None
    return None


def tool_calc(text: str):
    t = text.lower().strip().rstrip("?!. ")
    if re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", t):
        return None
    lead = re.match(r"^(?:(?:please|hey|ok|so)\s+)*(?:what is|what's|whats|calculate|compute|work out|"
                    r"how much is|solve|tell me)\s+", t)
    body = t[lead.end():] if lead else t
    m = re.fullmatch(r"(?:the\s+)?square root of (\d+(?:\.\d+)?)", body)
    if m:
        return f"The square root of {m.group(1)} is {fmt(float(m.group(1)) ** 0.5)}."
    # Verbal arithmetic forms that don't use an infix operator.
    verbal = [
        (r"(?:add|sum) (-?\d+(?:\.\d+)?) (?:and|to) (-?\d+(?:\.\d+)?)", lambda a,b: a+b),
        (r"subtract (-?\d+(?:\.\d+)?) from (-?\d+(?:\.\d+)?)", lambda a,b: b-a),
        (r"multiply (-?\d+(?:\.\d+)?) by (-?\d+(?:\.\d+)?)", lambda a,b: a*b),
        (r"divide (-?\d+(?:\.\d+)?) by (-?\d+(?:\.\d+)?)", lambda a,b: a/b),
    ]
    for pattern, operation in verbal:
        match = re.fullmatch(pattern, body)
        if match:
            try:
                return f"The answer is {fmt(operation(*map(float, match.groups())))}."
            except ZeroDivisionError:
                return "You cannot divide by zero."
    body = body.replace(",", "")
    for a, b in ((" to the power of ", "**"), (" squared", "**2"), (" cubed", "**3"),
                 (" multiplied by ", "*"), (" times ", "*"), (" divided by ", "/"),
                 (" over ", "/"), (" plus ", "+"), (" minus ", "-"), (" mod ", "%"),
                 ("×", "*"), ("÷", "/"), ("^", "**")):
        body = body.replace(a, b)
    body = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", body)
    if not re.fullmatch(r"[\d\s+\-*/().%]+", body) or not re.search(r"\d", body):
        return None
    has_op = re.search(r"[+*/%^]", body) or re.search(r"\s-\s", body) or re.search(r"\d\s*-\s*\d", body) and lead
    if not (lead or has_op):
        return None
    if not re.search(r"[+\-*/%]", body):
        return None
    if not lead and re.fullmatch(r"\s*\d+-\d+\s*", body):
        return None  # looks like a phone number or range
    try:
        return f"The answer is {fmt(safe_eval(body.strip()))}."
    except Exception:
        return None


_SUB_VERBS = r"(?:give|gave|giving|give away|gave away|eat|ate|eaten|lose|lost|sell|sold|spend|spent|use|used|drop|dropped|throw|threw|donate|donated|break|broke|remove|removed|take|took)"
_ADD_VERBS = r"(?:buy|bought|get|got|find|found|receive|received|pick up|picked up|win|won|earn|earned|collect|collected|add|added|make|made)"


def tool_running_total(text: str):
    t = text.lower().replace(",", " ")
    if len(re.findall(r"how many", t)) != 1:
        return None
    start = re.search(r"\b(?:have|has|start(?:ed)? with|own|got)\s+(\d+(?:\.\d+)?)\b", t)
    if not start or not re.search(r"how many[^.?!]*\b(left|have|remain\w*|now|do i have|are left)\b", t):
        return None
    total = float(start.group(1))
    steps = 0
    for m in re.finditer(r"\b(" + _SUB_VERBS + r"|" + _ADD_VERBS + r")\b(?: away| up| out)?(?: \w+){0,2}? (\d+(?:\.\d+)?)\b", t[start.end():]):
        verb = m.group(1)
        v = float(m.group(2))
        if re.fullmatch(_SUB_VERBS, verb):
            total -= v
        else:
            total += v
        steps += 1
    if steps == 0:
        return None
    return f"You have {fmt(total)} left." if total >= 0 else None


# ----------------------------------------------------------------------------
# Tool: timers, reminders and stopwatch
# ----------------------------------------------------------------------------

_TIMERS: dict = {}
_TIMER_SEQ = [0]
_STOPWATCH = {"start": None}
_lock = threading.Lock()


def notify(msg: str):
    print(f"\n*** {msg} ***\a", flush=True)
    if os.environ.get("CRUTCH_QUIET") == "1":
        return
    if shutil.which("termux-tts-speak"):
        subprocess.Popen(["termux-tts-speak", msg])
    if shutil.which("termux-vibrate"):
        subprocess.Popen(["termux-vibrate", "-d", "400"])


def _parse_seconds(t: str):
    total = 0.0
    found = False
    if re.search(r"\bhalf an hour\b", t):
        total, found = total + 1800, True
    for m in re.finditer(r"(\d+(?:\.\d+)?|an?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b", t):
        n = 1.0 if m.group(1) in ("a", "an") else float(m.group(1))
        u = m.group(2)[0]
        total += n * {"h": 3600, "m": 60, "s": 1}[u]
        found = True
    return total if found else None


def _fire(tid):
    with _lock:
        info = _TIMERS.pop(tid, None)
    if info:
        notify(f"Reminder: {info['label']}" if info["label"] else "Timer done.")


def tool_timer(text: str):
    t = text.lower().strip().rstrip("?!. ")
    if re.search(r"\bstopwatch\b", t):
        if re.search(r"\b(start|begin|reset)\b", t):
            _STOPWATCH["start"] = time.monotonic()
            return "Stopwatch started."
        if re.search(r"\b(stop|check|lap|how long|time|read)\b", t):
            if _STOPWATCH["start"] is None:
                return "The stopwatch is not running."
            el = time.monotonic() - _STOPWATCH["start"]
            if re.search(r"\bstop\b", t):
                _STOPWATCH["start"] = None
                return f"Stopped at {fmt_secs(el)}."
            return f"{fmt_secs(el)} so far."
        return None
    is_remind = re.search(r"\bremind me (?:in|after)\b", t)
    t = re.sub(r"^(?:start|begin|put on|make|create) (?:a |the )?countdown(?: timer)? (?:for|of) ", "set a timer for ", t)
    t = re.sub(r"^(?:set|start|put on|create) (?:a |the )?alarm in ", "set a timer for ", t)
    t = re.sub(r"^remind me after ", "remind me in ", t)
    t = re.sub(r"^(?:how much longer|how long to go|how much time remains) on (?:my|the) (?:countdown|timer)$", "time left on the timer", t)
    t = re.sub(r"^(?:stop|cancel|end) (?:my|the) countdown$", "cancel the timer", t)
    bare_left = re.fullmatch(r"(?:how (?:long|much time|many minutes)(?: is)?(?: there)? (?:left|remaining)|time left|how much longer)", t)
    if bare_left and _TIMERS:
        with _lock:
            items = sorted(_TIMERS.items())
        return " ".join(f"Timer {tid}{' for ' + i['label'] if i['label'] else ''}: "
                        f"{fmt_secs(max(0, i['end'] - time.monotonic()))} left." for tid, i in items)
    if not (re.search(r"\btimers?\b", t) or is_remind):
        return None
    if re.search(r"\b(cancel|stop|clear|delete|remove|turn off)\b", t):
        with _lock:
            n = len(_TIMERS)
            for info in _TIMERS.values():
                info["timer"].cancel()
            _TIMERS.clear()
        return f"Cancelled {plural(n, 'timer')}." if n else "No timers are running."
    if re.search(r"\b(left|remaining|status|check|how long|how much|running|list)\b", t) and not re.search(r"\b(set|start|for)\b", t):
        with _lock:
            items = sorted(_TIMERS.items())
        if not items:
            return "No timers are running."
        return " ".join(f"Timer {tid}{' for ' + i['label'] if i['label'] else ''}: "
                        f"{fmt_secs(max(0, i['end'] - time.monotonic()))} left." for tid, i in items)
    secs = _parse_seconds(t)
    if not secs or secs <= 0:
        return None
    label = None
    m = re.search(r"\bremind me in .+? to (.+)$", t)
    if m:
        label = m.group(1).strip()
    with _lock:
        _TIMER_SEQ[0] += 1
        tid = _TIMER_SEQ[0]
        timer = threading.Timer(secs, _fire, args=(tid,))
        timer.daemon = True
        _TIMERS[tid] = {"end": time.monotonic() + secs, "label": label, "timer": timer}
        timer.start()
    return f"Timer set for {fmt_secs(secs)}." if not label else f"I will remind you in {fmt_secs(secs)} to {label}."


# ----------------------------------------------------------------------------
# Tool: lists and notes (stored locally in CRUTCH_HOME/data.json)
# ----------------------------------------------------------------------------

def _load():
    p = home_dir() / "data.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"lists": {}, "notes": []}


def _save(d):
    p = home_dir() / "data.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    os.replace(tmp, p)


def _clean_name(s: str) -> str:
    return re.sub(r"^(?:my|the)\s+", "", s.strip().lower())


def tool_lists_notes(text: str):
    t = text.strip().lower().rstrip("?!. ")
    d = _load()
    lists, notes = d["lists"], d["notes"]

    t = re.sub(r"^(?:write down|jot down|save this note|keep a note|make a note of|note to self)\s+(?:that\s+)?", "remember ", t)
    t = re.sub(r"^(?:put|place) (.+?) on (?:my|the) (shopping|grocery|todo|to do) list$", r"add \1 to my \2 list", t)
    t = re.sub(r"^(?:take|cross) (.+?) off (?:my|the) (shopping|grocery|todo|to do) list$", r"remove \1 from my \2 list", t)
    t = re.sub(r"^(?:what do i need to buy|what am i buying|what's on my grocery list)$", "what's on my shopping list", t)
    t = re.sub(r"^(?:show me|read out|check) (?:everything on )?(?:my|the) (shopping|grocery|todo|to do) list$", r"show my \1 list", t)
    m = re.match(r"^(?:please\s+)?(?:remember|make a note|note)(?:\s+that)?\s+(.+)$", t)
    if m:
        notes.append({"text": m.group(1), "when": now().isoformat(timespec="minutes")})
        _save(d)
        return "Got it. I'll remember that."
    m = re.match(r"^(?:what do you remember|what did i tell you|what notes do i have|show (?:me )?(?:my )?notes|"
                 r"read (?:me )?(?:my )?notes|recall)(?:\s+(?:about|on|regarding)\s+(.+))?$", t)
    if m:
        q = m.group(1)
        hits = [n["text"] for n in notes if not q or q in n["text"]]
        if not hits:
            return "I have no notes about that." if q else "I have no notes yet."
        return "You told me: " + "; ".join(hits[-5:]) + "."
    m = re.match(r"^forget\s+(?:about\s+|that\s+)?(.+)$", t)
    if m:
        before = len(notes)
        d["notes"] = [n for n in notes if m.group(1) not in n["text"]]
        if len(d["notes"]) < before:
            _save(d)
            return f"Forgot {plural(before - len(d['notes']), 'note')}."
        return None

    m = re.match(r"^(?:please\s+)?(?:add|put|include|place)\s+(.+?)\s+(?:to|on)\s+(?:my\s+|the\s+|a\s+)?(?:(.+?)\s+)?list$", t)
    if m:
        items = [i.strip() for i in re.split(r",|\band\b", m.group(1)) if i.strip()]
        name = _clean_name(m.group(2)) if m.group(2) else (next(iter(lists)) if len(lists) == 1 else "shopping")
        lst = lists.setdefault(name, [])
        for i in items:
            if i not in lst:
                lst.append(i)
        _save(d)
        return f"Added {', '.join(items)} to your {name} list."
    m = re.match(r"^(?:remove|delete|take|cross|strike)\s+(.+?)(?:\s+(?:from|off|out of)\s+(?:my\s+|the\s+|a\s+)?(?:(.+?)\s+)?list)?$", t)
    if m:
        item = m.group(1).strip()
        explicit = "list" in t
        if m.group(2):
            name = _clean_name(m.group(2))
            if name not in lists:
                return None
            holders = [name]
        else:
            holders = [n for n, v in lists.items() if item in v] or \
                      [n for n, v in lists.items() if any(item in x for x in v)]
        if len(holders) == 1:
            lst = lists[holders[0]]
            hit = item if item in lst else next(x for x in lst if item in x)
            lst.remove(hit)
            _save(d)
            return f"Removed {hit} from your {holders[0]} list."
        if len(holders) > 1:
            return f"{item} is on more than one list. Which one?"
        return f"{item} isn't on your list." if explicit else None
    m = re.match(r"^(?:clear|empty|wipe)\s+(?:my\s+|the\s+)?(.+?)\s+list$", t)
    if m and _clean_name(m.group(1)) in lists:
        lists[_clean_name(m.group(1))] = []
        _save(d)
        return f"Cleared your {_clean_name(m.group(1))} list."
    if re.match(r"^(?:what|which) lists do i have$", t) or t == "show my lists":
        names = [n for n, v in lists.items() if v]
        return ("You have lists for " + ", ".join(names) + ".") if names else "You have no lists yet."
    m = re.match(r"^(?:what(?:'s| is| are)?(?:\s+(?:on|in))?|show(?:\s+me)?|read(?:\s+me)?|tell me|list)\s+"
                 r"((?:my\s+|the\s+)?.+?)\s+list$", t)
    if m:
        name = _clean_name(m.group(1))
        if name in lists:
            items = lists[name]
            return (f"Your {name} list has: " + ", ".join(items) + ".") if items else f"Your {name} list is empty."
        if m.group(1).startswith("my "):
            return f"Your {name} list is empty."
    return None


# ----------------------------------------------------------------------------
# Tool: fact table (extend by editing facts.json in CRUTCH_HOME)
# ----------------------------------------------------------------------------

DEFAULT_FACTS = [
    {"match": ["boil", "egg"],
     "answer": "Soft-boiled takes about 6 minutes, medium about 8 to 9, and hard-boiled 10 to 12. "
               "Start timing when the water is boiling and the egg is in."},
    {"match": ["chicken", "temp"],
     "answer": "Chicken is safe at an internal temperature of 74 degrees Celsius, which is 165 Fahrenheit."},
    {"match": ["boiling point"],
     "answer": "Water boils at 100 degrees Celsius, or 212 Fahrenheit, at sea level."},
    {"match": ["cook", "pasta"],
     "answer": "Most dried pasta takes 8 to 12 minutes in boiling salted water. Check the packet and taste it."},
    {"match": ["cook", "rice"],
     "answer": "White rice usually takes about 15 to 18 minutes covered, roughly 1 part rice to 2 parts water. "
               "Check the packet for your type."},
]


def tool_facts(text: str):
    t = text.lower()
    facts = list(DEFAULT_FACTS)
    p = home_dir() / "facts.json"
    if p.exists():
        try:
            facts = json.loads(p.read_text()) + facts
        except Exception:
            pass
    for f in facts:
        if all(w.lower() in t for w in f["match"]):
            return f["answer"]
    return None


# ----------------------------------------------------------------------------
# Offline geography and basic knowledge: anchored questions, NOT chat interception.
# Facts are intentionally stable: no live populations, current presidents or prices.
# "States" covers all 50 US states and Australia's six states + two territories.
# ----------------------------------------------------------------------------

US_STATES = {
    "alabama": ("Montgomery", "AL"), "alaska": ("Juneau", "AK"),
    "arizona": ("Phoenix", "AZ"), "arkansas": ("Little Rock", "AR"),
    "california": ("Sacramento", "CA"), "colorado": ("Denver", "CO"),
    "connecticut": ("Hartford", "CT"), "delaware": ("Dover", "DE"),
    "florida": ("Tallahassee", "FL"), "georgia": ("Atlanta", "GA"),
    "hawaii": ("Honolulu", "HI"), "idaho": ("Boise", "ID"),
    "illinois": ("Springfield", "IL"), "indiana": ("Indianapolis", "IN"),
    "iowa": ("Des Moines", "IA"), "kansas": ("Topeka", "KS"),
    "kentucky": ("Frankfort", "KY"), "louisiana": ("Baton Rouge", "LA"),
    "maine": ("Augusta", "ME"), "maryland": ("Annapolis", "MD"),
    "massachusetts": ("Boston", "MA"), "michigan": ("Lansing", "MI"),
    "minnesota": ("Saint Paul", "MN"), "mississippi": ("Jackson", "MS"),
    "missouri": ("Jefferson City", "MO"), "montana": ("Helena", "MT"),
    "nebraska": ("Lincoln", "NE"), "nevada": ("Carson City", "NV"),
    "new hampshire": ("Concord", "NH"), "new jersey": ("Trenton", "NJ"),
    "new mexico": ("Santa Fe", "NM"), "new york": ("Albany", "NY"),
    "north carolina": ("Raleigh", "NC"), "north dakota": ("Bismarck", "ND"),
    "ohio": ("Columbus", "OH"), "oklahoma": ("Oklahoma City", "OK"),
    "oregon": ("Salem", "OR"), "pennsylvania": ("Harrisburg", "PA"),
    "rhode island": ("Providence", "RI"), "south carolina": ("Columbia", "SC"),
    "south dakota": ("Pierre", "SD"), "tennessee": ("Nashville", "TN"),
    "texas": ("Austin", "TX"), "utah": ("Salt Lake City", "UT"),
    "vermont": ("Montpelier", "VT"), "virginia": ("Richmond", "VA"),
    "washington": ("Olympia", "WA"), "west virginia": ("Charleston", "WV"),
    "wisconsin": ("Madison", "WI"), "wyoming": ("Cheyenne", "WY"),
}
AU_REGIONS = {
    "new south wales": ("Sydney", "NSW", "state"),
    "victoria": ("Melbourne", "VIC", "state"),
    "queensland": ("Brisbane", "QLD", "state"),
    "south australia": ("Adelaide", "SA", "state"),
    "western australia": ("Perth", "WA", "state"),
    "tasmania": ("Hobart", "TAS", "state"),
    "northern territory": ("Darwin", "NT", "territory"),
    "australian capital territory": ("Canberra", "ACT", "territory"),
}
# Frequently asked country capitals. Complex capital arrangements are described
# carefully, rather than giving an oversimplified answer.
COUNTRY_CAPITALS = {
    "afghanistan": "Kabul", "albania": "Tirana", "algeria": "Algiers",
    "andorra": "Andorra la Vella", "angola": "Luanda", "antigua and barbuda": "Saint John's",
    "argentina": "Buenos Aires", "armenia": "Yerevan", "australia": "Canberra",
    "austria": "Vienna", "azerbaijan": "Baku", "bahamas": "Nassau",
    "bahrain": "Manama", "bangladesh": "Dhaka", "barbados": "Bridgetown",
    "belarus": "Minsk", "belgium": "Brussels", "belize": "Belmopan",
    "benin": "Porto-Novo (official capital; the government is based in Cotonou)",
    "bhutan": "Thimphu", "bolivia": "Sucre (constitutional capital; the government sits in La Paz)",
    "bosnia and herzegovina": "Sarajevo", "botswana": "Gaborone", "brazil": "Brasília",
    "brunei": "Bandar Seri Begawan", "bulgaria": "Sofia", "burkina faso": "Ouagadougou",
    "burundi": "Gitega", "cabo verde": "Praia", "cambodia": "Phnom Penh",
    "cameroon": "Yaoundé", "canada": "Ottawa", "central african republic": "Bangui",
    "chad": "N'Djamena", "chile": "Santiago", "china": "Beijing",
    "colombia": "Bogotá", "comoros": "Moroni", "costa rica": "San José",
    "croatia": "Zagreb", "cuba": "Havana", "cyprus": "Nicosia",
    "czechia": "Prague", "democratic republic of the congo": "Kinshasa",
    "denmark": "Copenhagen", "djibouti": "Djibouti", "dominica": "Roseau",
    "dominican republic": "Santo Domingo", "ecuador": "Quito", "egypt": "Cairo",
    "el salvador": "San Salvador", "equatorial guinea": "Ciudad de la Paz (declared the capital in January 2026; government relocation is ongoing)", "eritrea": "Asmara",
    "estonia": "Tallinn", "eswatini": "Mbabane (administrative capital) and Lobamba (legislative and royal capital)",
    "ethiopia": "Addis Ababa", "fiji": "Suva", "finland": "Helsinki",
    "france": "Paris", "gabon": "Libreville", "gambia": "Banjul",
    "georgia": "Tbilisi", "germany": "Berlin", "ghana": "Accra",
    "greece": "Athens", "grenada": "Saint George's", "guatemala": "Guatemala City",
    "guinea": "Conakry", "guinea-bissau": "Bissau", "guyana": "Georgetown",
    "haiti": "Port-au-Prince", "honduras": "Tegucigalpa", "hungary": "Budapest",
    "iceland": "Reykjavík", "india": "New Delhi", "indonesia": "Jakarta (official capital in 2026; the government plans to relocate to Nusantara)",
    "iran": "Tehran", "iraq": "Baghdad", "ireland": "Dublin",
    "israel": "Jerusalem (its status is internationally disputed)", "italy": "Rome",
    "ivory coast": "Yamoussoukro (official capital; government mostly operates in Abidjan)",
    "jamaica": "Kingston", "japan": "Tokyo", "jordan": "Amman",
    "kazakhstan": "Astana", "kenya": "Nairobi", "kiribati": "South Tarawa",
    "kuwait": "Kuwait City", "kyrgyzstan": "Bishkek", "laos": "Vientiane",
    "latvia": "Riga", "lebanon": "Beirut", "lesotho": "Maseru",
    "liberia": "Monrovia", "libya": "Tripoli", "liechtenstein": "Vaduz",
    "lithuania": "Vilnius", "luxembourg": "Luxembourg", "madagascar": "Antananarivo",
    "malawi": "Lilongwe", "malaysia": "Kuala Lumpur (official capital; administrative centre is Putrajaya)",
    "maldives": "Malé", "mali": "Bamako", "malta": "Valletta",
    "marshall islands": "Majuro", "mauritania": "Nouakchott", "mauritius": "Port Louis",
    "mexico": "Mexico City", "micronesia": "Palikir", "moldova": "Chișinău",
    "monaco": "Monaco", "mongolia": "Ulaanbaatar", "montenegro": "Podgorica",
    "morocco": "Rabat", "mozambique": "Maputo", "myanmar": "Naypyidaw",
    "namibia": "Windhoek", "nauru": "Yaren District (de facto seat; no official capital)",
    "nepal": "Kathmandu", "netherlands": "Amsterdam (constitutional capital; government is in The Hague)",
    "new zealand": "Wellington", "nicaragua": "Managua", "niger": "Niamey",
    "nigeria": "Abuja", "north korea": "Pyongyang", "north macedonia": "Skopje",
    "norway": "Oslo", "oman": "Muscat", "pakistan": "Islamabad",
    "palau": "Ngerulmud", "panama": "Panama City", "papua new guinea": "Port Moresby",
    "paraguay": "Asunción", "peru": "Lima", "philippines": "Manila",
    "poland": "Warsaw", "portugal": "Lisbon", "qatar": "Doha",
    "republic of the congo": "Brazzaville", "romania": "Bucharest", "russia": "Moscow",
    "rwanda": "Kigali", "saint kitts and nevis": "Basseterre",
    "saint lucia": "Castries", "saint vincent and the grenadines": "Kingstown",
    "samoa": "Apia", "san marino": "San Marino", "sao tome and principe": "São Tomé",
    "saudi arabia": "Riyadh", "senegal": "Dakar", "serbia": "Belgrade",
    "seychelles": "Victoria", "sierra leone": "Freetown", "singapore": "Singapore",
    "slovakia": "Bratislava", "slovenia": "Ljubljana", "solomon islands": "Honiara",
    "somalia": "Mogadishu", "south africa": "Pretoria (executive), Cape Town (legislative) and Bloemfontein (judicial)",
    "south korea": "Seoul", "south sudan": "Juba", "spain": "Madrid",
    "sri lanka": "Sri Jayawardenepura Kotte (legislative capital; Colombo is the main commercial centre)",
    "sudan": "Khartoum", "suriname": "Paramaribo", "sweden": "Stockholm",
    "switzerland": "Bern (federal city, the de facto capital)", "syria": "Damascus",
    "tajikistan": "Dushanbe", "tanzania": "Dodoma", "thailand": "Bangkok",
    "timor-leste": "Dili", "togo": "Lomé", "tonga": "Nukuʻalofa",
    "trinidad and tobago": "Port of Spain", "tunisia": "Tunis", "turkey": "Ankara",
    "turkmenistan": "Ashgabat", "tuvalu": "Funafuti", "uganda": "Kampala",
    "ukraine": "Kyiv", "united arab emirates": "Abu Dhabi", "united kingdom": "London",
    "united states": "Washington, D.C.", "uruguay": "Montevideo", "uzbekistan": "Tashkent",
    "vanuatu": "Port Vila", "vatican city": "Vatican City", "venezuela": "Caracas",
    "vietnam": "Hanoi", "yemen": "Sana'a (constitutional capital; the internationally recognised government operates from Aden)",
    "zambia": "Lusaka", "zimbabwe": "Harare",
}
COUNTRY_ALIASES = {
    "usa": "united states", "us": "united states", "u s": "united states",
    "america": "united states", "united states of america": "united states",
    "uk": "united kingdom", "u k": "united kingdom", "britain": "united kingdom",
    "great britain": "united kingdom", "uae": "united arab emirates",
    "czech republic": "czechia", "holland": "netherlands",
    "turkiye": "turkey", "türkiye": "turkey", "burma": "myanmar",
    "cape verde": "cabo verde", "cote d'ivoire": "ivory coast",
    "côte d'ivoire": "ivory coast", "swaziland": "eswatini",
    "east timor": "timor-leste", "dr congo": "democratic republic of the congo",
    "dr c": "democratic republic of the congo", "drc": "democratic republic of the congo",
}
US_ABBREVIATIONS = {code.lower(): state for state, (_, code) in US_STATES.items()}
AU_ABBREVIATIONS = {code.lower(): state for state, (_, code, _) in AU_REGIONS.items()}


def _geo_entity(value: str) -> str:
    v = re.sub(r"^(?:the )", "", value.lower().strip(" .?!"))
    v = re.sub(r"^(?:the )?(?:us |american |australian )?(?:state|country|nation|territory) of ", "", v)
    v = re.sub(r"\s+(?:in|of) (?:the )?(?:united states|usa|us|america|australia)$", "", v)
    v = re.sub(r"\s+the (?:us state|american state|country|nation|state|territory)$", "", v)
    v = re.sub(r"\s+(?:country|nation|state)$", "", v)
    v = re.sub(r"\s+(?:state|country)$", "", v)
    return COUNTRY_ALIASES.get(v, v)


def tool_geography(text: str):
    t = re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip(" ?!.,")
    t = re.sub(r"^(?:please |hey |hey aria |aria |can you tell me |could you tell me |do you know )", "", t)
    # Fixed, unambiguous lists and counts.
    if re.fullmatch(r"(?:how many|number of) (?:states|us states|american states) (?:are (?:there )?)?(?:in )?(?:the )?(?:usa|us|america|united states)", t):
        return "The United States has 50 states. Washington, D.C. is a federal district, not a state."
    if re.fullmatch(r"(?:how many|number of) (?:states|states and territories) (?:are there )?in australia", t):
        return "Australia has six states and two main self-governing mainland territories."
    if re.fullmatch(r"(?:name|list|what are|tell me) (?:all |the )?(?:australian |australia'?s )?states(?: and territories)?", t) and "austral" in t:
        return "Australia's states are " + ", ".join(n.title() for n, (_, _, kind) in AU_REGIONS.items() if kind == "state") + "; its two main mainland territories are Northern Territory and Australian Capital Territory."
    if re.fullmatch(r"(?:what(?:'s| is)|what does|tell me what) ([a-z]{2,3}) (?:stand for|mean)", t):
        abbr = re.fullmatch(r"(?:what(?:'s| is)|what does|tell me what) ([a-z]{2,3}) (?:stand for|mean)", t).group(1)
        if abbr in US_ABBREVIATIONS and abbr in AU_ABBREVIATIONS:
            return f"{abbr.upper()} means {US_ABBREVIATIONS[abbr].title()} in the US or {AU_ABBREVIATIONS[abbr].title()} in Australia."
        if abbr in US_ABBREVIATIONS:
            return f"{abbr.upper()} is the US postal abbreviation for {US_ABBREVIATIONS[abbr].title()}."
        if abbr in AU_ABBREVIATIONS:
            return f"{abbr.upper()} is the Australian abbreviation for {AU_ABBREVIATIONS[abbr].title()}."

    if re.fullmatch(r"(?:how many|number of) (?:us |american |united states )?states (?:are there|does (?:the )?(?:us|usa|america|united states) have|in (?:the )?(?:us|usa|america|united states))?", t):
        return "The United States has 50 states. Washington, D.C. is a federal district, not a state."
    if re.fullmatch(r"(?:how many|number of) (?:states|states and territories) (?:are (?:there )?in|does) australia(?: have)?", t):
        return "Australia has six states and two main self-governing mainland territories."
    if re.fullmatch(r"(?:(?:name|list|what are|tell me) (?:all |the )?(?:50 |united states |us |american )?states(?: of (?:the )?(?:us|united states|america))?|list all 50 states)", t):
        return "The 50 US states are: " + ", ".join(n.title() for n in US_STATES) + "."
    if re.fullmatch(r"(?:name|list|what are|tell me) (?:all |the )?(?:australian )?states(?: and territories)?(?: of australia)?", t) and "austral" in t:
        return "Australia's states are " + ", ".join(n.title() for n, (_, _, kind) in AU_REGIONS.items() if kind == "state") + "; its two main mainland territories are Northern Territory and Australian Capital Territory."
    # State/territory abbreviations and full names.
    m = re.fullmatch(r"(?:what(?:'s| is)|give me|tell me|find|name) (?:the )?(?:state |territory )?(?:abbreviation|postal code|two.letter code|short form|abbreviation of) (?:for|of)? ?(?:the )?(.+)", t)
    if not m:
        m = re.fullmatch(r"(?:how do you abbreviate|how is) (.+?)(?: abbreviated)?", t)
    if m:
        entity = _geo_entity(m.group(1))
        if entity in US_STATES:
            return f"The US state abbreviation for {entity.title()} is {US_STATES[entity][1]}."
        if entity in AU_REGIONS:
            return f"The Australian abbreviation for {entity.title()} is {AU_REGIONS[entity][1]}."
    m = re.fullmatch(r"(?:what(?:'s| is)|which|name|tell me) (?:the )?(?:us |american |australian )?(?:state|territory) (?:is )?(?:abbreviated|shortened|known) (?:as )?([a-z]{2,3})", t)
    if m:
        a = m.group(1)
        if a in US_ABBREVIATIONS:
            return f"{a.upper()} stands for {US_ABBREVIATIONS[a].title()} in the United States."
        if a in AU_ABBREVIATIONS:
            return f"{a.upper()} stands for {AU_ABBREVIATIONS[a].title()} in Australia."
    m = re.fullmatch(r"(?:what(?:'s| is)|which|name|give me|tell me)? ?(?:the )?(?:capital|capital city|state capital) (?:of|for|in) (?:the )?(.+)", t)
    if not m:
        m = re.fullmatch(r"(?:what(?:'s| is) )?(?:the )?(.+?)(?:'s|') (?:capital|capital city|state capital)", t)
    if not m:
        m = re.fullmatch(r"(?:where is|what city is) (?:the )?capital of (?:the )?(.+)", t)
    if m:
        entity = _geo_entity(m.group(1))
        country_only = bool(re.search(r"\b(country|nation)\b", t))
        state_only = bool(re.search(r"\b(state|territory)\b", t))
        if entity == "georgia" and not (country_only or state_only):
            return "Georgia the US state has Atlanta as its capital; Georgia the country has Tbilisi."
        if entity in US_STATES and not country_only:
            return f"The capital of {entity.title()}, USA, is {US_STATES[entity][0]}."
        if entity in AU_REGIONS and not country_only:
            capital, _, kind = AU_REGIONS[entity]
            return f"The capital of the Australian {kind} of {entity.title()} is {capital}."
        if entity in COUNTRY_CAPITALS and not state_only:
            name = "the United States" if entity == "united states" else entity.title()
            capital = COUNTRY_CAPITALS[entity]
            return f"The capital of {name} is {capital}{'' if capital.endswith('.') else '.'}"
    # Short direct questions about the size of states (not changing population).
    if re.fullmatch(r"(?:what(?:'s| is)|which is) (?:the )?(?:largest|biggest) (?:us|american|united states) state(?: by (?:land )?area)?", t) or re.fullmatch(r"(?:what(?:'s| is)|which is) (?:the )?(?:largest|biggest) state in (?:the )?(?:us|usa|united states|america)", t):
        return "Alaska is the largest US state by area."
    if re.fullmatch(r"(?:what(?:'s| is)|which is) (?:the )?smallest (?:us|american|united states) state(?: by (?:land )?area)?", t) or re.fullmatch(r"(?:what(?:'s| is)|which is) (?:the )?smallest state in (?:the )?(?:us|usa|united states|america)", t):
        return "Rhode Island is the smallest US state by area."
    if re.fullmatch(r"(?:what(?:'s| is)|which is) (?:the )?(?:largest|biggest) (?:australian )?state(?: by (?:land )?area)?", t) and "australian" in t:
        return "Western Australia is Australia's largest state by area."
    if re.fullmatch(r"(?:what(?:'s| is)|which is) (?:the )?smallest (?:australian )?state(?: by (?:land )?area)?", t) and "australian" in t:
        return "Tasmania is Australia's smallest state by area."
    # Inverse country/state capital question. Only give the answer for unique cases.
    m = re.fullmatch(r"(?:what|which) (country|nation|us state|american state|australian state|state) (?:has|is|contains) (?:the )?(?:capital (?:city )?(?:of |in )?|city of |capital )?(.+?)(?: as (?:its|the) capital)?", t)
    if m:
        mode, place = m.groups()
        place = place.lower()
        if mode in ("country", "nation"):
            matches = [k for k,v in COUNTRY_CAPITALS.items() if v.lower().split(" (")[0] == place]
            if len(matches) == 1:
                return f"{COUNTRY_CAPITALS[matches[0]]} is the capital of {matches[0].title()}."
        else:
            d = AU_REGIONS if "australian" in mode else US_STATES
            matches = [k for k,v in d.items() if v[0].lower() == place]
            if len(matches) == 1:
                return f"{d[matches[0]][0]} is the capital of {matches[0].title()}."
    return None

# Curated fixed factual answers, not arbitrary sentence similarity. Each regex
# must consume the whole question, so ordinary chat still goes to the LLM.
BASIC_QA = [
    (r"(?:what(?:'s| is)|tell me) (?:the )?(?:fastest|top speed of an?|maximum running speed of an?) ostrich|how fast (?:can|does) an? ostrich (?:run|go)", "An ostrich can run at about 70 kilometres per hour."),
    (r"(?:what(?:'s| is) (?:the )?|which is (?:the )?)(?:fastest land animal|fastest animal on land)", "The cheetah is the fastest land animal, reaching about 100 to 120 kilometres per hour in short bursts."),
    (r"(?:what(?:'s| is) (?:the )?|which is (?:the )?)(?:largest animal|biggest animal|largest mammal|biggest mammal)(?: in the world)?", "The blue whale is the largest living animal."),
    (r"(?:what(?:'s| is) (?:the )?|which is (?:the )?)(?:tallest (?:living )?animal)", "The giraffe is the tallest living land animal."),
    (r"(?:how many|number of) (?:legs|arms) (?:does|do) (?:an? )?spider(?:s)? have", "Spiders have eight legs."),
    (r"(?:how many|number of) legs (?:does|do) (?:an? )?insect(?:s)? have", "Insects have six legs."),
    (r"(?:how many|number of) hearts (?:does|do) (?:an? )?octopus(?:es)? have", "An octopus has three hearts."),
    (r"(?:what(?:'s| is)|which is) (?:the )?(?:largest ocean|biggest ocean|largest ocean on earth)", "The Pacific Ocean is Earth's largest ocean."),
    (r"(?:what(?:'s| is)|which is) (?:the )?(?:largest (?:hot )?desert|biggest (?:hot )?desert)(?: on earth| in the world)?", "Antarctica is the largest desert overall. The Sahara is the largest hot desert."),
    (r"(?:what(?:'s| is)|which is) (?:the )?(?:highest (?:mountain|peak)|tallest mountain)(?: in the world| on earth)?", "Mount Everest has the highest summit above sea level, at approximately 8,849 metres."),
    (r"(?:what(?:'s| is)|which is) (?:the )?(?:deepest ocean trench|deepest (?:part|point) of the ocean)", "The Challenger Deep in the Mariana Trench is the deepest known part of the ocean."),
    (r"(?:what(?:'s| is)|which is) (?:the )?(?:largest country|biggest country)(?: by area| in the world)?", "Russia is the largest country by land area."),
    (r"(?:what(?:'s| is)|which is) (?:the )?(?:smallest country)(?: in the world)?", "Vatican City is the world's smallest independent state by area."),
    (r"(?:how many|number of) continents (?:are there|does earth have|on earth)", "In the commonly used seven-continent model, Earth has seven continents."),
    (r"(?:what are|name|list|tell me) (?:the )?(?:seven|7|all) continents", "The seven continents are Africa, Antarctica, Asia, Europe, North America, South America and Australia."),
    (r"(?:what(?:'s| is)|explain) (?:the )?equator", "The equator is an imaginary circle around Earth at zero degrees latitude."),
    (r"(?:how many|number of) planets (?:are there|in (?:our|the) solar system)", "There are eight planets in our Solar System."),
    (r"(?:list|name|what are|tell me) (?:the )?(?:eight|8|all) planets(?: in (?:our|the) solar system)?", "The eight planets in order from the Sun are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune."),
    (r"(?:what(?:'s| is)|which is) (?:the )?(?:closest planet to (?:the )?sun|first planet from (?:the )?sun)", "Mercury is the closest planet to the Sun."),
    (r"(?:what(?:'s| is)|which is) (?:the )?(?:largest planet|biggest planet)(?: in (?:our|the) solar system)?", "Jupiter is the largest planet in our Solar System."),
    (r"(?:what(?:'s| is)|which is) (?:the )?(?:smallest planet)(?: in (?:our|the) solar system)?", "Mercury is the smallest of the eight planets."),
    (r"(?:what(?:'s| is)|which is) (?:the )?(?:hottest planet)(?: in (?:our|the) solar system)?", "Venus is the hottest planet in our Solar System."),
    (r"(?:what(?:'s| is)|which is) (?:the )?(?:red planet)", "Mars is called the Red Planet because of iron oxides on its surface."),
    (r"(?:is|why is) pluto (?:a planet|not a planet|a dwarf planet)", "Pluto is classified as a dwarf planet because it has not cleared its orbital neighbourhood."),
    (r"(?:what(?:'s| is)|how long is) (?:a |the )?light.?year", "A light-year is a unit of distance: the distance light travels in a vacuum in one year, about 9.46 trillion kilometres."),
    (r"(?:what(?:'s| is)|how fast is) (?:the )?speed of light(?: in (?:a )?vacuum)?", "Light travels in a vacuum at exactly 299,792,458 metres per second."),
    (r"(?:what(?:'s| is)|how fast is) (?:the )?speed of sound(?: in air)?", "Sound travels in air at roughly 343 metres per second at 20 degrees Celsius."),
    (r"(?:what(?:'s| is)|what is) (?:the )?(?:gravity|gravitational acceleration) (?:on|of) earth", "Earth's surface gravity is about 9.81 metres per second squared."),
    (r"(?:how (?:old|many years old) is|what(?:'s| is) the age of) (?:the )?earth", "Earth formed approximately 4.54 billion years ago."),
    (r"(?:how far (?:away )?is|what(?:'s| is) the (?:average )?distance (?:to|between)) (?:the )?moon(?: from earth| and earth)?", "The Moon's average distance from Earth is about 384,400 kilometres."),
    (r"(?:what(?:'s| is)|what is) (?:the )?(?:chemical )?formula (?:of|for) water", "The chemical formula of water is H2O."),
    (r"(?:what(?:'s| is)|what is) (?:the )?(?:chemical )?symbol (?:for|of) (?:gold|oxygen|iron|silver|hydrogen|carbon|nitrogen|helium)", None),
    (r"(?:what(?:'s| is)|what is) (?:the )?(?:freezing point|temperature water freezes at|freezing temperature) (?:of |for )?water(?: in celsius)?", "Pure water freezes at 0 degrees Celsius at standard atmospheric pressure."),
    (r"(?:what(?:'s| is)|what is) (?:the )?(?:boiling point|boiling temperature) (?:of |for )?water", "Water boils at 100 degrees Celsius at sea-level pressure."),
    (r"(?:what(?:'s| is)|what is) (?:the )?(?:main gas|most common gas) in (?:earth'?s |the )?atmosphere", "Nitrogen makes up about 78 percent of Earth's dry atmosphere."),
    (r"(?:what(?:'s| is)|what is) (?:the )?(?:percentage|percent|amount) of oxygen in (?:the )?air", "Oxygen makes up about 21 percent of Earth's dry atmosphere."),
    (r"(?:what(?:'s| is)|explain|define) (?:the )?photosynthesis", "Photosynthesis is the process by which plants and some other organisms use light to make sugars from carbon dioxide and water, releasing oxygen."),
    (r"(?:what(?:'s| is)|explain|define) (?:the )?dna|what does dna stand for", "DNA stands for deoxyribonucleic acid. It carries genetic information in living organisms."),
    (r"(?:what(?:'s| is)|explain|define) (?:the )?mitochondri(?:a|on)", "Mitochondria are cell structures that produce most of the ATP used by many eukaryotic cells."),
    (r"(?:how many|number of) bones (?:does|do) (?:a |an )?(?:adult )?human(?:s)? have", "A typical adult human skeleton has 206 bones."),
    (r"(?:how many|number of) chambers (?:are in|does) (?:the |a )?(?:human )?heart(?: have)?", "The human heart has four chambers."),
    (r"(?:what(?:'s| is)|what is) (?:the )?normal (?:neutral )?ph(?: of water)?", "A neutral solution has a pH of 7 at 25 degrees Celsius."),
    (r"(?:what(?:'s| is)|what is) (?:the )?value of pi", "Pi is approximately 3.141592653589793."),
    (r"(?:how many|number of) days (?:in|are in|does) (?:a |one )?leap year(?: have)?", "A leap year has 366 days."),
    (r"(?:how many|number of) (?:seconds|minutes) (?:are there |are |is )?in (?:a |one )?(?:hour|day)", None),
    (r"(?:when|what year|in what year) (?:did )?(?:world war one|world war 1|ww1|the first world war) (?:start|begin|end|happen|take place)?", "World War One lasted from 1914 to 1918."),
    (r"(?:when|what year|in what year) (?:did )?(?:world war two|world war 2|ww2|the second world war) (?:start|begin|end|happen|take place)?", "World War Two lasted from 1939 to 1945."),
    (r"(?:when|what year|in what year) (?:did )?(?:the )?holocaust (?:start|begin|end|happen|take place)?", "The Nazi persecution of Jews began in 1933; the systematic mass murder known as the Holocaust took place mainly from 1941 to 1945."),
    (r"(?:who (?:was|is) (?:responsible for|behind) (?:the )?holocaust|who carried out (?:the )?holocaust)", "Nazi Germany, led by Adolf Hitler, organised and carried out the Holocaust with the participation of collaborators."),
    (r"(?:when|what year|in what year) (?:did )?(?:the )?titanic (?:sink|sank)", "The RMS Titanic sank on April 15, 1912."),
    (r"(?:when|what year|in what year) (?:did )?(?:humans|people|man|we) (?:first )?(?:land|walk) on (?:the )?moon", "Humans first landed on the Moon with Apollo 11 on July 20, 1969."),
    (r"(?:who (?:was|is) (?:the )?first (?:man|person|human) (?:on|to walk on) (?:the )?moon|who first walked on the moon)", "Neil Armstrong was the first person to walk on the Moon, in 1969."),
    (r"(?:when|what year|in what year) (?:did )?australia (?:become a federation|federate|federated)", "Australia federated on January 1, 1901."),
    (r"(?:when|what year|in what year) (?:was|did) (?:the )?eiffel tower (?:built|constructed|finished|completed|open)?", "The Eiffel Tower was constructed between 1887 and 1889 and opened in 1889."),
    (r"(?:when|what year|in what year) (?:was|did) (?:the )?statue of liberty (?:built|dedicated|opened|finished|completed)?", "The Statue of Liberty was dedicated in New York in 1886."),
    (r"(?:when|what year|in what year) (?:did )?(?:the )?berlin wall (?:fall|come down|open)", "The Berlin Wall opened on November 9, 1989."),
    (r"(?:when|what year|in what year) (?:did )?(?:the )?united states (?:declare independence|become independent)|when is american independence day", "The United States adopted its Declaration of Independence on July 4, 1776."),
    (r"(?:who (?:invented|is credited with inventing) (?:the )?telephone)", "Alexander Graham Bell is commonly credited with the first successful telephone patent, in 1876; several inventors contributed to its development."),
    (r"(?:what(?:'s| is)|which is) (?:the )?(?:national animal of australia)", "Australia's coat of arms features a kangaroo and an emu; Australia has no single officially designated national animal."),
    (r"(?:how (?:do|can) i |what(?:'s| is) the (?:best )?way to )?(?:cook|make|prepare) (?:some )?(?:scrambled eggs|scrambled egg)", "Beat the eggs with a little salt, pour them into a lightly buttered pan on low heat, and stir gently until they are softly set."),
    (r"(?:how (?:do|can) i |what(?:'s| is) the (?:best )?way to )?(?:cook|make|prepare) (?:some )?(?:fried eggs|fried egg)", "Heat a little oil or butter in a pan over medium-low heat, crack in the egg, and cook until the white has set. Cook the yolk to your preference."),
    (r"(?:how (?:do|can) i |what(?:'s| is) the (?:best )?way to )?(?:cook|make|prepare) (?:some )?eggs", "You can boil, fry or scramble eggs. Which method would you like?"),
]


def tool_basic_knowledge(text: str):
    t = re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip(" .?!")
    t = re.sub(r"^(?:please |hey |could you tell me |can you tell me |do you know )", "", t)
    t = re.sub(r"^tell me (?:about )?", "what is ", t)
    t = re.sub(r"^how to cook (eggs|fried eggs|scrambled eggs)$", r"cook \1", t)
    t = re.sub(r"^whats\b", "what's", t)
    symbols = {"gold": "Au", "oxygen": "O", "iron": "Fe", "silver": "Ag", "hydrogen": "H", "carbon": "C", "nitrogen": "N", "helium": "He"}
    m = re.fullmatch(r"(?:what(?:'s| is)|name|give me) (?:the )?(?:chemical )?symbol (?:for|of|is) (gold|oxygen|iron|silver|hydrogen|carbon|nitrogen|helium)", t)
    if not m:
        m = re.fullmatch(r"what (?:chemical )?symbol is (gold|oxygen|iron|silver|hydrogen|carbon|nitrogen|helium)", t)
    if m:
        return f"The chemical symbol for {m.group(1)} is {symbols[m.group(1)]}."
    m = re.fullmatch(r"how many (seconds|minutes) (?:are there |are |is )?in (?:a |an |one )?(hour|day)", t)
    if m:
        unit, period = m.groups()
        number = {("seconds", "hour"): 3600, ("seconds", "day"): 86400,
                  ("minutes", "hour"): 60, ("minutes", "day"): 1440}[(unit, period)]
        return f"There are {number:,} {unit} in a {period}."
    # These specific historical question forms need to be phrased naturally.
    m = re.fullmatch(r"(?:when (?:was|did)|what year (?:was|did)|in what year (?:was|did)) (?:the )?(holocaust|world war (?:one|two|1|2)|ww[12]|the first world war|the second world war)(?: (?:start|begin|end|happen|take place))?", t)
    if m:
        subject = m.group(1)
        if subject == "holocaust":
            return "The Nazi persecution of Jews began in 1933; the systematic mass murder known as the Holocaust took place mainly from 1941 to 1945."
        return "World War One lasted from 1914 to 1918." if subject in ("world war one", "world war 1", "ww1", "the first world war") else "World War Two lasted from 1939 to 1945."
    m = re.fullmatch(r"(?:who wrote|who is the author of|who was the author of) (?:the )?(hamlet|romeo and juliet|macbeth|a christmas carol|pride and prejudice|1984|nineteen eighty-four|the hobbit|the lord of the rings)", t)
    if m:
        book = m.group(1)
        authors = {"hamlet": "William Shakespeare", "romeo and juliet": "William Shakespeare", "macbeth": "William Shakespeare", "a christmas carol": "Charles Dickens", "pride and prejudice": "Jane Austen", "1984": "George Orwell", "nineteen eighty-four": "George Orwell", "the hobbit": "J. R. R. Tolkien", "the lord of the rings": "J. R. R. Tolkien"}
        return f"{authors[book]} wrote {book.title()}."
    m = re.fullmatch(r"(?:who painted|who made the painting|who is the painter of) (?:the )?(mona lisa|starry night|the starry night)", t)
    if m:
        return "Leonardo da Vinci painted the Mona Lisa." if m.group(1) == "mona lisa" else "Vincent van Gogh painted The Starry Night."
    for pattern, answer in BASIC_QA:
        if answer and re.fullmatch(pattern, t):
            return answer
    return None

# Fixed and computable holidays only; no region-dependent public holiday guesses.
FIXED_HOLIDAYS = {
    "new year's day": (1, 1), "new years day": (1, 1), "new year": (1, 1),
    "australia day": (1, 26), "valentine's day": (2, 14), "valentines day": (2, 14),
    "april fools' day": (4, 1), "april fools day": (4, 1),
    "anzac day": (4, 25), "independence day": (7, 4), "fourth of july": (7, 4),
    "halloween": (10, 31), "christmas": (12, 25), "christmas day": (12, 25),
    "boxing day": (12, 26), "new year's eve": (12, 31), "new years eve": (12, 31),
}


def _easter_sunday(y: int) -> date:
    # Gregorian computus.
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f, g = (b + 8) // 25, (b - ((b + 8) // 25) + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = 1 + (h + l - 7 * m + 114) % 31
    return date(y, month, day)


def _holiday_date(name: str, year: int) -> date | None:
    if name in FIXED_HOLIDAYS:
        month, day = FIXED_HOLIDAYS[name]
        return date(year, month, day)
    if name in ("easter", "easter sunday"):
        return _easter_sunday(year)
    if name == "good friday":
        return _easter_sunday(year) - timedelta(days=2)
    if name == "easter monday":
        return _easter_sunday(year) + timedelta(days=1)
    if name in ("thanksgiving", "us thanksgiving", "american thanksgiving"):
        d = date(year, 11, 1)
        return d + timedelta(days=(3 - d.weekday()) % 7 + 21)
    if name in ("mother's day", "mothers day"):
        d = date(year, 5, 1)
        return d + timedelta(days=(6 - d.weekday()) % 7 + 7)  # Australia/US
    if name in ("father's day", "fathers day"):
        d = date(year, 9, 1)
        return d + timedelta(days=(6 - d.weekday()) % 7)  # Australia
    return None

_HOLIDAY_PATTERN = "|".join(re.escape(n) for n in sorted(set(FIXED_HOLIDAYS) | {
    "easter", "easter sunday", "good friday", "easter monday", "thanksgiving",
    "us thanksgiving", "american thanksgiving", "mother's day", "mothers day",
    "father's day", "fathers day"}, key=len, reverse=True))


def tool_holidays(text: str):
    t = re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip("?!., ")
    t = re.sub(r"^(?:please |hey |tell me |could you tell me |can you tell me |do you know )", "", t)
    t = re.sub(r"\baustralian (father's day|fathers day|mother's day|mothers day)\b", r"\1", t)
    # "Next holiday" is a fact lookup, not a question for the 0.8B model.
    # This calendar covers selected holidays/observances, NOT every local
    # statutory/public holiday. Public-holiday requests need a regional calendar.
    if "public holiday" not in t and re.fullmatch(
        r"(?:(?:what(?: is|'s)|which|when is|tell me) (?:the |our )?"
        r"(?:closest|nearest|next|soonest|first) (?:upcoming |coming )?"
        r"(?:holiday|celebration|observance)(?: coming up| after today| from today)?|"
        r"what (?:holiday|celebration|observance) (?:is )?(?:coming up next|comes next|is next)|"
        r"(?:when|what) (?:is|'s) the next holiday)", t):
        choices = ("new year's day", "australia day", "valentine's day",
                   "good friday", "easter", "easter monday", "anzac day",
                   "mother's day", "father's day", "halloween",
                   "christmas", "boxing day", "new year's eve")
        today = now().date()
        upcoming = [(_holiday_date(name, year), name)
                    for name in choices for year in (today.year, today.year + 1)]
        upcoming = [(d, name) for d, name in upcoming if d is not None and d >= today]
        if upcoming:
            day, name = min(upcoming)
            diff = (day - today).days
            suffix = "today" if diff == 0 else f"in {plural(diff, 'day')}"
            return (f"The next holiday or observance in my calendar is "
                    f"{name.title()} on {fmt_date(day)}, {suffix}.")
    # Narrow follow-up phrasing still returns a verifiable date.
    m_follow = re.fullmatch(r"(?:what about|how about|and) (?:the )?(" + _HOLIDAY_PATTERN + r")", t)
    if m_follow:
        name = m_follow.group(1)
        today = now().date()
        day = _holiday_date(name, today.year)
        if day is not None:
            if day < today:
                day = _holiday_date(name, today.year + 1)
            diff = (day - today).days
            return f"{name.title()} is {fmt_date(day)}, " + ("today." if diff == 0 else f"in {plural(diff, 'day')}.")
    m = re.fullmatch(r"(?:when(?: is|'s| does)|what(?:'s| is) (?:the )?date (?:of|for)|what day (?:is|does)|"
                     r"how (?:many days|long)(?: is it)? (?:until|till|to)|days? (?:until|till|to)) "
                     r"(?:the )?(" + _HOLIDAY_PATTERN + r")(?: (?:fall|happen|occur))?(?: (?:this|next) year| in (\d{4}))?", t)
    if not m:
        m = re.fullmatch(r"(?:on what date|what date|which date) (?:is|does) (?:the )?(" + _HOLIDAY_PATTERN + r")(?: (?:fall|happen|occur))?(?: (?:this|next) year| in (\d{4}))?", t)
    if not m:
        return None
    name, year_text = m.groups()
    today = now().date()
    year = int(year_text) if year_text else today.year
    if not (1583 <= year <= 9998):
        return None
    day = _holiday_date(name, year)
    if day is None:
        return None
    if not year_text and day < today:
        year += 1
        day = _holiday_date(name, year)
    diff = (day - today).days
    suffix = f", in {diff} days" if diff > 0 else ", today" if diff == 0 else f", {abs(diff)} days ago"
    return f"{name.title()} is {fmt_date(day)}{suffix}."


# ----------------------------------------------------------------------------
# Tool: school timetable ("how long is left of this class?", "what's next?")
# ----------------------------------------------------------------------------
# Built from the school calendar screenshots (week of 12-16 Oct 2026):
#   Monday has its own bell times; Tuesday and Thursday share bells;
#   Wednesday and Friday share bells. The timetable repeats every week.
# To change it (new term): python crutch.py --write-timetable, then edit
# timetable.json in the data folder. If that file exists it replaces this one.

_DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_SUBJECTS = {
    "home group": {"name": "Home Group", "room": "R8",
                   "aliases": ["home group", "homegroup", "home room", "homeroom"]},
    "science": {"name": "Science", "room": "UL", "aliases": ["science"]},
    "english": {"name": "Essential English", "room": "R28",
                "aliases": ["essential english", "english"]},
    "maths": {"name": "Maths", "room": "U3", "aliases": ["mathematics", "maths", "math"]},
    "music": {"name": "Music", "room": "MS", "aliases": ["music"]},
    "exploring": {"name": "Exploring Identities and Futures", "room": "R27",
                  "aliases": ["exploring identities and futures", "exploring identities",
                              "identities and futures", "exploring"]},
    "vines": {"name": "Vines and Wines", "room": "STA1",
              "aliases": ["vines and wines", "vines", "wines"]},
    "digital art": {"name": "Digital Art and Graphics", "room": "T5",
                    "aliases": ["digital art and graphics", "digital art", "graphics", "art"]},
    "student dev": {"name": "Student Development", "room": "R8",
                    "aliases": ["student development", "student dev"]},
}

_BELLS_MON = [("08:50", "09:00"), ("09:00", "09:40"), ("09:40", "10:20"), ("10:20", "11:00"),
              ("11:30", "12:10"), ("12:10", "12:50"), ("13:20", "14:00"), ("14:00", "14:40")]
_BELLS_TUE = [("08:50", "09:00"), ("09:00", "09:45"), ("09:45", "10:30"), ("10:30", "11:15"),
              ("11:45", "12:30"), ("12:30", "13:15"), ("13:45", "14:30"), ("14:30", "15:15")]
_BELLS_WED = [("08:50", "09:00"), ("09:00", "09:40"), ("09:40", "10:20"), ("10:20", "11:00"),
              ("11:30", "12:10"), ("12:10", "12:50"), ("12:50", "13:20"), ("13:50", "14:35"),
              ("14:35", "15:15")]
_WEEK = {
    "monday": (_BELLS_MON, ["home group", "science", "english", "maths", "music", "exploring",
                            "vines", "digital art"]),
    "tuesday": (_BELLS_TUE, ["home group", "exploring", "music", "music", "maths", "maths",
                             "science", "vines"]),
    "wednesday": (_BELLS_WED, ["home group", "english", "science", "science", "exploring",
                               "exploring", "student dev", "digital art", "music"]),
    "thursday": (_BELLS_TUE, ["home group", "digital art", "vines", "vines", "english",
                              "english", "science", "maths"]),
    "friday": (_BELLS_WED, ["home group", "music", "digital art", "digital art", "maths",
                            "vines", "student dev", "exploring", "english"]),
}


def _default_timetable():
    days = {}
    for day, (bells, seq) in _WEEK.items():
        assert len(bells) == len(seq), day
        days[day] = [[a, b, k] for (a, b), k in zip(bells, seq)]
    return {"subjects": _SUBJECTS, "days": days}


def write_timetable_template() -> Path:
    p = home_dir() / "timetable.json"
    p.write_text(json.dumps(_default_timetable(), indent=2), encoding="utf-8")
    return p


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _load_timetable():
    data = _default_timetable()
    p = home_dir() / "timetable.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if "subjects" in d and "days" in d:
                data = d
        except Exception:
            pass
    days = {}
    for name in _DAY_NAMES:
        days[name] = sorted((_hm(a), _hm(b), k) for a, b, k in data["days"].get(name, []))
    return data["subjects"], days


def _block_end(periods, i):
    """End of the class including back-to-back periods of the same subject."""
    end, j = periods[i][1], i + 1
    while j < len(periods) and periods[j][2] == periods[i][2] and periods[j][0] == end:
        end = periods[j][1]
        j += 1
    return end


def _next_block(periods, i):
    j = i + 1
    while j < len(periods) and periods[j][2] == periods[i][2] and periods[j][0] == periods[j - 1][1]:
        j += 1
    return j if j < len(periods) else None


def _block_starts(periods, key):
    out = []
    for i, (a, b, k) in enumerate(periods):
        if k != key:
            continue
        if i > 0 and periods[i - 1][2] == key and periods[i - 1][1] == a:
            continue
        out.append(a)
    return out


def _tt_name(subj, key):
    return subj.get(key, {}).get("name", key.title())


def _tt_room(subj, key):
    r = subj.get(key, {}).get("room")
    return f", room {r}" if r else ""


def _day_label(off: int, name: str) -> str:
    return "today" if off == 0 else "tomorrow" if off == 1 else ("next " + name.title() if off >= 7 else name.title())


def _first_after_today(days, wd):
    for off in range(1, 8):
        name = _DAY_NAMES[(wd + off) % 7]
        if days[name]:
            return off, name, days[name][0]
    return None


def _tt_state(periods, secs):
    if not periods:
        return "none", None
    if secs < periods[0][0] * 60:
        return "before", 0
    if secs >= periods[-1][1] * 60:
        return "after", None
    for i, (a, b, k) in enumerate(periods):
        if a * 60 <= secs < b * 60:
            return "in", i
    for i, (a, b, k) in enumerate(periods):
        if a * 60 > secs:
            return "break", i
    return "after", None


def _list_day(subj, periods, label):
    blocks = []
    for i, (a, b, k) in enumerate(periods):
        if i > 0 and periods[i - 1][2] == k and periods[i - 1][1] == a:
            continue
        blocks.append(f"{_tt_name(subj, k)} at {fmt_time(a)}")
    return f"{label}: " + ", ".join(blocks) + f". School finishes at {fmt_time(periods[-1][1])}."


def tool_timetable(text: str):
    import math
    t = re.sub(r"\s+", " ", text.lower().replace("\u2019", "'")).strip().rstrip("?!. ")
    if not t:
        return None
    subj, days = _load_timetable()
    n = now()
    wd = n.weekday()
    secs = n.hour * 3600 + n.minute * 60 + n.second
    periods = days[_DAY_NAMES[wd]]
    state, idx = _tt_state(periods, secs)
    pre = r"(?:(?:so|ok|okay|hey)\s+)*"

    alias = {}
    for key, info in subj.items():
        for a_ in info.get("aliases", []):
            alias[a_.lower()] = key
    alias_re = "|".join(re.escape(a_) for a_ in sorted(alias, key=len, reverse=True))
    has_subject = re.search(r"\b(" + alias_re + r")\b", t)

    def upcoming(count):
        """(day offset, day name, next `count` classes). Back-to-back periods of one subject count once."""
        if state in ("before", "break"):
            plist, i, off, name = periods, idx, 0, _DAY_NAMES[wd]
        elif state == "in":
            plist, i, off, name = periods, _next_block(periods, idx), 0, _DAY_NAMES[wd]
        else:
            nxt = _first_after_today(days, wd)
            if not nxt:
                return None
            off, name, _first = nxt
            plist, i = days[name], 0
        out = []
        while i is not None and i < len(plist) and len(out) < count:
            out.append(plist[i])
            i = _next_block(plist, i)
        return off, name, out

    def next_class_text(count=1):
        r = upcoming(count)
        if r is None:
            return "There are no classes on your timetable."
        off, name, blocks = r
        if not blocks:
            return f"That is your last class today. School finishes at {fmt_time(periods[-1][1])}."
        parts = []
        for k, (a_, b_, key) in enumerate(blocks):
            when = f" {_day_label(off, name)}" if (k == 0 and off) else ""
            parts.append(f"{_tt_name(subj, key)}{when} at {fmt_time(a_)}{_tt_room(subj, key)}")
        return "Next is " + ", then ".join(parts) + "."

    # Natural classroom follow-ups, without matching "next train", "next week" etc.
    if re.fullmatch(pre + r"(?:what do i (?:got|have) (?:after|next)|what(?:'s| is) (?:my )?(?:next|following) (?:class|lesson|subject|period)|"
                    r"(?:which|what) lesson (?:comes|is) (?:next|after this)|what(?:'s| is) after (?:this|my class)|"
                    r"what (?:class|lesson|subject) (?:do i have|have i got) after this|what (?:class|lesson) is up next|"
                    r"what have i got (?:next|after this)|(?:next|following) (?:class|lesson)|"
                    r"what do i have (?:next|after this) and (?:then )?(?:after|next))", t):
        count = 2 if re.search(r"and (?:then )?(?:after|next)", t) else 1
        return next_class_text(count)
    if re.fullmatch(pre + r"(?:what (?:lesson|class|subject) have i got|what (?:lesson|class|subject) is (?:it|on) now|"
                    r"what am i studying right now|what subject am i in|what do i have (?:right )?now)", t):
        if state == "in":
            a_, b_, k = periods[idx]
            return f"You are in {_tt_name(subj, k)}{_tt_room(subj, k)}, until {fmt_time(b_)}."
        if state in ("before", "break"):
            return next_class_text()
        return "School is finished for today." if state == "after" else "There are no classes today."
    if re.fullmatch(pre + r"(?:what classes do i have (?:today|tomorrow)|show (?:me )?my (?:classes|lessons|schedule) (?:for )?(?:today|tomorrow)|"
                    r"what(?:'s| is) my (?:school )?schedule (?:for )?(?:today|tomorrow))", t):
        off = 1 if "tomorrow" in t else 0
        name = _DAY_NAMES[(wd + off) % 7]
        return _list_day(subj, days[name], "Tomorrow" if off else "Today") if days[name] else f"No classes {'tomorrow' if off else 'today'}."

    # ---- how long is left of this class ---------------------------------
    bare_left = re.fullmatch(pre + r"(?:how (?:long|much time|many minutes)(?: is)?(?: there)? (?:left|remaining)"
                             r"|how much longer|time left|time remaining)", t)
    class_left = re.search(r"\b(class|lesson|period)\b", t) and (
        re.search(r"\bhow\b.*\b(left|remaining|longer)\b", t)
        or (re.search(r"\b(this|current|my) (class|lesson|period)\b", t) and (
            re.search(r"\bwhen\b.*\b(end|ends|finish|finishes|over|done)\b", t)
            or re.search(r"\bhow long\b.*\b(until|till|before)\b.*\b(end|ends|finish|finishes|over|done)\b", t))))
    if bare_left and _TIMERS:
        return None  # a timer is running; let the timer tool answer
    if bare_left or class_left:
        if state == "in":
            a_, b_, k = periods[idx]
            mins = math.ceil((b_ * 60 - secs) / 60)
            out = f"{plural(mins, 'minute')} left in {_tt_name(subj, k)}."
            be = _block_end(periods, idx)
            if be > b_:
                out += f" It continues into the next period until {fmt_time(be)}."
            return out
        if state == "break":
            a_, b_, k = periods[idx]
            mins = math.ceil((a_ * 60 - secs) / 60)
            return (f"You are on a break right now. {_tt_name(subj, k)} starts in "
                    f"{fmt_dur(mins)}, at {fmt_time(a_)}{_tt_room(subj, k)}.")
        if state == "before":
            a_, b_, k = periods[0]
            return f"School has not started. {_tt_name(subj, k)} is at {fmt_time(a_)}."
        if state == "after":
            return "School is finished for today."
        return "There are no classes today."

    # ---- what do I have next (optionally "and then after") ---------------
    next_strict = (
        re.fullmatch(pre + r"what(?:'s| is| do i have| have i got| do we have| class is| comes| am i doing)?"
                     r"\s*(?:my |the )?(?:next|after this)(?: class| lesson| period| one)?(?: after this)?", t)
        or re.fullmatch(pre + r"(?:what(?:'s| is)|which)?\s*(?:my )?next (?:class|lesson|period)", t)
        or re.fullmatch(pre + r"what(?:'s| is) after this(?: one)?", t))
    next_loose = (re.search(r"\bnext\b", t) and re.search(r"\b(what|which|when)\b", t)
                  and re.search(r"\b(class|lesson|period|do i have|have i got|do we have|am i doing)\b", t)
                  and not has_subject
                  and not re.search(r"\b(list|note|notes|shopping|reminder|timer|week|month|year|holiday|"
                                    r"weekend|train|bus)\b", t))
    if next_strict or next_loose:
        count = 2 if re.search(r"\b(then|after that|and after|after)\b", t.replace("after this", "")) else 1
        if state == "none":
            return "There are no classes today. " + next_class_text(count)
        return next_class_text(count)

    # ---- what class am I in ---------------------------------------------
    if re.fullmatch(pre + r"(?:what (?:class|lesson|period) (?:is this|am i in|is it|are we in|do i have(?: now| right now)?)"
                    r"|what class am i in(?: right now| now)?|what(?:'s| is) (?:on|happening)(?: right)? now"
                    r"|what do i have(?: right)? now|what are we doing(?: right)? now|which class am i in)", t):
        if state == "in":
            a_, b_, k = periods[idx]
            return f"You are in {_tt_name(subj, k)}{_tt_room(subj, k)}, until {fmt_time(b_)}."
        if state == "break":
            return "You are on a break. " + next_class_text()
        if state == "before":
            return "School has not started. " + next_class_text()
        if state == "after":
            return "School is finished for today."
        return "There are no classes today."

    # ---- when does school finish ----------------------------------------
    if (re.search(r"\bwhen\b.*\b(school|classes|lessons|the day)\b.*\b(finish|finishes|end|ends|over|done|out)\b", t)
            or re.search(r"\bwhat time\b.*\b(school|classes|lessons)\b.*\b(finish|finishes|end|ends|out)\b", t)
            or re.fullmatch(r"(?:what time do (?:i|we) (?:finish|get out)|when do (?:i|we) (?:finish|get out|go home))(?: today)?", t)):
        if not periods:
            return "There is no school today."
        end = periods[-1][1]
        if secs < end * 60:
            return (f"School finishes at {fmt_time(end)} today, in "
                    f"{fmt_dur(math.ceil((end * 60 - secs) / 60))}.")
        return f"School finished at {fmt_time(end)} today."

    # ---- timetable for a day --------------------------------------------
    bare_day = re.fullmatch(pre + r"(?:what(?:'s| is)? )?(?:on )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
                            r"(?:'s)?(?: timetable| classes| lessons)?", t)
    daym = bare_day or re.search(r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:'s)?\b", t)
    list_intent = (bare_day or re.search(r"\b(timetable|classes|lessons)\b", t)
                   or re.search(r"\bwhat\b.*\b(do i have|have i got|do we have|am i doing|are we doing)\b", t)
                   or re.search(r"\bwhat(?:'s| is) on\b", t))
    if daym and list_intent:
        w = daym.group(1)
        off = 0 if w == "today" else 1 if w == "tomorrow" else (_DAY_NAMES.index(w) - wd) % 7
        name = _DAY_NAMES[(wd + off) % 7]
        d0 = n.date() + timedelta(days=off)
        if bare_day:
            label = f"{name.title()}, {d0:%B} {d0.day}"
        else:
            label = "Today" if w == "today" else "Tomorrow" if w == "tomorrow" else name.title()
        if not days[name]:
            return f"No classes on {label}." if bare_day else (
                "No classes today." if label == "Today" else "No classes tomorrow." if label == "Tomorrow"
                else f"No classes on {name.title()}.")
        return _list_day(subj, days[name], label)
    if re.fullmatch(pre + r"(?:what(?:'s| is) my |show my |read my )?(?:timetable|classes|lessons)(?: today)?", t):
        if not periods:
            return "There are no classes today."
        return _list_day(subj, periods, "Today")

    # ---- a specific subject ---------------------------------------------
    sm = re.fullmatch(r"(?:when|what time)\b.*?\b(" + alias_re + r")\b"
                      r"(?: class| lesson| period| starts?| begins?| next| today| tomorrow| again| this week)*", t)
    um = re.fullmatch(r"how (?:long|many minutes|much time)(?: is it)? (?:until|till|before)\b.*?\b(" + alias_re + r")\b"
                      r"(?: class| lesson| period| starts?| begins?)*", t)
    if sm or um:
        key = alias[(sm or um).group(1)]
        name = _tt_name(subj, key)
        now_min = secs / 60
        if um or (sm and re.search(r"\bnext\b", t)):
            for off in range(0, 8):
                dname = _DAY_NAMES[(wd + off) % 7]
                for start in _block_starts(days[dname], key):
                    if off * 1440 + start > now_min:
                        diff = off * 1440 + start - now_min
                        if um and diff <= 720:
                            return f"{name} starts in {fmt_dur(math.ceil(diff))}, at {fmt_time(start)}{_tt_room(subj, key)}."
                        return (f"Your next {name} is {_day_label(off, dname)} at "
                                f"{fmt_time(start)}{_tt_room(subj, key)}.")
            return f"{name} isn't on your timetable."
        parts = []
        for dname in _DAY_NAMES:
            for start in _block_starts(days[dname], key):
                parts.append(f"{dname.title()} at {fmt_time(start)}")
        return f"{name} is on " + ", ".join(parts) + "." if parts else f"{name} isn't on your timetable."
    return None


# ----------------------------------------------------------------------------
# Tool: small talk and questions about Aria herself (instant canned replies)
# ----------------------------------------------------------------------------
# Edit these to change how she talks. First matching line wins.

SMALLTALK = [
    (r"(?:hello|hi|hey|hiya|howdy|yo|good (?:morning|afternoon|evening|night))(?: there)?",
     "Hello. What can I do for you?"),
    (r"aria|yes aria|are you there|are you listening|you there|aria are you there", "Yes?"),
    (r"how are you(?: doing| today| going| feeling)?|how'?s it going|how have you been|are you ok(?:ay)?|everything okay",
     "I'm good, thanks. What can I do for you?"),
    (r"(?:thanks|thank you|thank you very much|thanks a lot|cheers|ta|thx|much appreciated|appreciate it|thankyou)(?: so much)?",
     "You're welcome."),
    (r"(?:you(?:'re| are) )?dis+miss?ed|that'?s all|that is all|never ?mind",
     "Okay. Just say my name if you need me."),
    (r"what(?:'s| is) your name|who are you|what are you|what should i call you|tell me your name|your name|introduce yourself",
     "I'm Aria, a small offline assistant that runs on your device."),
    (r"who (?:made|built|created|programmed) you",
     "I was built as a personal project, to work without needing internet."),
    (r"what can you do|what do you do|help|help me|show me your features|list your abilities|what are your (?:features|tools|skills|abilities)|what can i ask you|how can you help me",
     "I can do maths, dates and times, travel timing, unit conversions, timers, lists and notes, "
     "and your class timetable. For anything else I use a small language model, which can be wrong."),
    (r"(?:did i|do you|have you been)(?: equip(?:ped)? you with| give you| have| got)?(?: any| a)? "
     r"(?:persistent |long.?term )?memory|do you remember (?:things|stuff|what i (?:said|told you)|me)",
     "I keep your lists, notes and timetable on this device, so those stay. "
     "I don't remember our conversations, though."),
]


def tool_smalltalk(text: str):
    t = re.sub(r"\s+", " ", text.lower().replace("\u2019", "'")).strip().rstrip("?!., ")
    for pat, reply in SMALLTALK:
        if re.fullmatch(pat, t):
            return reply
    return None


# ----------------------------------------------------------------------------
# Tool: several sums in one message ("18 x 2, 18 x 3, ...")
# ----------------------------------------------------------------------------

def _speak_expr(e: str) -> str:
    e = re.sub(r"(?<=\d)\s*[x×*]\s*(?=\d)", " times ", e.strip())
    for a, b in (("/", " divided by "), ("÷", " divided by "), ("+", " plus "), ("^", " to the power of ")):
        e = e.replace(a, b)
    e = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " minus ", e)
    return re.sub(r"\s+", " ", e).strip()


def tool_multicalc(text: str):
    t = text.lower().strip().rstrip("?!. ")
    t = re.sub(r"^(?:(?:please|hey|ok)\s+)*(?:tell me|give me|what are|what is|calculate|compute|work out|solve)"
               r"(?: the)?(?: answers?(?: to)?| results?(?: of)?)?\s*:?\s*", "", t)
    pieces = [p.strip() for p in re.split(r"\s*(?:,|;|\band\b|&)\s*", t) if p.strip()]
    if len(pieces) < 2 or len(pieces) > 20:
        return None
    out = []
    for p in pieces:
        r = tool_calc(p)
        m = re.fullmatch(r"The answer is (.+)\.", r or "")
        if not m:
            return None
        out.append(f"{_speak_expr(p)} is {m.group(1)}")
    return ", ".join(out) + "."


# ----------------------------------------------------------------------------
# Tools: saved notes - answer "what is the gate code?" and save "gate code is 4473"
# (these run last so they never steal a message from a real tool)
# ----------------------------------------------------------------------------

_NOTE_STOP = {"the", "my", "our", "a", "an", "of", "is", "was", "are", "me", "you", "to", "for"}


def tool_note_lookup(text: str):
    t = re.sub(r"\s+", " ", text.lower().replace("\u2019", "'")).strip().rstrip("?!. ")
    m = re.match(r"^(?:what(?:'s| is| was| are)|whats|tell me|remind me(?: what)?(?: is)?|do you know|"
                 r"what do you know about|can you recall|can you remind me what|what did i say about)\s+(.+)$", t)
    if not m:
        return None
    words = [w for w in re.findall(r"[a-z0-9]+", m.group(1)) if w not in _NOTE_STOP]
    if not words:
        return None
    for note in reversed(_load()["notes"]):
        low = note["text"].lower()
        if all(w in low for w in words):
            return f"You told me: {note['text']}."
    return None


_NOT_STATEMENT = re.compile(r"^(?:what|who|when|where|why|how|which|is|are|do|does|did|can|could|will|would|"
                            r"tell|show|set|add|remove|remember|i|it|that|this|there|he|she|they|we|you)\b")


def tool_note_capture(text: str):
    t = re.sub(r"\s+", " ", text.strip().lower().replace("\u2019", "'")).rstrip(".!")
    if "?" in t or _NOT_STATEMENT.match(t):
        return None
    m = re.fullmatch(r"(?:my |the |our )?([a-z][a-z' ]{1,30}?) (?:is|are|=|:) ([^?]*\d[^?]*)", t)
    if not m or len(m.group(2).split()) > 8:
        return None
    key, value = m.group(1).strip(), m.group(2).strip()
    d = _load()
    d["notes"].append({"text": f"{key} is {value}", "when": now().isoformat(timespec="minutes")})
    _save(d)
    return f"Got it. {key.capitalize()} is {value}."


# ----------------------------------------------------------------------------
# Wake-name stripping: "Hey Aria, what's the time?" -> "what's the time?"
# ----------------------------------------------------------------------------

_ADDR_RE = re.compile(r"^\s*(?:(?:hey|hi|hello|ok|okay|yo|good (?:morning|afternoon|evening|night))\s*,?\s*)*"
                      r"aria\b\s*[,.!:;-]*\s*", re.I)


def strip_address(t: str) -> str:
    s = _ADDR_RE.sub("", t, count=1)
    s = re.sub(r"[,\s]+aria\s*([?.!]*)$", lambda m: m.group(1), s, flags=re.I)
    return s.strip() or t


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------

TOOLS = [
    ("smalltalk", tool_smalltalk),
    ("timer", tool_timer),
    ("lists", tool_lists_notes),
    ("timetable", tool_timetable),
    ("schedule", tool_schedule),
    ("running_total", tool_running_total),
    ("datetime", tool_datetime),
    ("holidays", tool_holidays),
    ("units", tool_units),
    ("percent", tool_percent),
    ("multicalc", tool_multicalc),
    ("calculator", tool_calc),
    ("facts", tool_facts),
    ("geography", tool_geography),
    ("knowledge", tool_basic_knowledge),
    ("notes_lookup", tool_note_lookup),
    ("notes_capture", tool_note_capture),
]


def route(text: str):
    t = strip_address(text.strip())
    t = re.sub(r"^(?:(?:please|hey|hi|okay|ok|so)\s*[,!]?\s+){1,3}", "", t, flags=re.I)
    t = re.sub(r"^(?:can|could|would) you (?:please )?(?:tell me|let me know) (?:what|when|where|who|how|which)\b",
               lambda m: re.search(r"(what|when|where|who|how|which)$", m.group(), re.I).group(), t, flags=re.I)
    t = re.sub(r"^(?:can|could|would) you (?:please )?(?=(?:set|start|stop|cancel|add|remove|delete|convert|calculate|show|read|remind)\b)", "", t, flags=re.I)
    t = re.sub(r"^what the time is\??$", "what time is it", t, flags=re.I)
    t = re.sub(r"^(?:can|could|would) you (?:please )?tell me (?:the )?(time|date)\??$", r"what is the \1", t, flags=re.I)
    if not t:
        return None
    for name, fn in TOOLS:
        try:
            out = fn(t)
        except Exception as e:  # a buggy tool must never crash the assistant
            if os.environ.get("CRUTCH_DEBUG"):
                print(f"[debug] {name} failed: {e!r}", file=sys.stderr)
            out = None
        if out:
            return out, name
    # Expand explicit sentence phrasing only *after* existing deterministic
    # routes miss. Hassil is a recognizer, not a second set of answer engines.
    try:
        import aria_hassil_router
        expanded = aria_hassil_router.route(t, dict(TOOLS), sys.modules[__name__])
        if expanded:
            return expanded
    except (ImportError, OSError) as e:
        # Standalone crutch.py still works if optional Aria files are absent.
        if os.environ.get("CRUTCH_DEBUG"):
            print(f"[debug] Hassil unavailable: {e!r}", file=sys.stderr)
    except Exception as e:
        if os.environ.get("CRUTCH_DEBUG"):
            print(f"[debug] Hassil recognition failed: {e!r}", file=sys.stderr)
    return None


# ----------------------------------------------------------------------------
# Model fallback
# ----------------------------------------------------------------------------

class ModelUnavailable(Exception):
    pass


def plain(text: str, *, max_sentences: int | None = None) -> str:
    """Remove non-voice formatting; sentence limit can be overridden per reply.

    max_sentences=None uses the configured short-answer default.
    max_sentences=0 preserves all sentences (for explicit detailed requests).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:rightarrow|to)\b", "to", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"[$*#`_]", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    limit = MAX_SENTENCES if max_sentences is None else max_sentences
    if limit > 0:
        text = " ".join(re.split(r"(?<=[.!?])\s+", text)[:limit])
    return text


def ask_model(text: str, system: str | None = None, *, detailed: bool = False,
              history: list[dict[str, str]] | None = None) -> str:
    """Ask the model with separate speaker roles and request-scoped length.

    Direct calls remain compatible: history=None sends just a user message.
    The bridge supplies up to three preceding messages (including Crutch
    answers as assistant messages), never a flattened pseudo-conversation.
    """
    if system is None:
        system = DETAILED_SYSTEM_PROMPT if detailed else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}]
    for item in (history or [])[-3:]:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            messages.append({"role": item["role"], "content": content.strip()})
    messages.append({"role": "user", "content": text + (" /no_think" if NO_THINK else "")})
    payload = {"model": MODEL_NAME, "temperature": TEMPERATURE,
               "max_tokens": DETAIL_MAX_TOKENS if detailed else MAX_TOKENS,
               "stream": True, "messages": messages}
    if NO_THINK:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(MODEL_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})

    req = urllib.request.Request(MODEL_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})

    full_response = []
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                line = line.decode('utf-8').strip()
                if not line.startswith("data:"):
                    continue
                if line == "data: [DONE]":
                    break
                try:
                    chunk = json.loads(line[5:].strip())
                    token = chunk["choices"][0]["delta"].get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        full_response.append(token)
                except Exception:
                    continue
        print()
        return "".join(full_response)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        raise ModelUnavailable(str(e))


def respond(text: str, use_model: bool = True):
    """Returns (reply, source). source is a tool name, 'model' or 'none'."""
    r = route(text)
    if r:
        return r
    try:
        with open(home_dir() / "unmatched.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{text}\n")
    except OSError:
        pass
    if not use_model:
        return "No tool matched that.", "none"
    try:
        return ask_model(strip_address(text.strip()), detailed=wants_detailed_reply(text)), "model"
    except ModelUnavailable:
        return ("No tool matched that, and the model server isn't reachable. "
                "Start LM Studio or llama-server, or check CRUTCH_URL."), "none"


# ----------------------------------------------------------------------------
# Self-test and compare
# ----------------------------------------------------------------------------

_SCHEDULE_Q = ("I have a meeting at 4:10 PM and an appointment at 7:30 PM, and I need to be home by "
               "9:00 PM. Each event lasts 1 hour. The meeting, the appointment and my home are each "
               "15 kilometers from one another, and I travel at an average of 30 kilometers per hour. "
               "What is the latest time I can leave each event to be on time?")

COMPARE_PROMPTS = [
    "What is the capital of France?",
    "Give me three steps to boil an egg, numbered, one line each.",
    "If I have 12 apples and give away 5, then buy 3, how many do I have?",
    "A trip takes 30 minutes. I need to arrive by 7:30 PM. What is the latest time I can leave?",
    _SCHEDULE_Q,
    "What is 15 percent of 48?",
    "Convert 72 Fahrenheit to Celsius.",
    "How many days until December 25?",
    "What's the weather in Paris right now?",
    "Tell me a short joke.",
]

# (message, substring that must be in the reply, expected source)
TESTS = [
    (_SCHEDULE_Q, "Leave the meeting by 7:00 PM. Leave the appointment by 8:30 PM", "schedule"),
    ("A trip takes 30 minutes. I need to arrive by 7:30 PM. What is the latest time I can leave?", "leave by 7:00 PM", "schedule"),
    ("A trip is 15 km at 30 km/h. I need to arrive by 7:30 PM. What's the latest I can leave?", "leave by 7:00 PM", "schedule"),
    ("If I leave at 6 PM and the trip is 40 minutes, when do I arrive?", "6:40 PM", "schedule"),
    ("How long does 15 km at 30 km/h take?", "30 minutes", "schedule"),
    ("The meeting at 4:30 PM and the dinner at 8:00 PM, home by 10:00 PM, each lasts 2 hours, "
     "trips take 20 minutes", "Leave the meeting by 7:40 PM", "schedule"),
    ("7:30 PM minus 30 minutes", "7:00 PM", "datetime"),
    ("what time is 45 minutes after 6:15 pm", "7:00 PM", "datetime"),
    ("how long from 4:10 pm to 7:30 pm", "3 hours 20 minutes", "datetime"),
    ("what time is it", "It is 10:00 AM.", "datetime"),
    ("what's the date today", "Tuesday, October 13, 2026", "datetime"),
    ("what day of the week is 2026-12-25", "Friday", "datetime"),
    ("how many days until December 25", "73 days until Friday, December 25, 2026, which is 10 weeks and 3 days", "datetime"),
    ("what time will it be in 90 minutes", "11:30 AM", "datetime"),
    ("what date is 10 days from now", "Friday, October 23, 2026", "datetime"),
    ("What is 15 percent of 48?", "7.2", "percent"),
    ("20% off 80", "64", "percent"),
    ("split 86 between 3", "28.67", "percent"),
    ("what percent is 5 of 20", "25 percent", "percent"),
    ("12 - 5 + 3", "10", "calculator"),
    ("what's 15 times 4 plus 2", "62", "calculator"),
    ("square root of 144", "12", "calculator"),
    ("If I have 12 apples and give away 5, then buy 3, how many do I have?", "10", "running_total"),
    ("convert 5 miles to km", "8.05", "units"),
    ("72 f to c", "22.22", "units"),
    ("how many cups in a liter", "1 liter is 4.23 US cups.", "units"),
    ("what's 2 lbs in kg", "0.91", "units"),
    ("how long do I boil an egg", "6 minutes", "facts"),
    ("add milk and eggs to the shopping list", "Added milk, eggs", "lists"),
    ("what's on my shopping list", "milk, eggs", "lists"),
    ("remove milk from the shopping list", "Removed milk", "lists"),
    ("remember that the gate code is 4172", "remember", "lists"),
    ("what do you remember about the gate code", "4172", "lists"),
    ("set a timer for 2 minutes", "Timer set for 2 minutes.", "timer"),
    ("how much time is left on the timer", "left", "timer"),
    ("cancel the timer", "Cancelled 1 timer.", "timer"),
]

_D = datetime
TT_TESTS = [
    # Tuesday 13 Oct, 10:00 = second half of Music's first period; Music is a double (9:45-11:15)
    ("How long is left of this class?", "30 minutes left in Music. It continues into the next period until 11:15 AM.", "timetable", _D(2026, 10, 13, 10, 0)),
    ("What do I have next?", "Next is Maths at 11:45 AM, room U3.", "timetable", _D(2026, 10, 13, 10, 0)),
    ("when does this class end", "30 minutes left in Music", "timetable", _D(2026, 10, 13, 10, 0)),
    ("how long is left of this class", "25 minutes left in Maths.", "timetable", _D(2026, 10, 13, 12, 5)),
    ("how long is left of this class", "on a break", "timetable", _D(2026, 10, 13, 11, 20)),
    ("what do i have next", "Next is Maths at 11:45 AM", "timetable", _D(2026, 10, 13, 11, 20)),
    ("what do i have next", "Next is Home Group at 8:50 AM", "timetable", _D(2026, 10, 13, 8, 0)),
    ("what do i have next", "Next is Exploring Identities and Futures at 9:00 AM", "timetable", _D(2026, 10, 13, 8, 55)),
    ("what class am i in", "You are in Home Group, room R8, until 9:00 AM.", "timetable", _D(2026, 10, 13, 8, 55)),
    # Monday has its own bells (ends 2:40 PM)
    ("how long is left of this class?", "25 minutes left in Music.", "timetable", _D(2026, 10, 12, 11, 45)),
    ("what do I have next?", "last class today", "timetable", _D(2026, 10, 12, 14, 10)),
    ("when does school finish", "School finishes at 2:40 PM today, in 4 hours 40 minutes.", "timetable", _D(2026, 10, 12, 10, 0)),
    # Wednesday: 30-minute Student Development before lunch
    ("how long is left of this class?", "20 minutes left in Student Development.", "timetable", _D(2026, 10, 14, 13, 0)),
    ("what's next", "Next is Digital Art and Graphics at 1:50 PM, room T5.", "timetable", _D(2026, 10, 14, 13, 0)),
    # Thursday shares Tuesday's bells
    ("what do I have next?", "Next is Essential English at 11:45 AM", "timetable", _D(2026, 10, 15, 9, 50)),
    # Friday shares Wednesday's bells; double Digital Art
    ("how long is left of this class?", "30 minutes left in Digital Art and Graphics. It continues into the next period until 11:00 AM.", "timetable", _D(2026, 10, 16, 9, 50)),
    # after school / weekend roll over to the next school day
    ("what do i have next", "Next is Home Group Monday at 8:50 AM", "timetable", _D(2026, 10, 16, 15, 20)),
    ("what do i have next", "Next is Home Group Monday at 8:50 AM", "timetable", _D(2026, 10, 17, 12, 0)),
    ("what do i have next", "Next is Home Group tomorrow at 8:50 AM", "timetable", _D(2026, 10, 12, 17, 0)),
    ("how long is left of this class?", "School is finished for today.", "timetable", _D(2026, 10, 13, 16, 0)),
    ("how long is left of this class?", "There are no classes today.", "timetable", _D(2026, 10, 17, 10, 0)),
    # day lists and subject lookups
    ("what do I have tomorrow", "Tomorrow: Home Group at 8:50 AM, Essential English at 9:00 AM, Science at 9:40 AM, "
     "Exploring Identities and Futures at 11:30 AM, Student Development at 12:50 PM, "
     "Digital Art and Graphics at 1:50 PM, Music at 2:35 PM. School finishes at 3:15 PM.", "timetable", _D(2026, 10, 13, 10, 0)),
    ("what's my timetable on friday", "Friday: Home Group at 8:50 AM, Music at 9:00 AM", "timetable", _D(2026, 10, 13, 10, 0)),
    ("when is music", "Music is on Monday at 11:30 AM, Tuesday at 9:45 AM, Wednesday at 2:35 PM, Friday at 9:00 AM.", "timetable", _D(2026, 10, 13, 10, 0)),
    ("when is my next maths class", "Your next Maths is today at 11:45 AM, room U3.", "timetable", _D(2026, 10, 13, 10, 0)),
    ("how long until science", "Science starts in 3 hours 45 minutes, at 1:45 PM, room UL.", "timetable", _D(2026, 10, 13, 10, 0)),
    ("when is exploring identities and futures", "Exploring Identities and Futures is on Monday at 12:10 PM", "timetable", _D(2026, 10, 13, 10, 0)),
    # a running timer takes priority over the class answer for a bare "how long is left"
    ("set a timer for 5 minutes", "Timer set", "timer", _D(2026, 10, 13, 10, 0)),
    ("how much time is left", "Timer", "timer", _D(2026, 10, 13, 10, 0)),
    ("cancel the timer", "Cancelled 1 timer.", "timer", _D(2026, 10, 13, 10, 0)),
]
# Cases taken from a real unmatched.log (what the user actually typed)
LOG_TESTS = [
    ("Hello", "Hello. What can I do for you?", "smalltalk"),
    ("How are you?", "I'm good", "smalltalk"),
    ("Thank you", "You're welcome.", "smalltalk"),
    ("Aria", "Yes?", "smalltalk"),
    ("You are dissmissed.", "Just say my name", "smalltalk"),
    ("What is your name?", "I'm Aria", "smalltalk"),
    ("Did I equip you with persistent memory?", "lists, notes and timetable", "smalltalk"),
    ("What is the date tomorrow?", "Tomorrow is Wednesday, October 14, 2026.", "datetime"),
    ("What is the time & date", "It is 10:00 AM on Tuesday, October 13, 2026.", "datetime"),
    ("Good evening aria, what's the time?", "It is 10:00 AM.", "datetime"),
    ("Tell me the answers to: 18 x 2, 18 x 3, 18 x 4, 18 x 5, 18 x 6, 18 x 7, 18 x 8, 18 x 9 and 18 x 10",
     "18 times 2 is 36, 18 times 3 is 54, 18 times 4 is 72, 18 times 5 is 90, 18 times 6 is 108, "
     "18 times 7 is 126, 18 times 8 is 144, 18 times 9 is 162, 18 times 10 is 180.", "multicalc"),
    ("When is my next lesson?", "Next is Maths at 11:45 AM, room U3.", "timetable"),
    ("What class will I have next?", "Next is Maths at 11:45 AM", "timetable"),
    ("What class do I have next?", "Next is Maths at 11:45 AM", "timetable"),
    ("what class do i have next", "Next is Maths at 11:45 AM", "timetable"),
    ("Aria, What do I have next", "Next is Maths at 11:45 AM", "timetable"),
    ("What do I have next, and then after", "Next is Maths at 11:45 AM, room U3, then Science at 1:45 PM, room UL.", "timetable"),
    ("What class will I have next?", "Next is Home Group at 8:50 AM", "timetable", _D(2026, 10, 14, 1, 44)),
    ("friday", "Friday, October 16: Home Group at 8:50 AM, Music at 9:00 AM", "timetable"),
    ("add milk to the list", "Added milk to your shopping list.", "lists"),
    ("Remove milk from list", "Removed milk from your shopping list.", "lists"),
    ("add milk and bread to the shopping list", "Added milk, bread", "lists"),
    ("Remove milk", "Removed milk from your shopping list.", "lists"),
    ("gate code is 4473", "Got it. Gate code is 4473.", "notes_capture"),
    ("What is the gate code", "4473", "notes_lookup"),
]

MUST_FALL_THROUGH = [
    "Tell me a joke.", "What is a linked list?",
    "I have a dentist appointment tomorrow.", "call 555-1234",
    "how long is left of this movie", "when is the science fair", "what's the next train",
    "what time do I finish work", "when does the last class of the year end", "when is the music video out",
    "how many days left in the month", "what is a class in python",
    "Do you know much about the game minecraft?", "e", "s", "67", "what's next on my shopping list",
    "what do I have next week", "Tell me about Minecraft", "who is the president",
]


# Wider trigger coverage without catching unrelated chats; new requirements are
# also tested against every state/country capital below.
EXPANDED_TESTS = [
    ("What's the capital of France?", "Paris", "geography"),
    ("What's the top speed of an ostrich?", "70 kilometres", "knowledge"),
    ("Who wrote Hamlet?", "William Shakespeare", "knowledge"),
    ("Capital of Brazil", "Brasília", "geography"),
    ("Please tell me the capital of the USA", "Washington", "geography"),
    ("What is the capital of the state of Texas?", "Austin", "geography"),
    ("What is the capital of Georgia the country?", "Tbilisi", "geography"),
    ("What is the capital of Georgia?", "Atlanta", "geography"),
    ("What is the capital of Equatorial Guinea?", "Ciudad de la Paz", "geography"),
    ("What is the capital of Indonesia?", "Jakarta", "geography"),
    ("What does NSW stand for?", "New South Wales", "geography"),
    ("What's the state abbreviation for California?", "CA", "geography"),
    ("Which state has Austin as its capital?", "Texas", "geography"),
    ("How many states are in the United States?", "50 states", "geography"),
    ("How many states in Australia?", "six states", "geography"),
    ("Name all Australian states", "Tasmania", "geography"),
    ("List all 50 states", "Wyoming", "geography"),
    ("What is the largest state in the US?", "Alaska", "geography"),
    ("What is the smallest state in America?", "Rhode Island", "geography"),
    ("When is Halloween?", "October 31, 2026", "holidays"),
    ("When is Christmas?", "December 25, 2026", "holidays"),
    ("When is Easter?", "March 28, 2027", "holidays"),
    ("How many days until New Year's Day?", "January 1, 2027", "holidays"),
    ("When is Australian Father's Day?", "", "holidays"),
    ("What are the seven continents?", "Africa", "knowledge"),
    ("Which is the largest ocean?", "Pacific", "knowledge"),
    ("What is the largest planet?", "Jupiter", "knowledge"),
    ("What is the chemical symbol for gold?", "Au", "knowledge"),
    ("What chemical symbol is gold?", "Au", "knowledge"),
    ("What is the chemical formula for water?", "H2O", "knowledge"),
    ("What is the value of pi?", "3.141", "knowledge"),
    ("How fast can an ostrich run?", "70 kilometres", "knowledge"),
    ("How many bones does a human have?", "206", "knowledge"),
    ("When was the Holocaust?", "1941", "knowledge"),
    ("When was the Eiffel Tower built?", "1887", "knowledge"),
    ("When did the Titanic sink?", "1912", "knowledge"),
    ("How to cook eggs?", "boil, fry or scramble", "knowledge"),
    ("What do I got next", "Next is Maths", "timetable"),
    ("What do I have after", "Next is Maths", "timetable"),
    ("What lesson is up next?", "Next is Maths", "timetable"),
    ("What lesson is on now?", "You are in Music", "timetable"),
    ("Can you please tell me what the time is?", "10:00 AM", "datetime"),
    ("Please what time is it", "10:00 AM", "datetime"),
    ("convert 2 weeks to days", "14 days", "units"),
    ("2 acres in square meters", "8093.71", "units"),
    ("increase 50 by 20%", "60", "percent"),
    ("what is the percentage increase from 100 to 110", "10 percent increase", "percent"),
    ("add 8 and 9", "17", "calculator"),
    ("subtract 3 from 10", "7", "calculator"),
    ("start a countdown for 3 seconds", "Timer set for 3 seconds", "timer"),
    ("cancel the countdown", "Cancelled 1 timer", "timer"),
]

EXPANDED_FALL_THROUGH = [
    "What's the weather in Adelaide tomorrow?", "Can you tell me a joke?",
    "How do computers learn?", "Who is the president of France?",
    "What's the capital of Wakanda?", "Tell me about Paris",
    "What do I have next week?", "When does the next train leave?",
    "What is a class in Python?", "Where do you think I should travel?",
    "Explain why Germany lost World War II", "Explain the US constitution",
    "Explain in detail how to cook eggs", "Write a story about an ostrich",
    "Please explain what a light-year is in detail",
    "Does gravity matter in Minecraft?", "Who was responsible for it?",
    "When was it built?", "What is the main attraction there?",
    "Can you recommend a good game?", "How is my week looking?",
    "What is the capital of an imaginary country?",
    "Give me the latest news on NASA", "Are you better than other assistants?",
    "How many days left in the month?", "Is the state of mind important?",
    "Should I take the next bus?", "What do you think about France?",
    "Are there new holidays being proposed?", "What time is the concert?",
]

# Sentence-length selection is deliberately based on the latest original user
# message, not on context text assembled by the GUI.
DETAIL_MODE_TESTS = [
    ("Explain in detail how to cook eggs", True),
    ("Tell me thoroughly how a battery works", True),
    ("Please give a comprehensive explanation of gravity", True),
    ("Provide a detailed explanation of photosynthesis", True),
    ("Explain this step-by-step", True),
    ("Walk me through this process", True),
    ("Please break it down", True),
    ("Can you elaborate on that?", True),
    ("Go into the details of how it works", True),
    ("Give a longer answer", True),
    ("Please explain it fully", True),
    ("Explain this in depth", True),
    ("I want more detail", True),
    ("Tell me more information", True),
    ("What is the capital of France?", False),
    ("Do not go into detail", False),
    ("Don't explain in detail", False),
    ("Without a detailed explanation please", False),
    ("Give me a short answer", False),
    ("What is 2 plus 2?", False),
]


def selftest() -> int:
    global NOW_OVERRIDE
    NOW_OVERRIDE = datetime(2026, 10, 13, 10, 0)  # a Tuesday
    os.environ["CRUTCH_HOME"] = tempfile.mkdtemp(prefix="crutch_test_")
    os.environ["CRUTCH_QUIET"] = "1"
    fails = 0
    for case in TESTS + TT_TESTS + LOG_TESTS + EXPANDED_TESTS:
        msg, expect, src = case[:3]
        NOW_OVERRIDE = case[3] if len(case) > 3 else datetime(2026, 10, 13, 10, 0)
        r = route(msg)
        ok = r is not None and expect in r[0] and r[1] == src
        fails += not ok
        print(("PASS" if ok else "FAIL"), "|", msg[:70], "->", r if not ok else r[0][:90])
    NOW_OVERRIDE = datetime(2026, 10, 13, 10, 0)
    for msg in MUST_FALL_THROUGH + EXPANDED_FALL_THROUGH:
        r = route(msg)
        ok = r is None
        fails += not ok
        print(("PASS" if ok else "FAIL"), "| falls through:", msg, "" if ok else f"-> {r}")
    for msg, expected in DETAIL_MODE_TESTS:
        actual = wants_detailed_reply(msg)
        ok = actual == expected
        fails += not ok
        print(("PASS" if ok else "FAIL"), "| detailed reply:", msg, "->", actual)
    # Confirm that the ordinary sentence limit remains intact and can be
    # temporarily disabled without mutating global settings.
    sample = "First. Second. Third. Fourth. Fifth."
    limits_ok = (plain(sample) == "First. Second. Third."
                 and plain(sample, max_sentences=0) == sample
                 and plain(sample) == "First. Second. Third.")
    fails += not limits_ok
    print(("PASS" if limits_ok else "FAIL"), "| detailed limit is request-scoped")
    # Exhaustively check all bundled national and regional capital entries.
    added = 0
    for state, (capital, _) in US_STATES.items():
        r = route(f"What is the capital of the US state of {state}?")
        ok = r is not None and capital in r[0] and r[1] == "geography"
        fails += not ok
        added += 1
        if not ok:
            print("FAIL US capital", state, r)
    for region, (capital, _, _) in AU_REGIONS.items():
        r = route(f"What is the capital of {region}?")
        ok = r is not None and capital in r[0] and r[1] == "geography"
        fails += not ok
        added += 1
        if not ok:
            print("FAIL AU region capital", region, r)
    for country, capital in COUNTRY_CAPITALS.items():
        r = route(f"What is the capital of the country of {country}?")
        expected = capital.split(" (")[0]
        ok = r is not None and expected in r[0] and r[1] == "geography"
        fails += not ok
        added += 1
        if not ok:
            print("FAIL country capital", country, r)
    total = len(TESTS) + len(TT_TESTS) + len(LOG_TESTS) + len(EXPANDED_TESTS) + len(MUST_FALL_THROUGH) + len(EXPANDED_FALL_THROUGH) + len(DETAIL_MODE_TESTS) + 1 + added
    print(f"\n{total - fails} passed, {fails} failed ({added} exhaustive capital checks)")
    return 1 if fails else 0


def compare(path: str | None):
    prompts = COMPARE_PROMPTS
    if path:
        prompts = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for p in prompts:
        print("=" * 70)
        print("PROMPT:", p)
        try:
            raw = ask_model(p)
        except ModelUnavailable:
            raw = "(model server not reachable)"
        reply, src = respond(p)
        print(f"RAW MODEL : {raw}")
        print(f"WITH CRUTCH [{src}]: {reply}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Offline tool layer for small language models.")
    ap.add_argument("text", nargs="?", help="one-shot message")
    ap.add_argument("--no-model", action="store_true", help="tools only")
    ap.add_argument("--verbose", action="store_true", help="show which tool answered")
    ap.add_argument("--test", action="store_true", help="run the self-test")
    ap.add_argument("--write-timetable", action="store_true",
                    help="write the built-in timetable to timetable.json so you can edit it")
    ap.add_argument("--compare", nargs="?", const="", metavar="FILE",
                    help="compare raw model vs crutch on built-in prompts or a prompt file")
    a = ap.parse_args()

    if a.test:
        sys.exit(selftest())
    if a.write_timetable:
        print("Wrote", write_timetable_template())
        return
    if a.compare is not None:
        compare(a.compare or None)
        return

    def show(text):
        reply, src = respond(text, use_model=not a.no_model)
        print(f"{reply}  [{src}]" if a.verbose else reply)

    if a.text:
        show(a.text)
        return
    print("crutch ready. Type a message (blank line or Ctrl-C to quit).")
    try:
        while True:
            msg = input("you> ").strip()
            if not msg:
                break
            show(msg)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
