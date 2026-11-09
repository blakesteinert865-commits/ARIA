"""Optional Hassil phrase layer for Aria, loaded only after the existing tools miss.

Hassil parses the phrases; the existing Crutch functions still compute answers.
The bundled snapshot is Hassil 3.12.1, Apache-2.0, with optional import guards
for PyYAML and unicode-rbnf (Aria uses programmatic text-slot intents only).
No smart-home intents are imported.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import sys

_vendor = Path(__file__).resolve().parent / "vendor"
if str(_vendor) not in sys.path:
    sys.path.insert(0, str(_vendor))

from hassil import Intents, recognize_best  # bundled source, no cloud connection


def _v(items):
    """Text value list with canonical output; entries are (heard, canonical)."""
    return {"values": [{"in": hear, "out": output} for hear, output in items]}


def _add(data, name, *sentences):
    data[name] = {"data": [{"sentences": list(sentences)}]}


@lru_cache(maxsize=1)
def build_intents(backend):
    # Reuse the *same* Crutch instance loaded by the GUI bridge.
    # Importing `crutch` by name would otherwise duplicate process-local state.
    crutch = backend
    country = set(crutch.COUNTRY_CAPITALS) | set(crutch.COUNTRY_ALIASES)
    states = set(crutch.US_STATES) | set(crutch.AU_REGIONS)
    # Keep state and country names in distinct lists to disambiguate Georgia.
    regions = [(n, n) for n in sorted(states | country, key=len, reverse=True)]
    regions += [("the " + n, n) for n in sorted(country | states, key=len, reverse=True)]
    units = set()
    for category in crutch._UNITS.values():
        for k, (factor, long_name, aliases) in category.items():
            for alias in aliases:
                units.add(alias)
    # Include actual phrases as part of the list, not an unconstrained wildcard.
    holidays = set(crutch.FIXED_HOLIDAYS) | {"easter", "good friday", "mother's day", "fathers day", "father's day", "mothers day", "thanksgiving"}
    intent = {}
    _add(intent, 'TimeNow',
         "[please ](tell me|give me|say) [the] (current|local|exact) time [please]",
         "(do you know|can you tell me|could you tell me) [what] [the] time [is] [right now]",
         "what time (are we at|have we got|is it currently|is it at the moment)",
         "[so] (what|what's) [the] time (at the moment|currently|over here)",
         "i need to know [the] time")
    _add(intent, 'DateNow',
         "(what|which) date (are we on|is today|are we at)",
         "(can you tell me|could you tell me|tell me) [the] [current] date [please]",
         "(what|which) day of [the] month is it",
         "what's [the] calendar date [today]",
         "what day [of the week] are we on")
    _add(intent, 'TomorrowDate',
         "(what|which) date (is|will it be|are we on) tomorrow",
         "(what|which) day of the week [is it] tomorrow",
         "tell me tomorrow's (day|date)")
    _add(intent, 'YesterdayDate',
         "(what|which) date (was|were we on) yesterday",
         "what day [of the week] was yesterday")
    _add(intent, 'NextClass',
         "(which|what) (class|lesson|subject|period) (comes|is|do i have|have i got) (after this|after this one|next|up next)",
         "(tell me|show me) [what] [the] (next|following) (class|lesson|subject|period) [is]",
         "what (am i|are we) (doing|having|learning) (next|after this)",
         "what is (next|coming up) on my timetable",
         "what's [the] next thing on my timetable",
         "what (have i got|do i have) following this [class]",
         "when's my next (class|lesson|period)",
         "which (class|lesson|subject) follows [this] [one]",
         "what (class|lesson|subject) follows this one")
    _add(intent, 'NextTwoClasses',
         "what (are|will be) my next (two|2) (classes|lessons|subjects)",
         "(tell me|show me) my next (two|2) (classes|lessons)",
         "what (do i have|have i got) next and [then] after that")
    _add(intent, 'CurrentClass',
         "(which|what) (class|lesson|subject|period) [am] i [currently] in [right now]",
         "what (subject|class|lesson) (am i|are we) (doing|having|studying|in) [right now]",
         "(tell me|show me) [my] (current|present) (class|lesson|subject)")
    _add(intent, 'TodayClasses',
         "(tell me|show me|read out) (all|every) [of] my (classes|lessons|subjects) [for] today",
         "what (classes|lessons|subjects) (are|have i got) [on] [my timetable] today",
         "(tell me|show me) today's [school] timetable",
         "what is on my timetable for today")
    _add(intent, 'TomorrowClasses',
         "(tell me|show me|read out) (all|every) [of] my (classes|lessons|subjects) [for] tomorrow",
         "what (classes|lessons|subjects) (are|have i got) [on] [my timetable] tomorrow",
         "(tell me|show me) tomorrow's [school] timetable",
         "what is on my timetable for tomorrow")
    _add(intent, 'SchoolFinish',
         "how much longer (until|till|before) (school|classes|lessons) (finish|end)",
         "when is (school|the school day|my school day) over [today]",
         "tell me [what time] [my] (school|classes|lessons) (finishes|finish|ends|end) [today]")
    _add(intent, 'Capital',
         "(which|what) city (serves as|is|acts as) [the] capital [city] (of|for) {region}",
         "(tell me|give me|name|could you tell me) [the] capital [city] (of|for) {region}",
         "(what's|what is|which is) {region}'s capital [city]",
         "where (is|would i find) [the] capital [city] of {region}",
         "(what|which) (city|place) is {region}'s [state] capital",
         "{region} has (which|what) capital [city]")
    _add(intent, 'Abbreviation',
         "(give me|tell me|what's|what is) [the] [state] (postal abbreviation|postal code|abbreviation|short form) of {state}",
         "(what's|what is) {state}'s (two letter code|state abbreviation|postal abbreviation)",
         "what [two letter] abbreviation is used for {state}")
    _add(intent, 'Holiday',
         "how far away is [the] {holiday}",
         "(what|which) date (will|does) [the] {holiday} (fall on|happen on|occur on)",
         "[can you] tell me [the] date of [the] {holiday}",
         "how much longer (until|till|before) [the] {holiday}",
         "[please] remind me when [the] {holiday} is")
    _add(intent, 'ShowNotes',
         "(could you|can you|please) (tell me|show me|read out) (all|every one) of my [saved] notes",
         "list [out] everything you (remember|have noted down)")
    _add(intent, 'ShowLists',
         "[(can you|could you|please) ](tell me|list|show) all my [saved] lists",
         "what [different] lists have i (got|made|created)")
    _add(intent, 'ReadShopping',
         "[(can you|could you|please) ](read|recite|list|tell me) (everything|all items|all the items) (on|in) my (shopping|grocery) list",
         "(what|which) (items|things) [have i got] (on|in) my (shopping|grocery) list",
         "what (items|things) do i need [to buy] (at the store|at the shops)")
    _add(intent, 'Cooking',
         "(walk me through|tell me|explain) how to (cook|make|prepare) {egg_style} [properly]",
         "(what|which) [is the] (method|process|procedure|steps) [for] (cooking|making|preparing) {egg_style}")
    _add(intent, 'FastestAnimal',
         "(which|what) animal (runs|moves) [the] fastest on land",
         "which animal can run the fastest [on land]",
         "what land animal has the highest top speed")
    _add(intent, 'LargestOcean',
         "which ocean (covers the most area|is the biggest|is the largest) [on earth]",
         "which is earth's biggest ocean")
    _add(intent, 'UnitConversion',
         "what (does|would) {amount} {from_unit} (equal|come to|work out to) in {to_unit}",
         "(change|express|convert) {amount} {from_unit} (over to|into|as) {to_unit} [please]",
         "how much is {amount} {from_unit} (expressed|measured) in {to_unit}")
    _add(intent, 'PercentOf',
         "how much (does|would) {percentage} (percent|%) of {amount} (make|equal|come to)",
         "tell me [the value of] {percentage} (percent|%) of {amount}")
    _add(intent, 'Times',
         "what (would|do) (you get|i get) (if|when) [you] multiply {first} by {second}",
         "how much is {first} multiplied by {second}")
    # Avoid loose deletion, reminder, timers etc. These change state; Crutch's
    # existing stricter rules should remain the only automatic mutating routes.
    intents = {"language":"en", "intents": intent,
               "lists": {
                   "region": _v(regions),
                   "state": _v([(n,n) for n in sorted(states, key=len, reverse=True)]),
                   "holiday": _v([(n,n) for n in sorted(holidays, key=len, reverse=True)]),
                   "from_unit": _v([(n,n) for n in sorted(units, key=len, reverse=True)]),
                   "to_unit": _v([(n,n) for n in sorted(units, key=len, reverse=True)]),
                   "egg_style": _v([(n,n) for n in ["eggs", "scrambled eggs", "fried eggs", "boiled eggs"]]),
                   "amount": {"wildcard": True},
                   "percentage": {"wildcard": True},
                   "first": {"wildcard": True},
                   "second": {"wildcard": True},
                }}
    return Intents.from_dict(intents)


def match(text, backend):
    """Return a matched Hassil object, or None. No fuzzy partial matching."""
    # Full utterance matching; explicit prompts only. Do not treat quotes or
    # content containing multi-line prompts as standalone commands.
    if not text or '\n' in text or len(text) > 240:
        return None
    cleaned = re.sub(r"\s+", " ", text.lower().replace('’', "'")).strip().strip("?!., ")
    if not cleaned:
        return None
    return recognize_best(cleaned, build_intents(backend))


def route(text, tools, backend):
    """Return (answer, source) or None. Canonical questions reuse old tools."""
    r = match(text, backend)
    if r is None:
        return None
    slots = {k: str(v.value) for k, v in r.entities.items()}
    name = r.intent.name
    fixed = {
        'TimeNow': ('datetime', 'what time is it'),
        'DateNow': ('datetime', 'what is the date'),
        'TomorrowDate': ('datetime', 'what is the date tomorrow'),
        'YesterdayDate': ('datetime', 'what is the date yesterday'),
        'NextClass': ('timetable', 'what do i have next'),
        'NextTwoClasses': ('timetable', 'what do i have next and then after'),
        'CurrentClass': ('timetable', 'what class am i in now'),
        'TodayClasses': ('timetable', 'what classes do i have today'),
        'TomorrowClasses': ('timetable', 'what classes do i have tomorrow'),
        'SchoolFinish': ('timetable', 'when does school finish'),
        'ShowNotes': ('lists', 'show my notes'),
        'ShowLists': ('lists', 'show my lists'),
        'ReadShopping': ('lists', "what's on my shopping list"),
        'FastestAnimal': ('knowledge', 'what is the fastest land animal'),
        'LargestOcean': ('knowledge', 'what is the largest ocean'),
    }
    if name in fixed:
        function, canonical = fixed[name]
    elif name == 'Capital':
        function, canonical = 'geography', f"what is the capital of {slots['region']}"
    elif name == 'Abbreviation':
        function, canonical = 'geography', f"what is the abbreviation for {slots['state']}"
    elif name == 'Holiday':
        function, canonical = 'holidays', f"when is {slots['holiday']}"
    elif name == 'Cooking':
        function, canonical = 'knowledge', f"how to cook {slots['egg_style']}"
    elif name in ('UnitConversion', 'PercentOf', 'Times'):
        # Wildcards are unrestricted in Hassil: we must validate all captured
        # numeric values *before* invoking any tool.
        keys = ('amount',) if name == 'UnitConversion' else ('percentage', 'amount') if name == 'PercentOf' else ('first', 'second')
        for key in keys:
            val = slots.get(key, '').strip()
            if not re.fullmatch(r"-?\d{1,9}(?:\.\d{1,6})?", val):
                return None
            slots[key] = val
        if name == 'UnitConversion':
            function, canonical = 'units', f"{slots['amount']} {slots['from_unit']} to {slots['to_unit']}"
        elif name == 'PercentOf':
            function, canonical = 'percent', f"{slots['percentage']} percent of {slots['amount']}"
        else:
            function, canonical = 'calculator', f"multiply {slots['first']} by {slots['second']}"
    else:
        return None
    answer = tools[function](canonical)
    return (answer, 'hassil:' + name) if answer else None
