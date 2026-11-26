"""The HTML of the labelling page (``label_app.py``): one template with three
screens. ``start`` offers a sprint (10, 20 or 40 sentences); ``item`` shows
one sentence beside its article; ``finish`` sums up a sprint. Everything is
inline (no outside files), and the page works in light and dark mode."""

from jinja2 import Environment

PAGE = Environment(autoescape=True).from_string(
    """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Citation labels</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d2327;--muted:#5c6770;--mark:#fff1b8;--accent:#2c5f7c;--card:#fff;--line:#e3e0d8;--ok:#2e7d4f;--bad:#b3412e;--flag:#b3412e}
@media (prefers-color-scheme:dark){:root{--bg:#15191c;--fg:#e6e8ea;--muted:#9aa5ae;--mark:#5a4b12;--accent:#7fb3d1;--card:#1d2226;--line:#2c3338;--ok:#5cb884;--bad:#e07a66;--flag:#f0a08f}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:16px/1.55 Georgia,serif;margin:0;padding:20px 16px 16px}
.bar{position:fixed;top:0;left:0;right:0;height:6px;background:var(--line);z-index:5}
.bar span{display:block;height:100%;background:var(--accent);transition:width .4s ease}
.wrap{max-width:1280px;margin:auto}
.layout{display:block}
@media (min-width:1000px){.layout{display:grid;grid-template-columns:400px 1fr;gap:20px;align-items:start}
 .side{position:sticky;top:20px;max-height:calc(100vh - 28px);overflow:auto}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:16px}
.muted{color:var(--muted);font:14px/1.4 system-ui,sans-serif}
.ui{font:15px/1.4 system-ui,sans-serif}
.claim{font-size:1.15rem;margin:10px 0}
.flag{text-decoration:underline dotted var(--flag) 2px;text-underline-offset:3px}
.src{font:14px/1.4 system-ui,sans-serif;border-left:3px solid var(--accent);padding:4px 0 4px 10px;margin:10px 0}
.src b{font-size:15px}
.q{font:600 15px system-ui,sans-serif;margin:12px 0 6px}
button,.btn{font:600 16px system-ui,sans-serif;padding:10px 16px;margin:4px 6px 4px 0;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer;text-decoration:none;display:inline-block}
button.yes{border-color:var(--ok)}button.no{border-color:var(--bad)}
button.on,.btn.main{background:var(--accent);color:#fff;border-color:var(--accent)}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:3px;padding:0 4px;margin-left:4px;opacity:.75}
input[type=text]{width:100%;padding:8px;margin-top:8px;font:15px system-ui,sans-serif;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px}
mark{background:var(--mark);color:inherit}
.earlier{border:1px solid var(--line);border-radius:8px;padding:4px 12px;margin:12px 0;font:14px/1.45 system-ui,sans-serif}.earlier p{margin:8px 0}.earlier b.no{color:var(--bad)}.earlier b.yes{color:var(--ok)}
.article{white-space:pre-wrap;position:relative}
@media (max-width:999px){.article{max-height:60vh;overflow:auto}}
a{color:var(--accent)}
.big{font:700 40px/1.1 system-ui,sans-serif;margin:6px 0}
.stats{display:flex;gap:28px;flex-wrap:wrap;margin:12px 0}
.toast{position:fixed;top:14px;right:16px;font:600 14px system-ui,sans-serif;background:var(--ok);color:#fff;padding:6px 12px;border-radius:6px;animation:fade 1.2s ease forwards;z-index:6}
@keyframes fade{0%{opacity:0;transform:translateY(-4px)}15%{opacity:1;transform:none}70%{opacity:1}100%{opacity:0}}
.goal{height:4px;background:var(--line);border-radius:2px;margin-top:6px;overflow:hidden}
.goal span{display:block;height:100%;background:var(--ok)}
.cat{position:fixed;left:24px;bottom:0;z-index:7;pointer-events:none;display:flex;align-items:flex-end;gap:10px;animation:peek 3.6s ease-in-out forwards}
.cat svg{width:110px;height:84px;display:block}
.cat .say{font:600 14px system-ui,sans-serif;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:12px;padding:6px 10px;margin-bottom:46px;box-shadow:0 2px 6px rgba(0,0,0,.08)}
@keyframes peek{0%{transform:translateY(100%)}14%{transform:translateY(-4%)}20%{transform:translateY(0)}80%{transform:translateY(0)}100%{transform:translateY(110%)}}
@media (prefers-reduced-motion:reduce){.cat{animation:none;opacity:.95}}
.sleep{display:block;margin:0 auto 6px;width:150px;height:80px}
</style></head><body>
{% if screen == "item" and sprint %}<div class="bar" title="{{ sprint.done }} of {{ sprint.size }} in this sprint"><span style="width:{{ sprint.pct }}%"></span></div>{% endif %}
{% if saved %}<div class="toast">Saved ✓</div>{% endif %}
{% macro catsvg(kind) %}{% set base = "#e8953a" if kind == "ginger" else "#3a2d26" %}{% set patch = "#c06a1f" if kind == "ginger" else "#c7702e" %}{% set patch2 = "#c06a1f" if kind == "ginger" else "#e3a35e" %}
<svg viewBox="0 0 120 92" aria-hidden="true">
<polygon points="22,40 30,6 50,30" fill="{{ base }}"/><polygon points="98,40 90,6 70,30" fill="{{ base }}"/>
<polygon points="28,32 32,14 43,29" fill="#f2b7a8"/><polygon points="92,32 88,14 77,29" fill="#f2b7a8"/>
<ellipse cx="60" cy="56" rx="42" ry="34" fill="{{ base }}"/>
{% if kind == "ginger" %}<path d="M52 24 q8 6 16 0 M48 32 q12 7 24 0 M54 40 q6 3 12 0" stroke="{{ patch }}" stroke-width="3" fill="none" stroke-linecap="round"/>
{% else %}<path d="M60 22 C78 22 98 34 100 52 C88 50 74 40 62 44 Z" fill="{{ patch }}"/><path d="M22 60 C24 46 34 40 44 44 C42 54 34 62 22 60 Z" fill="{{ patch2 }}"/>{% endif %}
<ellipse cx="45" cy="54" rx="6" ry="7" fill="#f6e7a6"/><ellipse cx="75" cy="54" rx="6" ry="7" fill="#f6e7a6"/>
<ellipse cx="45" cy="55" rx="2.4" ry="5.5" fill="#1b1b1b"/><ellipse cx="75" cy="55" rx="2.4" ry="5.5" fill="#1b1b1b"/>
<path d="M56 66 h8 l-4 4 z" fill="#d9776b"/><path d="M60 70 q-4 5 -9 3 M60 70 q4 5 9 3" stroke="#1b1b1b" stroke-width="1.5" fill="none" stroke-linecap="round"/>
<path d="M36 66 h-20 M36 70 l-19 4 M84 66 h20 M84 70 l19 4" stroke="{{ '#fff4e6' if kind == 'ginger' else '#d8cfc7' }}" stroke-width="1.3" stroke-linecap="round"/>
<ellipse cx="36" cy="90" rx="13" ry="8" fill="{{ base }}"/><ellipse cx="84" cy="90" rx="13" ry="8" fill="{{ patch2 if kind == 'tortie' else base }}"/>
</svg>{% endmacro %}
{% macro sleeping(kind) %}{% set base = "#e8953a" if kind == "ginger" else "#3a2d26" %}{% set patch = "#c06a1f" if kind == "ginger" else "#c7702e" %}
<svg class="sleep" viewBox="0 0 150 80" aria-hidden="true">
<ellipse cx="78" cy="56" rx="56" ry="22" fill="{{ base }}"/><path d="M40 52 C60 40 100 40 124 54" stroke="{{ patch }}" stroke-width="5" fill="none" stroke-linecap="round"/>
<circle cx="34" cy="50" r="18" fill="{{ base }}"/><polygon points="22,40 24,24 36,34" fill="{{ base }}"/><polygon points="40,34 48,22 50,38" fill="{{ base }}"/>
<path d="M26 52 q4 3 8 0 M38 52 q4 3 8 0" stroke="#1b1b1b" stroke-width="1.6" fill="none" stroke-linecap="round"/>
<path d="M130 62 C146 60 146 40 132 40" stroke="{{ base }}" stroke-width="8" fill="none" stroke-linecap="round"/>
<text x="60" y="22" font-family="system-ui" font-size="14" fill="currentColor" opacity=".55">z z z</text>
</svg>{% endmacro %}
{% if cat %}<div class="cat">{{ catsvg(cat.kind) }}<div class="say">{{ cat.line }}</div></div>{% endif %}
<div class="wrap">

{% if screen == "start" %}
<div class="card" style="max-width:640px;margin:40px auto">
<div class="muted">Citation test set</div>
<div class="big">{{ done }} <span class="muted" style="font-size:20px">of {{ total }} labelled</span></div>
<div class="muted">First goal: {{ goal }} labelled{% if done >= goal %} (reached, thank you){% endif %}</div>
<div class="goal"><span style="width:{{ goal_pct }}%"></span></div>
{% if left %}
<p class="ui" style="margin-top:20px">How many in this sprint? Sentences from one article come one after another, so you read each article once.</p>
{% for n in sizes %}<a class="btn {{ 'main' if n == default_size }}" href="/start?n={{ n }}">{{ n }}{% if n == default_size %}<kbd>Enter</kbd>{% endif %}</a>{% endfor %}
{% else %}<p class="ui">Every sentence is labelled. Thank you.</p>{% endif %}
</div>

{% elif screen == "finish" %}
<div class="card" style="max-width:640px;margin:40px auto">
{{ sleeping(nap.kind) }}<div class="muted">Sprint done · {{ nap.line }}</div>
<div class="big">{{ sprint.done }} labelled</div>
<div class="stats ui">
<div><div class="muted">Time</div><b>{{ sprint.minutes }}</b></div>
<div><div class="muted">Per sentence</div><b>{{ sprint.per_item }}</b></div>
<div><div class="muted">Overall</div><b>{{ done }} of {{ total }}</b></div>
</div>
<div class="muted">First goal: {{ goal }} labelled{% if done >= goal %} (reached){% else %}, {{ goal - done }} to go{% endif %}</div>
<div class="goal"><span style="width:{{ goal_pct }}%"></span></div>
<p style="margin-top:20px">{% if left %}<a class="btn main" href="/start?n={{ sprint.size }}">Another {{ sprint.size }}<kbd>Enter</kbd></a>{% endif %}
<a class="btn" href="/">Stop for now</a></p>
{% if last %}<p class="muted" style="margin-top:14px"><a href="{{ last }}">Change the last answer</a></p>{% endif %}
</div>

{% else %}
<div class="layout">
<div class="side">
<div class="card">
<div class="muted">{% if sprint %}<b>{{ sprint.done }} of {{ sprint.size }}</b> in this sprint · {% endif %}{{ done }} of {{ total }} overall
{% if back is not none %} · <a href="{{ back }}">back</a><kbd>b</kbd>{% endif %}</div>
<div class="src"><b>{{ meta.date or "No date" }}</b>{% if meta.title %} · {{ meta.title }}{% endif %}<br>
<span class="muted">From the page of <b>{{ item.entity_name }}</b>{% if same_article %} · same article as the last one{% endif %}</span></div>
<p class="claim">{% for piece, flagged in claim_pieces %}{% if flagged %}<span class="flag" title="Not found anywhere in the article">{{ piece }}</span>{% else %}{{ piece }}{% endif %}{% endfor %}</p>
{% if flags %}<div class="muted">Dotted underline: not found anywhere in the article (word match only)</div>{% endif %}
{% if item.earlier %}<div class="earlier">{% for e in item.earlier %}<p><b>{{ e.who }}</b> said <b class="{{ e.label }}">{{ {"yes": "yes", "no": "no", "unsure": "can't tell"}[e.label] }}</b>{% if e.note %}: {{ e.note }}{% endif %}</p>{% endfor %}
<p class="muted">Marked in the article: the passages {{ item.earlier[-1].who }} quoted. A new answer replaces your earlier one.</p></div>{% endif %}
{% if item.level %}<p class="muted">Checkers' verdict: <b>{{ item.level }}</b>{% if item.reasoning %}: {{ item.reasoning }}{% endif %}</p>{% endif %}
<form method="post" action="/label" id="f">
<input type="hidden" name="item_id" value="{{ item.item_id }}">
<input type="hidden" name="index" value="{{ index }}">
{% for k, v in keep.items() %}<input type="hidden" name="{{ k }}" value="{{ v }}">{% endfor %}
<div class="q">{{ question }}</div>
<button class="yes {{ 'on' if current == 'yes' }}" name="label" value="yes">Yes<kbd>y</kbd></button>
<button class="no {{ 'on' if current == 'no' }}" name="label" value="no">No<kbd>n</kbd></button>
<button class="{{ 'on' if current == 'unsure' }}" name="label" value="unsure">Can't tell<kbd>u</kbd></button>
<input type="text" name="note" placeholder="Optional note · Enter when done" value="{{ note }}">
</form></div>
</div>
<div class="card article">{% for piece, marked in pieces %}{% if marked %}<mark class="m">{{ piece }}</mark>{% else %}{{ piece }}{% endif %}{% endfor %}</div>
</div>
{% endif %}
</div>
<script>
var m=document.querySelector('mark.m'),box=document.querySelector('.article');
if(m&&box){if(box.scrollHeight>box.clientHeight+4){box.scrollTop=m.offsetTop-box.offsetTop-box.clientHeight/3;}else{m.scrollIntoView({block:'center'});}}
document.addEventListener('keydown',function(e){
 if(e.target.tagName==='INPUT'){if(e.key==='Enter'){e.preventDefault();e.target.blur();}return;}
 if(e.metaKey||e.ctrlKey||e.altKey)return;
 {% if screen == "item" %}
 var k={y:'yes',n:'no',u:'unsure'}[e.key];
 if(k){var b=document.querySelector('button[value='+k+']');if(b)b.click();}
 {% if back is not none %}if(e.key==='b')location={{ back|tojson }};{% endif %}
 {% else %}
 if(e.key==='Enter'){var a=document.querySelector('a.btn.main');if(a)location=a.href;}
 {% endif %}
});
</script></body></html>"""
)
