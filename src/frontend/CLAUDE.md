# Web reader notes

The reader is a Starlette app (`app.py`) that fills Jinja2 templates
(`templates/`). It reads saved Parquet tables and the domain's articles file;
it never calls a model. Run it with `just frontend`.

## Where things are

| File | What it does |
|---|---|
| `app.py` | Routes and the `u()` template helper that keeps `?domain=` on links |
| `views.py` | Builds the plain dicts each template shows (no HTML except escaped, highlighted text) |
| `profile_render.py` | Profile Markdown → HTML with numbered citations, verdict marks and contents list |
| `data_access.py` | `TABLES`, `load_store()` (cached until a Parquet file changes), `Entity`, `Source` |
| `templates/base.html` | Top bar (brand, search, table links, domain picker) |
| `templates/entity.html` | Header, trust line, banner for stubs and link-only pages, contents, profile, sources disagree, in date order, appears with, sources, history, source panel |
| `templates/_panel.html` | The source panel fragment (`/panel`), also shown under a paragraph on narrow screens |
| `templates/_macros.html` | `entity_row`, `flags`, `kind_pill` (import them `with context`, or `u()` is undefined) |
| `corrections_view.py` | The correction forms, the `/corrections` list, the box on each page, and how a posted form becomes a line in the corrections file (`src/utils/corrections.py`) |
| `templates/correct_sentence.html`, `correct_page.html`, `corrections.html` | The correction pages (server only) |
| `static/reader.js` | Opens sources in the panel, filters marks, folds contents on phones |
| `export.py` | `just export-site <out>`: the static copy. Renders every page through the app, rewrites links to relative files, writes `static/search-index.js`. Entity pages carry `cite_data()` as JSON so `reader.js` can build the panel without `/panel` (served folder only; from disk a citation opens the source page) |
| `static/static-search.js` | Search in the static copy (loaded only there) |
| `static/styles.css` | Tokens for light and dark, all layout |

## Rules worth keeping

- **Claims must be split the way grounding splits them.** `parse_claims()`
  mirrors `_extract_cited_claims()` in `src/utils/grounding.py`; the test
  `test_claims_match_the_grounding_step` fails if they drift. If they drift,
  verdicts land on the wrong sentences.
- **Mark the exceptions, not the passes.** Supported sentences carry no mark.
  Partly supported = amber, not supported = red, not checked or checker
  unsure = grey. An older version (`?version=N`) shows no marks, because the
  stored report describes the current text only.
- **Name the checker and its miss rate.** `CHECKER_MISSES` in `views.py`
  holds the hand-test numbers; update it when the grounding bench is rerun.
- **Passages come from the checker, checked twice.** The panel marks a
  verdict's stored `passages` (found word for word in the article when
  grounding saved them) and looks each up again in the shown paragraphs;
  one it can't find is not marked. With no passage it falls back to marking
  a sentence copied word for word, and says the checker quoted nothing.
- **Escape model text.** Profile text is HTML-escaped before Markdown;
  article text is escaped before `<mark>` is added.
- **Every page works without JavaScript.** A citation is a link to
  `/sources/<id>`; the script only intercepts it.
- **New entity table** (court cases, say): add a `TableSpec` to `TABLES`.
  Navigation, search, lists, "appears with" and source pages loop over it.
- **Sources, not articles.** A source has a `kind` ("News article" today) so
  books, reports and translations can be added without a new page.
- **The static copy** sets the template global `static_site`. A template that
  adds a form or a link needing the server (a query other than `?version=`)
  must hide it when `static_site` is set, or the export test will show a
  broken link.
- **Corrections never touch the tables.** A form appends one line to the
  corrections file; pages draw the file on top (`Overlay` in
  `profile_render.py`, built by `corrections_view.overlay_for()`). Struck
  sentences are matched by `sentence_key()`, so render with the overlay
  wherever claim numbers must agree (page, panel, sentence form, the post
  handler), and never for an older version. A historian's sentence goes into
  the text as a placeholder and is put back escaped after Markdown. Posts must
  carry an `Origin` or `Referer` from the reader itself. Corrections are
  article-derived data: never commit a corrections file.
- **A run carries strikes out in the tables** (`src/engine/corrections_pass.py`):
  the struck sentence is gone from the stored text and the historian's
  sentence is in it, with its citation. The overlay then marks that sentence
  as theirs instead of adding it again (`Overlay.written`), and
  `corrections_view.status()` reads the run's `last-run.json` beside the
  corrections file (`run_rows()`) to say which run did it. Find pages with
  `resolve_ref()` (`find_entity()`), as the pipeline does, so both agree.
- The server listens on 127.0.0.1 unless `HINBOX_FRONTEND_HOST` is set.
- The repository is public: tests use made-up tables only. Never commit
  screenshots or fixtures from a real run. README screenshots come from
  `just demo-reader` (`scripts/demo_reader_data.py`, all invented).
