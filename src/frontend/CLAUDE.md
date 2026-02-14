# Frontend Development Notes

## FastHTML Documentation

FastHTML has limited training data in most LLMs. When working on this frontend,
consult these resources:

- **LLMs context file**: `https://www.fastht.ml/docs/llms-ctx.txt` — comprehensive
  single-file reference optimised for LLM consumption. Fetch this with WebFetch
  when you need to look up idiomatic patterns.
- **Main docs**: `https://www.fastht.ml/docs/` — human-readable docs site
- **FastHTML repo**: `https://github.com/AnswerDotAI/fasthtml`

## Key FastHTML Patterns

### App setup
```python
from fasthtml.common import *
app, rt = fast_app(
    static_path="src/frontend/static",
    hdrs=(Link(rel="stylesheet", href="/styles.css", type="text/css"),),
)
```
- `fast_app()` includes Pico CSS by default. Pass `pico=False` to disable.
- `hdrs` tuple injects elements into `<head>` of every page.
- `static_path` configures where static files are served from.

### HTML generation
- HTML elements are Python functions: `Div()`, `Span()`, `H2()`, `A()`, etc.
- CSS classes via `cls="my-class"` (not `class_` or `className`).
- Inline styles via `style="..."` string.
- Boolean attributes: `checked="checked"` or `None` to omit.
- HTMX attributes use underscores: `hx_get`, `hx_target`, `hx_swap`.
- Children are positional args: `Div(H2("Title"), P("Body"))`.
- `NotStr(html_string)` renders raw HTML without escaping.
- Return `""` (empty string) from a component to render nothing.

### Layouts
- `Titled(page_title, *children)` wraps in Container with H1 + meta title.
- For custom layouts, build `Html(Head(...), Body(...))` manually (what we do).
- `fast_app()` includes Pico CSS which provides semantic styling.

### Routes
```python
@rt("/path")
def handler(request): ...

@rt("/path/{key:path}")
def detail(key: str, request): ...
```

## Project-Specific Patterns

### Design system ("Archival Elegance")
- **Fonts**: Crimson Pro (headings), IBM Plex Sans (body) — loaded via Google Fonts
- **Colors**: Warm teal-slate primary (#2c5f7c), amber accent (#c97b3a)
- **CSS variables**: All defined in `:root` of `static/styles.css`

### Layout structure
- `main_layout()` in `app_config.py` provides the standard sidebar + content layout
- `titled_with_domain_picker()` provides full HTML page with domain switcher
- Filter panels go in the left sidebar (220px fixed)
- Content area is the flexible right panel

### Component conventions
- Reusable components live in `components.py` as plain functions returning FT elements
- Components return `""` when they have nothing to render (FastHTML ignores empty strings)
- Use CSS classes over inline styles wherever possible
- Entity detail pages share a common structure: breadcrumb, version selector, type badge, aliases, tags, profile text, confidence badge, articles

### CSS class reference
| Class | Used for |
|---|---|
| `.home-card` | Home page entity cards |
| `.home-grid` | Home page card grid |
| `.breadcrumb` | Detail page back-navigation |
| `.version-selector` | Profile version dropdown |
| `.filter-panel` | Sidebar filter controls |
| `.content-area` | Main content panel |
| `.entity-detail` | Detail page content wrapper |
| `.profile-text` | Long-form profile content |
| `.tag` | Pill badges for types/tags |
| `.filter-chip` | Toggle filter chips |
| `.card-link` | Styled links in cards |
| `.empty-state` | No-results message |
| `.article-list` | Related articles list |
