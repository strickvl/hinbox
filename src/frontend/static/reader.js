// Hinbox reader: opens a citation's source beside the profile (wide screens)
// or under its paragraph (narrow ones), and fades all but one kind of
// checker mark when a count in the trust line is clicked. Without this
// script every citation is still a plain link to the source page.
(function () {
  const desk = document.getElementById("desk");
  const prose = document.getElementById("prose");
  if (!desk || !prose) return;
  const panel = document.getElementById("panel");
  const wide = window.matchMedia("(min-width: 1181px)");
  let inline = null;
  // On a phone the contents list starts folded, so the profile comes first
  const toc = document.getElementById("toc-details");
  if (toc && window.matchMedia("(max-width: 760px)").matches) toc.open = false;

  function clearActive() {
    document.querySelectorAll(".s.active").forEach((el) => el.classList.remove("active"));
    document.querySelectorAll("sup.c a.sel").forEach((el) => el.classList.remove("sel"));
  }
  function close() {
    clearActive();
    if (inline) { inline.remove(); inline = null; }
    if (panel) panel.innerHTML = '<p class="panel-msg">Click a citation number to read that source here, next to the sentence it supports.</p>';
  }
  function wire(box) {
    box.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", close));
    const mark = box.querySelector("mark");
    if (mark) mark.scrollIntoView({ block: "center" });
    else { const body = box.querySelector(".pbody"); if (body) body.scrollTop = 0; }
  }
  // Show a panel's contents beside the profile (wide) or under the paragraph
  function show(link, fill) {
    clearActive();
    link.classList.add("sel");
    const claim = document.getElementById("claim-" + link.dataset.claim);
    if (claim) claim.classList.add("active");
    let box;
    if (wide.matches && panel) {
      if (inline) { inline.remove(); inline = null; }
      box = panel;
    } else {
      if (inline) inline.remove();
      inline = document.createElement("div");
      inline.className = "inline-src";
      const block = link.closest("p, li, h2, h3, h4") || link.parentElement;
      block.after(inline);
      box = inline;
    }
    fill(box);
    wire(box);
  }
  async function open(link) {
    if (document.body.dataset.static) return openStatic(link);
    const params = new URLSearchParams({ t: desk.dataset.table, k: desk.dataset.key, a: link.dataset.article, c: link.dataset.claim });
    if (desk.dataset.domain && desk.dataset.domain !== "guantanamo") params.set("domain", desk.dataset.domain);
    if (desk.dataset.version) params.set("v", desk.dataset.version);
    let html;
    try {
      const res = await fetch("/panel?" + params.toString());
      html = await res.text();
    } catch (e) {
      window.location.href = link.href;
      return;
    }
    show(link, (box) => { box.innerHTML = html; });
  }

  // The static copy (`just export-site`) has no /panel: build the same panel
  // from the source's exported page and the verdicts the profile page carries.
  // Browsers refuse this for a site opened from disk; the link then just
  // opens the source page.
  const LABELS = { supported: "Supported", partial: "Partly supported", not_supported: "Not supported", unclear: "Checker unsure", unchecked: "Not checked yet" };
  let cites = null;
  try { const el = document.getElementById("cite-data"); if (el) cites = JSON.parse(el.textContent); } catch (e) { cites = null; }
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text) n.textContent = text;
    return n;
  }
  function highlighted(text, needle) {
    const p = el("p");
    const at = needle.length >= 25 ? text.indexOf(needle) : -1;
    if (at < 0) { p.textContent = text; return [p, false]; }
    p.append(text.slice(0, at), el("mark", "", needle), text.slice(at + needle.length));
    return [p, true];
  }
  // Marks each stored passage piece (the article's own words, cut at line
  // breaks) where it appears in a paragraph; a piece not found is not marked
  function withPassages(text, pieces) {
    const spans = [];
    pieces.forEach((s) => {
      const at = text.indexOf(s);
      if (at >= 0 && !spans.some(([a, b]) => a < at + s.length && at < b)) spans.push([at, at + s.length]);
    });
    spans.sort((x, y) => x[0] - y[0]);
    const p = el("p");
    let last = 0;
    spans.forEach(([a, b]) => { p.append(text.slice(last, a), el("mark", "passage", text.slice(a, b))); last = b; });
    p.append(text.slice(last));
    return [p, spans.length > 0];
  }
  async function openStatic(link) {
    let doc, base;
    try {
      const res = await fetch(link.href);
      if (!res.ok) throw new Error(String(res.status));
      base = res.url || link.href;
      doc = new DOMParser().parseFromString(await res.text(), "text/html");
    } catch (e) {
      window.location.href = link.href;
      return;
    }
    const head = doc.querySelector("header.head");
    const c = (cites || {})[link.dataset.claim] || {};
    const aid = link.dataset.article;
    show(link, (box) => {
      box.textContent = "";
      const ph = el("div", "ph");
      const lbl = el("div", "lbl");
      const kind = doc.querySelector("header.head .eyebrow span");
      lbl.append(el("span", "", "Source " + link.textContent + (kind ? " · " + kind.textContent : "")));
      const close = el("button", "close", "×");
      close.type = "button"; close.setAttribute("data-close", ""); close.setAttribute("aria-label", "Close source");
      lbl.append(close);
      const h1 = doc.querySelector("h1");
      const d = el("div", "d");
      const date = head ? head.dataset.date : "";
      if (date) d.append(date + " · ");
      const page = el("a", "", "source page"); page.href = link.href; d.append(page);
      if (head && head.dataset.url) {
        const orig = el("a", "", "original");
        orig.href = head.dataset.url; orig.target = "_blank"; orig.rel = "noopener noreferrer";
        d.append(" · ", orig);
      }
      ph.append(lbl, el("div", "t", h1 ? h1.textContent : ""), d);
      box.append(ph);
      if (c.q) {
        const old = Boolean(desk.dataset.version);
        const v = old ? "" : (c.v || {})[aid] || "unchecked";
        const vd = el("div", "verdict v-" + v);
        vd.append(el("b", "", old ? "Older version" : LABELS[v] || v), el("span", "claim", "“" + c.q + "”"));
        const why = old ? "The checker's verdicts describe the current version only." : (c.r || {})[aid] || (v === "supported" ? "The checker found this in the source." : v === "unchecked" ? "No checker verdict is stored for this sentence and source." : "");
        if (why) vd.append(el("span", "", why));
        box.append(vd);
      }
      const body = el("div", "pbody");
      const needle = (c.q || "").replace(/[.,;:\s]+$/, "");
      const old = Boolean(desk.dataset.version);
      const pieces = old ? [] : (c.p || {})[aid] || [];
      const verdict = old ? "" : (c.v || {})[aid] || "unchecked";
      let marked = false, quoted = false;
      doc.querySelectorAll(".source-text p").forEach((sp) => {
        let p, hit = false;
        if (sp.classList.contains("empty")) p = el("p", "empty", sp.textContent);
        else if (pieces.length) { [p, hit] = withPassages(sp.textContent, pieces); quoted = quoted || hit; }
        else [p, hit] = highlighted(sp.textContent, needle);
        marked = marked || hit;
        body.append(p);
      });
      if (c.q && quoted) body.append(el("p", "note", "Highlighted: the words the checker quoted when it judged this sentence, found word for word in the source. " + (verdict === "supported" ? "They are what it took as support." : "They are what it compared the sentence with; read around them, since the checker can be wrong.")));
      else if (c.q && !marked && body.children.length && verdict !== "unchecked" && !old) body.append(el("p", "note", "The checker quoted nothing from this source that could be found in it word for word, so the whole source is shown with nothing highlighted."));
      box.append(body);
      const chips = [...doc.querySelectorAll(".side .chipl")].filter((a) => a.textContent !== (document.querySelector("h1") || {}).textContent);
      if (chips.length) {
        const ents = el("div", "ents");
        ents.append(el("span", "", "Named here:"));
        chips.slice(0, 24).forEach((a) => { const x = el("a", "chipl", a.textContent); x.href = new URL(a.getAttribute("href"), base).href; ents.append(x); });
        box.append(ents);
      }
    });
  }
  document.addEventListener("click", (ev) => {
    const link = ev.target.closest("sup.c a.cite");
    if (!link || ev.metaKey || ev.ctrlKey || ev.shiftKey) return;
    ev.preventDefault();
    open(link);
  });
  document.querySelectorAll(".trust .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const on = chip.getAttribute("aria-pressed") !== "true";
      document.querySelectorAll(".trust .chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
      prose.querySelectorAll(".s.hit").forEach((el) => el.classList.remove("hit"));
      prose.classList.toggle("filtering", on);
      if (on) {
        chip.setAttribute("aria-pressed", "true");
        const hits = prose.querySelectorAll(".s.v-" + chip.dataset.filter);
        hits.forEach((el) => el.classList.add("hit"));
        if (hits[0]) hits[0].scrollIntoView({ block: "center", behavior: "smooth" });
      }
    });
  });
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") close(); });
})();
