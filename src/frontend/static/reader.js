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
  async function open(link) {
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
    clearActive();
    link.classList.add("sel");
    const claim = document.getElementById("claim-" + link.dataset.claim);
    if (claim) claim.classList.add("active");
    if (wide.matches && panel) {
      if (inline) { inline.remove(); inline = null; }
      panel.innerHTML = html;
      wire(panel);
    } else {
      if (inline) inline.remove();
      inline = document.createElement("div");
      inline.className = "inline-src";
      inline.innerHTML = html;
      const block = link.closest("p, li, h2, h3, h4") || link.parentElement;
      block.after(inline);
      wire(inline);
    }
  }
  document.addEventListener("click", (ev) => {
    const link = ev.target.closest("sup.c a.cite");
    // The static copy has no server to ask for the panel: citations open the source page
    if (!link || document.body.dataset.static || ev.metaKey || ev.ctrlKey || ev.shiftKey) return;
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
