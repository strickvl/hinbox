// Search for the static copy (`just export-site`): there is no server to ask,
// so `search-index.js` lists every page's names and this script matches them
// as you type. It is only loaded in the static copy.
(function () {
  const index = window.HINBOX_INDEX || [];
  const root = document.body.dataset.root || "";
  const fold = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const rows = index.map((e) => ({ e, hay: fold([e.n].concat(e.o || []).join(" | ")) }));

  document.querySelectorAll("form[data-static-search]").forEach((form) => {
    const input = form.querySelector('input[type="search"]');
    const list = document.createElement("ul");
    list.className = "sresults";
    list.hidden = true;
    form.appendChild(list);
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const first = list.querySelector("a");
      if (first) window.location.href = first.href;
    });
    input.addEventListener("input", () => {
      const words = fold(input.value).split(/\s+/).filter(Boolean);
      list.innerHTML = "";
      if (!words.length) { list.hidden = true; return; }
      const hits = rows.filter((r) => words.every((w) => r.hay.includes(w)));
      hits.sort((a, b) => (b.e.c || 0) - (a.e.c || 0));
      hits.slice(0, 12).forEach(({ e }) => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = root + e.u;
        a.textContent = e.n;
        const kind = document.createElement("span");
        kind.className = "muted";
        kind.textContent = " " + e.t;
        li.append(a, kind);
        list.appendChild(li);
      });
      if (!hits.length) {
        const li = document.createElement("li");
        li.className = "muted";
        li.textContent = "No match";
        list.appendChild(li);
      }
      list.hidden = false;
    });
  });
})();
