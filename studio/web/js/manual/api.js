// HTTP client for manual mode. Relative URLs only (a gateway proxies /api/*); `fetchFn` is injectable (tests, the dev shim, an API-key wrapper).
export function makeApi({ fetchFn, base = "" } = {}) {
  const f = (...a) => (fetchFn || window.fetch.bind(window))(...a);
  const json = async (url, opts) => {
    const r = await f(base + url, opts);
    if (!r.ok) {
      let detail = "";
      try { detail = (await r.json()).detail || ""; } catch (_) { /* not json */ }
      throw new Error(`${r.status} ${typeof detail === "string" ? detail : JSON.stringify(detail)}`.trim());
    }
    return r.json();
  };
  const post = (url, body) => json(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return {
    applicable: (smiles, locked = []) => json(`/api/manual/applicable?smiles=${encodeURIComponent(smiles)}&locked=${locked.join(",")}`),
    preview: (req) => post("/api/manual/preview", req),
    validate: (smiles) => post("/api/molecules/validate", { smiles }),
    createSession: (body) => post("/api/manual/sessions", body),
    appendStep: (id, step) => post(`/api/manual/sessions/${encodeURIComponent(id)}/steps`, step),
    exportText: async (id) => {
      const r = await f(`${base}/api/manual/sessions/${encodeURIComponent(id)}/export`);
      if (!r.ok) throw new Error(`export failed: HTTP ${r.status}`);
      return r.text();
    },
  };
}
