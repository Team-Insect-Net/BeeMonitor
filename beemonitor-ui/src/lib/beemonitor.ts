export const API_BASE =
  process.env.NEXT_PUBLIC_BEEMONITOR_URL?.replace(/\/+$/, "") || "http://localhost:9099";

const API_KEY = process.env.BEEMONITOR_API_KEY;

function authHeaders() {
  return API_KEY ? { "x-api-key": API_KEY } : {};
}

export async function runVideo(form: FormData) {
  const res = await fetch(`${API_BASE}/run`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  const text = await res.text();
  try { return { status: res.status, data: JSON.parse(text) }; }
  catch { return { status: res.status, data: { ok: false, error: text } }; }
}

export async function listRuns() {
  const res = await fetch(`${API_BASE}/runs`, { headers: authHeaders(), cache: "no-store" });
  return res.json();
}

export async function listRunFiles(runId: string) {
  const res = await fetch(`${API_BASE}/runs/${encodeURIComponent(runId)}/files`, {
    headers: authHeaders(),
    cache: "no-store",
  });
  return res.json();
}

/** Derive a run-relative path from an absolute overlay path if needed. */
export function deriveRelPath(absPath: string, runId: string) {
  const i = absPath.lastIndexOf(`/${runId}/`);
  return i >= 0 ? absPath.slice(i + runId.length + 2) : absPath; // fallback: return as-is
}
