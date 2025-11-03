import { NextRequest } from "next/server";

const API_BASE = (process.env.NEXT_PUBLIC_BEEMONITOR_URL || "http://localhost:9099").replace(/\/+$/,"");
const API_KEY = process.env.BEEMONITOR_API_KEY;

export async function GET(req: NextRequest) {
  const url = new URL(req.url);
  const run = url.searchParams.get("run");
  const path = url.searchParams.get("path");
  if (!run || !path) {
    return new Response(JSON.stringify({ ok:false, error:"Missing run or path" }), { status: 400 });
  }

  const backend = `${API_BASE}/runs/${encodeURIComponent(run)}/download?path=${encodeURIComponent(path)}`;
  const res = await fetch(backend, { headers: API_KEY ? { "x-api-key": API_KEY } : {} });

  if (!res.ok) {
    const text = await res.text();
    return new Response(text || "Download failed", { status: res.status });
  }

  // Stream response and forward headers
  const ct = res.headers.get("content-type") ?? "application/octet-stream";
  const cd = res.headers.get("content-disposition") ?? `attachment; filename="${path.split("/").pop()}"`;
  return new Response(res.body, {
    status: 200,
    headers: {
      "content-type": ct,
      "content-disposition": cd,
      // Allow range for media scrubbing in some browsers/players
      "accept-ranges": res.headers.get("accept-ranges") ?? "bytes",
    },
  });
}
