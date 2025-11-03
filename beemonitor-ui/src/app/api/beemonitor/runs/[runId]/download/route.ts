import { backendFetch } from "@/lib/beemonitor";

export async function GET(req: Request, { params }: { params: { runId: string } }) {
  const { searchParams } = new URL(req.url);
  const path = searchParams.get("path");
  if (!path) {
    return new Response("Missing path", { status: 400 });
  }
  // Stream the file through — preserves auth on backend
  const res = await backendFetch(`/runs/${params.runId}/download?path=${encodeURIComponent(path)}`);
  if (!res.ok) {
    const msg = await res.text();
    return new Response(msg, { status: res.status });
  }
  const headers = new Headers(res.headers);
  // Let the browser download as attachment if backend sets it; otherwise pass through
  return new Response(res.body, { status: 200, headers });
}
