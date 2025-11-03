"use client";

import { useState } from "react";

export default function RunByPathPage() {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<any>(null);

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setBusy(true);
    setResult(null);
    const form = new FormData(e.currentTarget);
    const res = await fetch("/api/beemonitor/run-by-path", { method: "POST", body: form });
    const data = await res.json();
    setBusy(false);
    setResult(data);
  };

  return (
    <main className="p-6 space-y-4">
      <h1 className="text-xl font-semibold">Run by server path</h1>
      <form onSubmit={onSubmit} className="space-y-4">
        <input name="video_path" type="text" placeholder="/path/to/video.mp4" className="border px-2 py-1 w-full" required />
        <label className="inline-flex items-center gap-2">
          <input name="make_plots" type="checkbox" defaultChecked />
          <span>Generate plots</span>
        </label>
        <button disabled={busy} className="px-4 py-2 rounded bg-black text-white disabled:opacity-50">
          {busy ? "Processing…" : "Run"}
        </button>
      </form>
      {result && <pre className="bg-gray-100 p-3 rounded text-sm overflow-auto">{JSON.stringify(result, null, 2)}</pre>}
    </main>
  );
}
