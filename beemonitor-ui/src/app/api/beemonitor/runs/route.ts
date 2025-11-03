'use client';
import { useEffect, useState } from "react";
import { listRuns, listRunFiles, deriveRelPath } from "@/lib/beemonitor";

export default function RunsPage() {
  const [runs, setRuns] = useState<any[]>([]);
  const [files, setFiles] = useState<Record<string,string[]>>({});

  useEffect(() => { (async () => {
    const data = await listRuns();
    setRuns((data?.runs ?? []).slice().reverse());
  })(); }, []);

  async function open(runId: string) {
    const data = await listRunFiles(runId);
    setFiles((f) => ({ ...f, [runId]: data.files || [] }));
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="mx-auto max-w-5xl px-4 py-8">
        <h1 className="text-2xl font-semibold">Recent runs</h1>
        <div className="mt-6 space-y-4">
          {runs.map((r) => (
            <div key={r.run_id} className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-4">
              <div className="flex items-center justify-between">
                <div className="text-base font-medium">{r.run_id}</div>
                <button
                  onClick={() => open(r.run_id)}
                  className="rounded-lg border border-neutral-700 bg-neutral-800/60 px-3 py-1 text-xs hover:bg-neutral-700"
                >
                  List files
                </button>
              </div>
              <div className="mt-1 text-xs text-neutral-400">
                {r.n_trajectories} tracks • {r.n_events} events • {r.n_visits} visits • fps {r.fps?.toFixed?.(2)}
              </div>

              {/* overlay shortcut if present */}
              {r.overlay_video && (
                <div className="mt-3">
                  <a
                    href={`/api/files?run=${encodeURIComponent(r.run_id)}&path=${encodeURIComponent(deriveRelPath(r.overlay_video, r.run_id))}`}
                    className="inline-flex items-center gap-2 rounded-lg border border-neutral-700 bg-neutral-800/60 px-3 py-1 text-sm hover:bg-neutral-700"
                  >
                    Download overlay
                  </a>
                </div>
              )}

              {files[r.run_id] && (
                <div className="mt-3">
                  <div className="text-xs mb-2 text-neutral-400">Artifacts</div>
                  <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {files[r.run_id].map((p) => (
                      <li key={p}>
                        <a
                          href={`/api/files?run=${encodeURIComponent(r.run_id)}&path=${encodeURIComponent(p)}`}
                          className="text-sm underline decoration-neutral-500 hover:decoration-neutral-200"
                        >
                          {p}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
          {!runs.length && <div className="text-sm text-neutral-400">No runs yet.</div>}
        </div>
      </div>
    </div>
  );
}
