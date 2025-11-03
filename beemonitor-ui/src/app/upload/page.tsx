'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { deriveRelPath, runVideo } from '@/lib/beemonitor';

type RunResponse = {
  ok: boolean;
  run_id: string;
  video: string;
  fps: number;
  n_trajectories: number;
  n_events: number;
  n_visits: number;
  plots: boolean;
  overlay_video?: string | null;
  time: number;
};

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [makePlots, setMakePlots] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RunResponse | null>(null);
  const [showJSON, setShowJSON] = useState(true);

  const dropRef = useRef<HTMLLabelElement | null>(null);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  }, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const canRun = useMemo(() => !!file && !busy, [file, busy]);

  async function handleRun() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append('video', file);
      form.append('make_plots', String(makePlots));

      const { status, data } = await runVideo(form);
      if (status !== 200 || !data?.ok) {
        throw new Error(data?.error || `Run failed (${status})`);
      }
      setResult(data as RunResponse);
    } catch (err: any) {
      setError(err?.message || 'Something went wrong.');
    } finally {
      setBusy(false);
    }
  }

  const overlayHref = result?.overlay_video
    ? (() => {
        const rel = deriveRelPath(result.overlay_video!, result.run_id);
        return `/api/files?run=${encodeURIComponent(result.run_id)}&path=${encodeURIComponent(rel)}`;
      })()
    : null;

  return (
    <div className="min-h-screen w-full bg-neutral-950 text-neutral-100">
      <div className="mx-auto max-w-5xl px-4 py-8">
        <h1 className="text-2xl font-semibold tracking-tight">Upload a video</h1>
        <p className="mt-1 text-sm text-neutral-300">
          Drop a video or choose a file. Optionally generate plots (events/visits).
        </p>

        <div className="mt-6 grid gap-6 md:grid-cols-3">
          {/* LEFT: Controls */}
          <div className="md:col-span-1">
            <div
              onDrop={onDrop}
              onDragOver={onDragOver}
              className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-4"
            >
              <label
                ref={dropRef}
                htmlFor="file"
                className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-neutral-700 bg-neutral-900/60 p-6 transition hover:border-neutral-500"
              >
                <svg className="h-8 w-8 opacity-80" viewBox="0 0 24 24" fill="none">
                  <path d="M12 16V4m0 12l-3-3m3 3l3-3M4 16v4h16v-4" stroke="currentColor" strokeWidth="1.5" />
                </svg>
                <div className="text-center">
                  <div className="text-sm font-medium">
                    {file ? <span className="text-emerald-400">{file.name}</span> : 'Drag & drop your video'}
                  </div>
                  <div className="mt-1 text-xs text-neutral-400">or click to choose</div>
                </div>
                <input
                  id="file"
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
              </label>

              <div className="mt-4 flex items-center gap-2">
                <input
                  id="plots"
                  type="checkbox"
                  checked={makePlots}
                  onChange={(e) => setMakePlots(e.target.checked)}
                  className="h-4 w-4 accent-emerald-500"
                />
                <label htmlFor="plots" className="text-sm">Generate plots (events/visits)</label>
              </div>

              <button
                onClick={handleRun}
                disabled={!canRun}
                className={`mt-4 w-full rounded-xl px-4 py-2 text-sm font-medium transition
                  ${canRun ? 'bg-emerald-500 hover:bg-emerald-400 text-black' : 'bg-neutral-800 text-neutral-500 cursor-not-allowed'}`}
              >
                {busy ? 'Running…' : 'Run'}
              </button>

              {error && (
                <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-200">
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* RIGHT: Results */}
          <div className="md:col-span-2">
            <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-medium">Run result</h2>
                <button
                  onClick={() => setShowJSON((s) => !s)}
                  className="rounded-lg border border-neutral-700 bg-neutral-800/60 px-3 py-1 text-xs text-neutral-200 hover:bg-neutral-700"
                >
                  {showJSON ? 'Hide JSON' : 'Show JSON'}
                </button>
              </div>

              {/* Summary badges */}
              <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
                <Stat label="Trajectories" value={result?.n_trajectories ?? 0} />
                <Stat label="Events" value={result?.n_events ?? 0} />
                <Stat label="Visits" value={result?.n_visits ?? 0} />
                <Stat label="FPS" value={result?.fps ? result.fps.toFixed(2) : '—'} />
                <Stat label="Plots" value={result?.plots ? 'Yes' : 'No'} />
                <Stat label="Run ID" value={result?.run_id ?? '—'} mono />
              </div>

              {/* Overlay button */}
              {overlayHref && (
                <a
                  href={overlayHref}
                  className="mt-3 inline-flex items-center gap-2 rounded-lg border border-neutral-700 bg-neutral-800/60 px-3 py-1 text-sm hover:bg-neutral-700"
                >
                  Download overlay video
                </a>
              )}

              {/* JSON viewer */}
              {showJSON && (
                <pre className="mt-4 max-h-[360px] overflow-auto rounded-xl bg-neutral-950 p-4 text-xs text-neutral-100">
{JSON.stringify(result ?? { hint: 'Run to see output here.' }, null, 2)}
                </pre>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, mono = false }: { label: string; value: any; mono?: boolean }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-3">
      <div className="text-[11px] uppercase tracking-wide text-neutral-400">{label}</div>
      <div className={`mt-1 text-base ${mono ? 'font-mono' : 'font-semibold'}`}>{String(value)}</div>
    </div>
  );
}
