async function getRuns() {
  const res = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL || ""}/api/beemonitor/runs`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load runs");
  return res.json();
}

export default async function RunsPage() {
  const data = await getRuns();

  return (
    <main className="p-6 space-y-4">
      <h1 className="text-xl font-semibold">Runs</h1>
      {!data?.runs?.length ? (
        <p>No runs yet.</p>
      ) : (
        <ul className="space-y-2">
          {data.runs.slice().reverse().map((r: any) => (
            <li key={r.run_id} className="border rounded p-3">
              <div className="font-semibold">{r.run_id}</div>
              <div className="text-sm text-gray-600">video: {r.video}</div>
              <div className="text-sm">events: {r.n_events} • visits: {r.n_visits} • fps: {r.fps}</div>
              <a className="underline text-blue-600" href={`/runs/${encodeURIComponent(r.run_id)}`}>Open</a>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
