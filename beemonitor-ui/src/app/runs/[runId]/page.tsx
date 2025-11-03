import Link from "next/link";

async function getFiles(runId: string) {
  const res = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL || ""}/api/beemonitor/runs/${runId}/files`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load files");
  return res.json();
}

export default async function RunDetail({ params }: { params: { runId: string }}) {
  const { runId } = params;
  const data = await getFiles(runId);

  const eventsPng = data.files.find((f: string) => f.endsWith("events.png"));
  const visitsPng = data.files.find((f: string) => f.endsWith("visits.png"));

  return (
    <main className="p-6 space-y-4">
      <h1 className="text-xl font-semibold">Run: {runId}</h1>
      <Link className="underline" href="/runs">← Back</Link>

      <section className="space-y-2">
        <h2 className="font-semibold">Artifacts</h2>
        <ul className="list-disc pl-6">
          {data.files.map((f: string) => (
            <li key={f}>
              <a className="underline text-blue-600"
                 href={`/api/beemonitor/runs/${encodeURIComponent(runId)}/download?path=${encodeURIComponent(f)}`}>
                {f}
              </a>
            </li>
          ))}
        </ul>
      </section>

      <section className="grid md:grid-cols-2 gap-4">
        {eventsPng && <img alt="events" src={`/api/beemonitor/runs/${encodeURIComponent(runId)}/download?path=${encodeURIComponent(eventsPng)}`} />}
        {visitsPng && <img alt="visits" src={`/api/beemonitor/runs/${encodeURIComponent(runId)}/download?path=${encodeURIComponent(visitsPng)}`} />}
      </section>
    </main>
  );
}
