import Link from "next/link";

export default function Home() {
  return (
    <main className="p-6 space-y-4">
      <h1 className="text-2xl font-semibold">BeeMonitor UI</h1>
      <ul className="list-disc pl-6 space-y-2">
        <li><Link className="underline" href="/upload">Upload a video</Link></li>
        <li><Link className="underline" href="/run-by-path">Run by server path</Link></li>
        <li><Link className="underline" href="/runs">Browse runs</Link></li>
      </ul>
    </main>
  );
}
