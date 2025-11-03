import { NextResponse } from "next/server";
import { backendFetch } from "@/lib/beemonitor";

export async function GET(_: Request, { params }: { params: { runId: string }}) {
  const res = await backendFetch(`/runs/${params.runId}/files`);
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
