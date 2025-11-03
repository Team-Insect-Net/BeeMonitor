import { NextResponse } from "next/server";
import { backendFetch } from "@/lib/beemonitor";

export async function POST(req: Request) {
  const form = await req.formData(); // must contain "video" (File) and optional "make_plots"
  const res = await backendFetch("/run", { method: "POST", body: form });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
