import { NextResponse } from "next/server";
import { backendFetch } from "@/lib/beemonitor";

export async function POST(req: Request) {
  const form = await req.formData();
  const res = await backendFetch("/run-by-path", {
    method: "POST",
    body: form,
  });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
