export const runtime = "nodejs";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const prompt = typeof body?.prompt === "string" ? body.prompt.trim() : "";

  if (!prompt) {
    return Response.json({ error: "prompt required" }, { status: 400 });
  }

  const endpoint = process.env.MODAL_ENDPOINT_URL;
  const apiKey = process.env.PLAYGROUND_API_KEY;
  if (!endpoint || !apiKey) {
    return Response.json(
      { error: "server misconfigured: missing MODAL_ENDPOINT_URL or PLAYGROUND_API_KEY" },
      { status: 500 }
    );
  }

  const maxNewTokens = Math.max(1, Math.min(Number(body?.max_new_tokens) || 40, 80));

  try {
    const upstream = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_new_tokens: maxNewTokens, api_key: apiKey }),
      signal: AbortSignal.timeout(45_000),
    });

    if (!upstream.ok) {
      const detail = await upstream.text().catch(() => "");
      return Response.json(
        { error: `inference backend returned ${upstream.status}`, detail },
        { status: 502 }
      );
    }

    const data = await upstream.json();
    return Response.json(data);
  } catch {
    return Response.json({ error: "inference backend unreachable or timed out" }, { status: 504 });
  }
}
