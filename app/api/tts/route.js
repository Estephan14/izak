const DEFAULT_VOICE = 'ErXwobaYiN019PkySvjV';

export async function POST(request) {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    return Response.json({ error: 'ELEVENLABS_API_KEY missing' }, { status: 500 });
  }

  const { text, voiceId } = await request.json();
  if (!text?.trim()) {
    return Response.json({ error: 'text is required' }, { status: 400 });
  }

  const upstream = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${voiceId || DEFAULT_VOICE}`,
    {
      method: 'POST',
      headers: { 'xi-api-key': apiKey, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        model_id: 'eleven_flash_v2_5',
        voice_settings: { stability: 0.5, similarity_boost: 0.75, speed: 1.1 },
      }),
    }
  );

  return new Response(await upstream.arrayBuffer(), {
    status: upstream.status,
    headers: {
      'Content-Type': upstream.headers.get('content-type') || 'audio/mpeg',
      'Cache-Control': 'no-store',
    },
  });
}
