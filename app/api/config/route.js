export function GET() {
  return Response.json({
    openrouter: Boolean(process.env.OPENROUTER_API_KEY),
    elevenlabs: Boolean(process.env.ELEVENLABS_API_KEY),
  });
}
