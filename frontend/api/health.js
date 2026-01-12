export default async function handler(req, res) {
  const API_URL = process.env.API_URL || 'http://127.0.0.1:8000';
  // const API_URL = 'http://127.0.0.1:8000';

  try {
    const response = await fetch(`${API_URL}/health`, { method: 'GET' });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Health API Error:', response.status, errorText);
      throw new Error(`Health check failed with status ${response.status}`);
    }

    const data = await response.json();
    return res.status(200).json(data);
  } catch (error) {
    console.error('Health Check Error:', error);

    return res.status(200).json({
      status: 'degraded',
      model_loaded: false,
      mlb_loaded: false,
      llm_ready: false
    });
  }
}
