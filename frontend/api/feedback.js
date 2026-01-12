export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const API_URL = process.env.API_URL || 'http://127.0.0.1:8000';
  // const API_URL = 'http://127.0.0.1:8000';

  try {
    const response = await fetch(`${API_URL}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Feedback API Error:', response.status, errorText);
      return res.status(response.status).json({
        error: 'Failed to submit feedback',
        detail: errorText || `API responded with status ${response.status}`
      });
    }

    const data = await response.json();
    return res.status(200).json(data);
  } catch (error) {
    console.error('Feedback Submission Error:', error);
    return res.status(500).json({
      error: 'Failed to submit feedback',
      detail: error.message
    });
  }
}
