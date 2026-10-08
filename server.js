import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// Root health check endpoint
app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    message: 'Sleep Disorder Backend API is running!',
    endpoints: {
      health: '/api/health',
      predict: '/api/predict'
    }
  });
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    backend: 'Node.js Express',
    inferenceEngine: 'Python LightGBM',
    model: 'sleep_disorder_lightgbm_model.joblib'
  });
});

// Sleep Disorder Prediction Endpoint
app.post('/api/predict', (req, res) => {
  const inputData = req.body;

  if (!inputData || Object.keys(inputData).length === 0) {
    return res.status(400).json({ error: 'No input health metrics provided.' });
  }

  const scriptPath = path.join(__dirname, 'predict.py');
  const pythonCmd = process.env.PYTHON_BIN || (process.platform === 'win32' ? 'python' : 'python3');
  const pythonProcess = spawn(pythonCmd, [scriptPath]);

  let stdoutData = '';
  let stderrData = '';

  // Write JSON to python process stdin
  pythonProcess.stdin.write(JSON.stringify(inputData));
  pythonProcess.stdin.end();

  pythonProcess.stdout.on('data', (data) => {
    stdoutData += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    stderrData += data.toString();
  });

  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      console.error(`Python inference failed with code ${code}:`, stderrData);
      return res.status(500).json({
        error: 'Prediction calculation failed.',
        details: stderrData.trim() || `Exit code ${code}`
      });
    }

    try {
      const result = JSON.parse(stdoutData.trim());
      console.log(`[PREDICTION] Input: ${inputData.occupation || 'N/A'}, Age: ${inputData.age} -> Result: ${result.prediction} (${(result.confidence * 100).toFixed(1)}%)`);
      return res.json(result);
    } catch (parseError) {
      console.error('Failed to parse Python output:', stdoutData);
      return res.status(500).json({
        error: 'Invalid response from inference model.',
        raw: stdoutData
      });
    }
  });

  pythonProcess.on('error', (err) => {
    console.error('Failed to spawn Python process:', err);
    return res.status(500).json({
      error: 'Failed to start Python runtime.',
      message: err.message
    });
  });
});

const server = app.listen(PORT, '0.0.0.0', () => {
  console.log(`=================================================`);
  console.log(`Node.js Express Backend is running on port ${PORT}`);
  console.log(`Health check: http://0.0.0.0:${PORT}/api/health`);
  console.log(`ML Inference: Python LightGBM (sleep_disorder_lightgbm_model.joblib)`);
  console.log(`=================================================`);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n❌ Error: Port ${PORT} is already in use by another process!`);
    console.error(`👉 The server is already running on port ${PORT}.\n`);
  } else {
    console.error('Server error:', err);
  }
});
