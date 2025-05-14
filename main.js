// main.js - Updated version that checks if backend is already running
const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');

let mainWindow;
let pythonProcess;

// Check if backend is already running
function checkBackendHealth() {
  return new Promise((resolve) => {
    const options = {
      hostname: 'localhost',
      port: 8000,
      path: '/',
      method: 'GET',
      timeout: 2000
    };

    const req = http.request(options, (res) => {
      resolve(res.statusCode === 200);
    });

    req.on('error', () => {
      resolve(false);
    });

    req.on('timeout', () => {
      resolve(false);
    });

    req.end();
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  mainWindow.loadFile('index.html');

  if (!app.isPackaged) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

async function startPythonBackend() {
  // Check if backend is already running
  const isRunning = await checkBackendHealth();
  if (isRunning) {
    console.log('Backend is already running, skipping startup');
    return;
  }

  const script = path.join(__dirname, 'backend', 'app', 'main.py');
  
  if (!app.isPackaged) {
    pythonProcess = spawn('python', [script], {
      cwd: path.join(__dirname, 'backend')
    });
  } else {
    const executable = process.platform === 'win32' 
      ? path.join(process.resourcesPath, 'backend', 'main.exe')
      : path.join(process.resourcesPath, 'backend', 'main');
    
    pythonProcess = spawn(executable, [], {
      cwd: path.join(process.resourcesPath, 'backend')
    });
  }

  if (pythonProcess) {
    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
    });

    pythonProcess.on('error', (error) => {
      console.error(`Failed to start Python process: ${error}`);
    });
  }
}

app.on('ready', async () => {
  await startPythonBackend();
  
  // Wait a bit for the backend to start (if it wasn't already running)
  setTimeout(() => {
    createWindow();
  }, 2000);
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  // Only kill the python process if we started it
  if (pythonProcess) {
    pythonProcess.kill();
  }
});