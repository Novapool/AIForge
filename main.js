// main.js - Updated for packaging
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

// Wait for backend to be ready
function waitForBackend(retries = 30) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    
    const checkInterval = setInterval(async () => {
      attempts++;
      
      if (await checkBackendHealth()) {
        clearInterval(checkInterval);
        resolve();
      } else if (attempts >= retries) {
        clearInterval(checkInterval);
        reject(new Error('Backend failed to start'));
      }
    }, 1000);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    icon: path.join(__dirname, 'assets', 'icon.png') // Add if you have an icon
  });

  // Load the HTML file
  mainWindow.loadFile('index.html');

  // Open DevTools only in development
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

  if (!app.isPackaged) {
    // Development mode - run Python script directly
    const script = path.join(__dirname, 'backend', 'app', 'main.py');
    pythonProcess = spawn('python', [script], {
      cwd: path.join(__dirname, 'backend')
    });
  } else {
    // Production mode - run packaged executable
    let executableName = 'main';
    if (process.platform === 'win32') {
      executableName = 'main.exe';
    }
    
    const executable = path.join(process.resourcesPath, 'backend', executableName);
    const executableDir = path.join(process.resourcesPath, 'backend');
    
    pythonProcess = spawn(executable, [], {
      cwd: executableDir
    });
  }

  if (pythonProcess) {
    pythonProcess.stdout.on('data', (data) => {
      console.log(`Backend stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Backend stderr: ${data}`);
    });

    pythonProcess.on('error', (error) => {
      console.error(`Failed to start backend: ${error}`);
    });

    pythonProcess.on('close', (code) => {
      console.log(`Backend process exited with code ${code}`);
    });
  }
}

app.on('ready', async () => {
  try {
    await startPythonBackend();
    await waitForBackend();
    createWindow();
  } catch (error) {
    console.error('Failed to start application:', error);
    app.quit();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  // Kill the python process if we started it
  if (pythonProcess) {
    console.log('Stopping backend...');
    pythonProcess.kill();
  }
});

// Prevent app from exiting when windows are closed
app.on('before-quit', (event) => {
  if (pythonProcess) {
    event.preventDefault();
    pythonProcess.kill();
    setTimeout(() => {
      app.quit();
    }, 1000);
  }
});