// main.js - Updated with port detection and connection fixes
const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const net = require('net');

let mainWindow;
let pythonProcess;
let backendStarted = false;
const PORT = 8000;

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(app.isPackaged ? process.resourcesPath : __dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Check if a port is in use
function isPortInUse(port) {
  return new Promise((resolve) => {
    const server = net.createServer()
      .once('error', () => {
        // Port is in use
        resolve(true);
      })
      .once('listening', () => {
        // Port is free
        server.close();
        resolve(false);
      })
      .listen(port, '127.0.0.1');
  });
}

// Check if backend is already running
function checkBackendHealth() {
  return new Promise((resolve) => {
    const options = {
      hostname: 'localhost',
      port: PORT,
      path: '/health',  // Use the health endpoint we created
      method: 'GET',
      timeout: 3000 // Increased timeout
    };

    const req = http.request(options, (res) => {
      if (res.statusCode === 200) {
        console.log('Backend health check successful');
        resolve(true);
      } else {
        console.log(`Backend returned status code ${res.statusCode}`);
        resolve(false);
      }
    });

    req.on('error', (err) => {
      console.log(`Backend health check error: ${err.message}`);
      resolve(false);
    });

    req.on('timeout', () => {
      console.log('Backend health check timeout');
      req.destroy();
      resolve(false);
    });

    req.end();
  });
}

// Wait for backend to be ready with improved logging
async function waitForBackend(retries = 30, delay = 1000) {
  console.log(`Waiting for backend to be ready (max ${retries} attempts with ${delay}ms delay)`);
  
  for (let attempt = 1; attempt <= retries; attempt++) {
    console.log(`Backend health check attempt ${attempt}/${retries}`);
    
    if (await checkBackendHealth()) {
      console.log('Backend is ready!');
      return true;
    }
    
    // Wait before next attempt
    await new Promise(resolve => setTimeout(resolve, delay));
  }
  
  console.error('Backend failed to start after maximum retries');
  throw new Error('Backend failed to start');
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    icon: path.join(__dirname, 'assets', 'icon.png')
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
  // First check if something is already running on port 8000
  const portInUse = await isPortInUse(PORT);
  console.log(`Port ${PORT} in use: ${portInUse}`);
  
  // Then check if that something is our backend
  const backendRunning = await checkBackendHealth();
  console.log(`Backend already running: ${backendRunning}`);
  
  if (backendRunning) {
    console.log('Backend is already running and healthy, skipping startup');
    backendStarted = true;
    return;
  }
  
  // If port is in use but backend isn't responding, it could be another service or a zombie process
  if (portInUse) {
    console.warn(`Port ${PORT} is in use but backend health check failed. Another service might be using this port.`);
    console.warn('You may need to free up port 8000 or change the port in backend/app/main.py');
    backendStarted = false;
    return;
  }

  console.log('Starting Python backend...');
  
  try {
    if (!app.isPackaged) {
      // Development mode - run Python script directly
      const scriptPath = path.join(__dirname, 'backend', 'app', 'main.py');
      console.log(`Starting backend with python ${scriptPath}`);
      
      pythonProcess = spawn('python', [scriptPath], {
        cwd: path.join(__dirname, 'backend'),
        env: { ...process.env, PYTHONUNBUFFERED: '1' } // Ensure unbuffered output
      });
    } else {
      // Production mode - run packaged executable
      let executableName = 'main';
      if (process.platform === 'win32') {
        executableName = 'main.exe';
      }
      
      const executablePath = path.join(process.resourcesPath, 'backend', executableName);
      const executableDir = path.join(process.resourcesPath, 'backend');
      
      console.log(`Starting packaged backend: ${executablePath}`);
      pythonProcess = spawn(executablePath, [], {
        cwd: executableDir,
        env: { ...process.env, PYTHONUNBUFFERED: '1' } // Ensure unbuffered output
      });
    }

    if (pythonProcess) {
      backendStarted = true;
      
      pythonProcess.stdout.on('data', (data) => {
        console.log(`Backend stdout: ${data}`);
      });

      pythonProcess.stderr.on('data', (data) => {
        console.error(`Backend stderr: ${data}`);
        // Check for specific messages that indicate the server is actually ready
        const output = data.toString();
        if (output.includes('Application startup complete') || 
            output.includes('Uvicorn running on http://0.0.0.0:8000')) {
          backendStarted = true;
        }
        
        // Check for address already in use error
        if (output.includes('address already in use')) {
          console.error('Port 8000 is already in use by another process.');
          backendStarted = false;
        }
      });

      pythonProcess.on('error', (error) => {
        console.error(`Failed to start backend: ${error}`);
        backendStarted = false;
      });

      pythonProcess.on('close', (code) => {
        console.log(`Backend process exited with code ${code}`);
        backendStarted = false;
      });
      
      // Give the process a moment to start
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  } catch (error) {
    console.error(`Error starting backend: ${error}`);
    backendStarted = false;
    throw error;
  }
}

app.on('ready', async () => {
  try {
    await startPythonBackend();
    
    // If backend is already running or was started successfully, wait for it to be ready
    if (backendStarted) {
      await waitForBackend(45, 2000); // Increase attempts and delay
      createWindow();
    } else {
      // Backend couldn't be started, but maybe it's already running
      const backendRunning = await checkBackendHealth();
      if (backendRunning) {
        backendStarted = true;
        createWindow();
      } else {
        console.error('Backend failed to start. Check the logs for details.');
        
        // Create window anyway to show error to user
        createWindow();
      }
    }
  } catch (error) {
    console.error('Failed to start application:', error);
    if (pythonProcess) {
      pythonProcess.kill();
    }
    // Create window anyway to show error to user
    createWindow();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', async () => {
  if (mainWindow === null) {
    createWindow();
  }
});

app.on('will-quit', () => {
  // Kill the python process if we started it
  if (pythonProcess) {
    console.log('Stopping backend...');
    if (process.platform === 'win32') {
      // On Windows, we need to kill the process tree
      spawn('taskkill', ['/pid', pythonProcess.pid, '/t', '/f']);
    } else {
      // On Unix, we can just kill the process
      pythonProcess.kill('SIGTERM');
    }
  }
});

// Prevent app from exiting when windows are closed
app.on('before-quit', (event) => {
  if (pythonProcess) {
    event.preventDefault();
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', pythonProcess.pid, '/t', '/f']);
    } else {
      pythonProcess.kill('SIGTERM');
    }
    setTimeout(() => {
      app.quit();
    }, 1000);
  }
});