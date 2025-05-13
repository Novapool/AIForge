// main.js
const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  // Load the HTML file
  mainWindow.loadFile('index.html');

  // Open DevTools in development
  if (!app.isPackaged) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function startPythonBackend() {
  const script = path.join(__dirname, 'backend', 'app', 'main.py');
  
  // In development, use python directly
  if (!app.isPackaged) {
    pythonProcess = spawn('python', [script], {
      cwd: path.join(__dirname, 'backend')
    });
  } else {
    // In production, use the packaged executable
    const executable = process.platform === 'win32' 
      ? path.join(process.resourcesPath, 'backend', 'main.exe')
      : path.join(process.resourcesPath, 'backend', 'main');
    
    pythonProcess = spawn(executable, [], {
      cwd: path.join(process.resourcesPath, 'backend')
    });
  }

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

app.on('ready', () => {
  startPythonBackend();
  
  // Wait a bit for the backend to start
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
  if (pythonProcess) {
    pythonProcess.kill();
  }
});