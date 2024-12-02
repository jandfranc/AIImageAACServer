// server.js
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const multer = require('multer'); // For handling file uploads

const app = express();
const port = 3000;

// Middleware
app.use(bodyParser.json());
app.use(cors());

// Serve static files from the "images" and "audio" directories
app.use('/images', express.static(path.join(__dirname, 'images')));
app.use('/audio', express.static(path.join(__dirname, 'audio')));

// Ensure the directories exist
const ensureDirectoryExistence = (dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
};

ensureDirectoryExistence(path.join(__dirname, 'images'));
ensureDirectoryExistence(path.join(__dirname, 'audio'));

// Multer setup for audio uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'audio/');
  },
  filename: function (req, file, cb) {
    // Use timestamp and original name to avoid conflicts
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const ext = path.extname(file.originalname);
    cb(null, `${uniqueSuffix}${ext}`);
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // Limit files to 10MB
  fileFilter: (req, file, cb) => {
    // Accept only audio files
    if (file.mimetype.startsWith('audio/')) {
      cb(null, true);
    } else {
      cb(new Error('Only audio files are allowed!'), false);
    }
  }
});

// Function to generate and save images
const generateAndSaveImages = (text) => {
  const imagePaths = [];
  // Example: Create a simple image with text
  const canvas = createCanvas(200, 200);
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 200, 200);

  ctx.fillStyle = '#000000';
  ctx.font = '20px Arial';
  ctx.fillText(text, 50, 100);

  const filename = `image-${Date.now()}.png`;
  const filepath = path.join(__dirname, 'images', filename);
  const buffer = canvas.toBuffer('image/png');
  fs.writeFileSync(filepath, buffer);
  imagePaths.push(`http://localhost:${port}/images/${filename}`);

  return imagePaths;
};

// API Route to generate images
app.post('/generate-images', (req, res) => {
  const { text, token } = req.body;

  if (!text || !token) {
    return res.status(400).json({ error: 'Text and token are required' });
  }

  try {
    if (token !== 'expected-token') {
      throw new Error('Invalid token');
    }

    // Generate and save images dynamically
    const imageUrls = generateAndSaveImages(text);
    return res.json({ imageUrls });
  } catch (error) {
    return res.status(403).json({ error: error.message });
  }
});

// API Route to upload audio
// API Route to upload audio
app.post('/upload-audio', upload.single('audio'), (req, res) => {
  console.log("File received:", req.file); // Debugging file info
  console.log("Body:", req.body);          // Debugging other body data
  console.log("Headers:", req.headers);    // Debugging headers

  const token = req.headers.token; // Token from headers

  if (!req.file) {
    return res.status(400).json({ error: 'No audio file uploaded' });
  }

  if (!token) {
    fs.unlinkSync(req.file.path); // Clean up if no token
    return res.status(400).json({ error: 'Token is required' });
  }

  try {
    if (token !== 'expected-token') {
      fs.unlinkSync(req.file.path); // Clean up if token is invalid
      throw new Error('Invalid token');
    }

    const audioUrl = `http://localhost:${port}/audio/${req.file.filename}`;
    return res.json({ audioUrl });
  } catch (error) {
    return res.status(403).json({ error: error.message });
  }
});



// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
  console.log(`Serving images at http://localhost:${port}/images/`);
  console.log(`Serving audio at http://localhost:${port}/audio/`);
});
