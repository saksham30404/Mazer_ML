# Mazer_ML
AI Music Generator
Overview
A Python-based tool to generate music from natural language prompts, supporting styles like jazz, electronic, and rock, with emotions like mystery or energy. Features a FastAPI web interface and CLI.
Features

Parse natural language prompts for style, emotion, and duration.
Supports multiple styles and emotions with unique scales.
Synthesizes instruments (piano, drums, etc.) with ADSR envelopes.
Applies EQ, compression, and reverb for polished audio.
FastAPI endpoints for remote access.
CLI for local music generation.

Requirements

Python 3.8+
Dependencies: numpy, pretty_midi, soundfile, fastapi, uvicorn, pydantic, requests, scipy, tqdm

Installation

Clone the repo:git clone https://github.com/your-username/ai-music-generator.git
cd ai-music-generator


Install dependencies:pip install -r requirements.txt



Usage
CLI

Run:python music_ai.py


Enter a prompt (e.g., "Create a dreamy synthwave track for 90 seconds").
Music is saved in generated_music/.

API

Start the server:uvicorn music_ai:app --reload

Endpoints:
GET /: API info.
GET /styles: List styles.
GET /emotions: List emotions.
POST /generate: Generate music. Example:{"prompt": "Make a relaxing ambient piece for 2 minutes"}

Example

CLI: "Make a mysterious jazz song"
API:curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"prompt":"Make a dreamy synthwave track"}' --output music.wav

Limitations

Max duration: 10 minutes.
Some styles may sound similar.
Basic drum synthesis.

Contributing

Fork and create a branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add feature").
Push (git push origin feature/your-feature).
Open a pull request.

License
MIT License. See LICENSE file.
Contact
Open a GitHub issue for questions or feedback.
