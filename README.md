# SAM 3D Quiz App

A multiple choice quiz web application built with Flask. Features SQLite persistence, progress tracking, and smart question prioritization (unanswered → incorrect → correct).

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
# Clone and enter the directory
cd webapp_multiple_choice

# Install dependencies with uv
uv sync
```

## Running the App

### Quick Start (Foreground)

```bash
uv run quiz
```

The app will display access URLs:
- **Local network**: `http://<local-ip>:5000`
- **mDNS** (same network): `http://thangquiz.local:5000`
- **Tailscale**: `http://<tailscale-ip>:5000`

---

## Running in Background

### Option 1: `nohup` (Simple)

Run the app and close your terminal without stopping it:

```bash
nohup uv run quiz > quiz.log 2>&1 &
```

To stop it later:
```bash
pkill -f "quiz"
```

### Option 2: `screen` (Recommended for SSH)

```bash
# Start a screen session
screen -S quiz

# Run the app
uv run quiz

# Detach: press Ctrl+A, then D

# Reattach later
screen -r quiz
```

### Option 3: `tmux` (Alternative to screen)

```bash
# Start tmux session
tmux new -s quiz

# Run the app
uv run quiz

# Detach: press Ctrl+B, then D

# Reattach later
tmux attach -t quiz
```

### Option 4: systemd Service (Production)

Create a service file:

```bash
sudo nano /etc/systemd/system/quiz.service
```

Add this content (adjust paths):

```ini
[Unit]
Description=SAM 3D Quiz App
After=network.target

[Service]
Type=simple
User=ptthang
WorkingDirectory=/home/ptthang/cursor/webapp_multiple_choice
ExecStart=/home/ptthang/.local/bin/uv run quiz
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable quiz
sudo systemctl start quiz

# Check status
sudo systemctl status quiz

# View logs
journalctl -u quiz -f
```

---

## Accessing via Tailscale

Since mDNS doesn't work over Tailscale, use one of these methods:

### Option A: Direct IP

```bash
# Get your Tailscale IP
tailscale ip -4

# Access from iOS/other device
# http://<tailscale-ip>:5000
```

### Option B: Tailscale Serve (Recommended)

Expose the app with a clean HTTPS URL:

```bash
# Start serving (runs in background)
tailscale serve --bg 5000

# Check the URL
tailscale serve status

# Access at: https://<machine-name>.<tailnet>.ts.net
```

To stop:
```bash
tailscale serve --https=443 off
```

### Option C: MagicDNS

1. Go to [Tailscale Admin Console](https://login.tailscale.com/admin/dns)
2. Enable **MagicDNS**
3. Access via: `http://<machine-hostname>:5000`

---

## Features

- **Dashboard**: View all questions with status (✓ correct, ✗ incorrect, ○ unanswered)
- **Smart Prioritization**: Automatically serves unanswered questions first, then incorrect ones
- **Progress Persistence**: SQLite database saves your progress
- **Reset**: Clear all progress and start fresh

## File Structure

```
├── app.py           # Main Flask application
├── questions.json   # Quiz questions
├── quiz.db          # SQLite database (auto-created)
├── templates/
│   ├── dashboard.html
│   └── quiz.html
└── pyproject.toml   # Dependencies
```

## Adding Questions

Edit `questions.json`:

```json
[
  {
    "id": 1,
    "source": "Chapter 1",
    "context": "Optional context for the question",
    "question": "Your question text?",
    "options": {
      "A": "Option 1",
      "B": "Option 2",
      "C": "Option 3",
      "D": "Option 4"
    },
    "answer": "B",
    "explanation": "Why B is correct..."
  }
]
```
