"""
SAM 3D Quiz Web Application
A multiple choice quiz webapp built with Flask with SQLite persistence.
"""

import atexit
import json
import random
import socket
import sqlite3
from pathlib import Path

from flask import Flask, g, redirect, render_template, request, session, url_for
from zeroconf import IPVersion, ServiceInfo, Zeroconf

app = Flask(__name__)
app.secret_key = "sam3d-quiz-secret-key-2024"

DATABASE = Path(__file__).parent / "quiz.db"


# Load questions from JSON file
def load_questions():
    questions_path = Path(__file__).parent / "questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)


QUESTIONS = load_questions()


# Database functions
def get_db():
    """Get database connection for current request."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    """Close database connection at end of request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Initialize database tables."""
    db = sqlite3.connect(DATABASE)
    db.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL,
            selected_answer TEXT NOT NULL,
            is_correct BOOLEAN NOT NULL,
            attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    db.close()


def get_question_stats():
    """Get statistics for all questions."""
    db = get_db()

    # Get the latest attempt for each question
    rows = db.execute("""
        SELECT question_id, selected_answer, is_correct, attempted_at
        FROM attempts a1
        WHERE attempted_at = (
            SELECT MAX(attempted_at) 
            FROM attempts a2 
            WHERE a2.question_id = a1.question_id
        )
        ORDER BY question_id
    """).fetchall()

    stats = {}
    for row in rows:
        stats[row["question_id"]] = {
            "selected_answer": row["selected_answer"],
            "is_correct": bool(row["is_correct"]),
            "attempted_at": row["attempted_at"],
        }

    return stats


def get_next_question_id():
    """Get next question ID, prioritizing unanswered questions."""
    stats = get_question_stats()
    all_ids = list(range(len(QUESTIONS)))

    # Separate into unanswered, incorrect, and correct
    unanswered = [qid for qid in all_ids if qid not in stats]
    incorrect = [
        qid for qid in all_ids if qid in stats and not stats[qid]["is_correct"]
    ]
    correct = [qid for qid in all_ids if qid in stats and stats[qid]["is_correct"]]

    # Priority: unanswered > incorrect > correct
    if unanswered:
        return random.choice(unanswered)
    elif incorrect:
        return random.choice(incorrect)
    else:
        return random.choice(correct) if correct else random.choice(all_ids)


def record_attempt(question_id, selected_answer, is_correct):
    """Record a question attempt."""
    db = get_db()
    db.execute(
        "INSERT INTO attempts (question_id, selected_answer, is_correct) VALUES (?, ?, ?)",
        (question_id, selected_answer, is_correct),
    )
    db.commit()


def get_overall_stats():
    """Get overall quiz statistics."""
    stats = get_question_stats()
    total = len(QUESTIONS)
    answered = len(stats)
    correct = sum(1 for s in stats.values() if s["is_correct"])
    incorrect = answered - correct
    unanswered = total - answered

    return {
        "total": total,
        "answered": answered,
        "correct": correct,
        "incorrect": incorrect,
        "unanswered": unanswered,
        "percentage": round((correct / total) * 100, 1) if total > 0 else 0,
    }


def reset_all_progress():
    """Clear all attempts from database."""
    db = get_db()
    db.execute("DELETE FROM attempts")
    db.commit()


# Initialize database on startup
init_db()


@app.route("/")
def index():
    """Redirect to dashboard."""
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    """Show dashboard with all questions and their status."""
    stats = get_question_stats()
    overall = get_overall_stats()

    # Build question list with status
    questions_with_status = []
    for idx, q in enumerate(QUESTIONS):
        status = "unanswered"
        if idx in stats:
            status = "correct" if stats[idx]["is_correct"] else "incorrect"

        questions_with_status.append(
            {
                "id": idx,
                "question": q["question"],
                "source": q["source"],
                "status": status,
                "correct_answer": q["answer"] if status != "unanswered" else None,
                "selected_answer": stats[idx]["selected_answer"]
                if idx in stats
                else None,
            }
        )

    return render_template(
        "dashboard.html", questions=questions_with_status, stats=overall
    )


@app.route("/quiz")
def quiz():
    """Display a question (random, prioritizing unanswered)."""
    # Get question ID from URL or pick next one
    question_id = request.args.get("q", type=int)

    if question_id is None:
        question_id = get_next_question_id()
        return redirect(url_for("quiz", q=question_id))

    if question_id < 0 or question_id >= len(QUESTIONS):
        return redirect(url_for("dashboard"))

    question = QUESTIONS[question_id]
    overall = get_overall_stats()

    # Check if already answered in this session
    answered = session.get(f"answered_{question_id}", False)
    selected_answer = session.get(f"selected_{question_id}")
    is_correct = session.get(f"correct_{question_id}")

    return render_template(
        "quiz.html",
        question=question,
        question_id=question_id,
        current_num=question_id + 1,
        total=len(QUESTIONS),
        stats=overall,
        answered=answered,
        selected_answer=selected_answer,
        is_correct=is_correct,
    )


@app.route("/submit", methods=["POST"])
def submit_answer():
    """Process answer submission."""
    question_id = request.form.get("question_id", type=int)
    selected = request.form.get("answer")

    if question_id is None or selected is None:
        return redirect(url_for("dashboard"))

    if question_id < 0 or question_id >= len(QUESTIONS):
        return redirect(url_for("dashboard"))

    question = QUESTIONS[question_id]
    is_correct = selected == question["answer"]

    # Record attempt in database
    record_attempt(question_id, selected, is_correct)

    # Store in session for immediate display
    session[f"answered_{question_id}"] = True
    session[f"selected_{question_id}"] = selected
    session[f"correct_{question_id}"] = is_correct

    return redirect(url_for("quiz", q=question_id))


@app.route("/next")
def next_question():
    """Get next random question (prioritizing unanswered)."""
    # Clear session answer state
    for key in list(session.keys()):
        if key.startswith(("answered_", "selected_", "correct_")):
            session.pop(key, None)

    return redirect(url_for("quiz"))


@app.route("/question/<int:question_id>")
def view_question(question_id):
    """View a specific question."""
    # Clear session answer state for fresh view
    session.pop(f"answered_{question_id}", None)
    session.pop(f"selected_{question_id}", None)
    session.pop(f"correct_{question_id}", None)

    return redirect(url_for("quiz", q=question_id))


@app.route("/reset", methods=["POST"])
def reset():
    """Reset all progress."""
    reset_all_progress()
    session.clear()
    return redirect(url_for("dashboard"))


def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Create a socket to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_tailscale_ip():
    """Get the Tailscale IP if available."""
    import subprocess
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def register_mdns(hostname: str, port: int):
    """Register the webapp with mDNS so it's accessible as hostname.local"""
    local_ip = get_local_ip()
    ip_bytes = socket.inet_aton(local_ip)
    
    # Create zeroconf instance with IPv4 only for better iOS compatibility
    zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
    
    # Register HTTP service - this helps with service discovery
    http_service = ServiceInfo(
        "_http._tcp.local.",
        f"{hostname}._http._tcp.local.",
        addresses=[ip_bytes],
        port=port,
        properties={"path": "/"},
        server=f"{hostname}.local.",
    )
    
    # Also register a "workstation" service which better advertises the hostname
    # This is what macOS uses and iOS recognizes well
    workstation_service = ServiceInfo(
        "_workstation._tcp.local.",
        f"{hostname}._workstation._tcp.local.",
        addresses=[ip_bytes],
        port=port,
        properties={},
        server=f"{hostname}.local.",
    )
    
    # Register both services
    zeroconf.register_service(http_service)
    zeroconf.register_service(workstation_service)
    
    print(f"\n{'='*50}")
    print(f"🌐 mDNS registered: http://{hostname}.local:{port}")
    print(f"📍 Local IP: http://{local_ip}:{port}")
    print(f"💡 On iOS, try: http://{hostname}.local:{port}")
    print(f"   Or directly: http://{local_ip}:{port}")
    print(f"{'='*50}\n")
    
    # Cleanup on exit
    def cleanup():
        print("\n🔌 Unregistering mDNS services...")
        zeroconf.unregister_service(http_service)
        zeroconf.unregister_service(workstation_service)
        zeroconf.close()
    
    atexit.register(cleanup)
    return zeroconf


def main():
    """Entry point for uv run."""
    hostname = "thangquiz"
    port = 5000
    
    # Register mDNS service (for local network)
    register_mdns(hostname, port)
    
    # Show Tailscale IP if available
    tailscale_ip = get_tailscale_ip()
    if tailscale_ip:
        print(f"🔗 Tailscale: http://{tailscale_ip}:{port}")
        print(f"{'='*50}\n")
    
    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)


if __name__ == "__main__":
    main()
