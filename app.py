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
    """Get statistics for all questions (latest attempt only)."""
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


def get_attempt_counts():
    """Get correct/wrong attempt counts for each question (for Anki-like ranking)."""
    db = get_db()

    rows = db.execute("""
        SELECT question_id,
               SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_count,
               SUM(CASE WHEN is_correct THEN 0 ELSE 1 END) as wrong_count
        FROM attempts
        GROUP BY question_id
    """).fetchall()

    counts = {}
    for row in rows:
        correct = row["correct_count"]
        wrong = row["wrong_count"]
        total = correct + wrong
        # Calculate wrong ratio: wrong / (correct + wrong)
        ratio = wrong / total if total > 0 else 0
        counts[row["question_id"]] = {
            "correct": correct,
            "wrong": wrong,
            "total": total,
            "wrong_ratio": ratio,
        }

    return counts


def get_categories():
    """Get ordered list of unique categories/sources."""
    seen = []
    for q in QUESTIONS:
        source = q.get("source", "Unknown")
        if source not in seen:
            seen.append(source)
    return seen


def get_questions_by_category():
    """Get questions grouped by category, preserving order."""
    categories = {}
    for idx, q in enumerate(QUESTIONS):
        source = q.get("source", "Unknown")
        if source not in categories:
            categories[source] = []
        categories[source].append(idx)
    return categories


def get_next_question_id(source=None):
    """Get next question ID using Anki-like ranking algorithm.

    Priority:
    1. Unanswered questions (in order within category)
    2. Answered questions sorted by wrong_ratio = wrong / (correct + wrong)

    Args:
        source: If provided, only consider questions from this source/category.
    """
    stats = get_question_stats()
    attempt_counts = get_attempt_counts()
    categories = get_categories()
    questions_by_cat = get_questions_by_category()

    def get_next_in_category(cat_ids):
        """Get next question in a category using Anki-like ranking."""
        # Priority 1: Unanswered questions (first in order)
        unanswered = [qid for qid in cat_ids if qid not in stats]
        if unanswered:
            return unanswered[0]

        # Priority 2: Answered questions sorted by wrong_ratio (highest first)
        answered = [qid for qid in cat_ids if qid in stats]
        if answered:
            # Sort by wrong_ratio descending, then by question order
            answered_with_ratio = []
            for qid in answered:
                ratio = attempt_counts.get(qid, {}).get("wrong_ratio", 0)
                answered_with_ratio.append((qid, ratio))

            # Sort by ratio descending
            answered_with_ratio.sort(key=lambda x: -x[1])

            # Return the question with highest wrong ratio
            return answered_with_ratio[0][0]

        return None

    # If source is specified, only look in that category
    if source and source in questions_by_cat:
        cat_ids = questions_by_cat[source]
        next_id = get_next_in_category(cat_ids)
        if next_id is not None:
            return next_id
        # Category complete, return first question of this category
        return cat_ids[0] if cat_ids else 0

    # Otherwise, find first category with unanswered/incorrect questions
    for cat in categories:
        cat_ids = questions_by_cat.get(cat, [])
        next_id = get_next_in_category(cat_ids)
        if next_id is not None:
            return next_id

    # All complete, return first question
    return 0


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
    """Show dashboard with all questions grouped by category."""
    stats = get_question_stats()
    attempt_counts = get_attempt_counts()
    overall = get_overall_stats()
    categories = get_categories()
    questions_by_cat = get_questions_by_category()

    # Build categories with questions and stats
    categories_data = []
    for cat in categories:
        cat_question_ids = questions_by_cat.get(cat, [])
        cat_questions = []
        cat_correct = 0
        cat_incorrect = 0
        cat_unanswered = 0

        for idx in cat_question_ids:
            q = QUESTIONS[idx]
            status = "unanswered"
            if idx in stats:
                status = "correct" if stats[idx]["is_correct"] else "incorrect"

            if status == "correct":
                cat_correct += 1
            elif status == "incorrect":
                cat_incorrect += 1
            else:
                cat_unanswered += 1

            # Get attempt counts for Anki-like stats
            q_attempts = attempt_counts.get(idx, {})

            cat_questions.append(
                {
                    "id": idx,
                    "question": q["question"],
                    "source": q["source"],
                    "status": status,
                    "correct_answer": q["answer"] if status != "unanswered" else None,
                    "selected_answer": stats[idx]["selected_answer"]
                    if idx in stats
                    else None,
                    "attempts": {
                        "correct": q_attempts.get("correct", 0),
                        "wrong": q_attempts.get("wrong", 0),
                        "total": q_attempts.get("total", 0),
                        "wrong_ratio": round(q_attempts.get("wrong_ratio", 0) * 100),
                    },
                }
            )

        categories_data.append(
            {
                "name": cat,
                "questions": cat_questions,
                "total": len(cat_questions),
                "correct": cat_correct,
                "incorrect": cat_incorrect,
                "unanswered": cat_unanswered,
                "complete": cat_unanswered == 0 and cat_incorrect == 0,
            }
        )

    return render_template(
        "dashboard.html", categories=categories_data, stats=overall
    )


@app.route("/quiz")
def quiz():
    """Display a question, following category order."""
    # Get question ID from URL or pick next one
    question_id = request.args.get("q", type=int)
    source = request.args.get("source")

    if question_id is None:
        question_id = get_next_question_id(source=source)
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
    # Check if same source filtering is requested
    same_source = request.args.get("same_source")
    source = request.args.get("source") if same_source else None

    # Clear session answer state
    for key in list(session.keys()):
        if key.startswith(("answered_", "selected_", "correct_")):
            session.pop(key, None)

    # Get next question ID with optional source filter
    question_id = get_next_question_id(source=source)
    return redirect(url_for("quiz", q=question_id))


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


def save_questions():
    """Save QUESTIONS to the JSON file."""
    questions_path = Path(__file__).parent / "questions.json"
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump(QUESTIONS, f, indent=2, ensure_ascii=False)


@app.route("/api/questions/import", methods=["POST"])
def import_questions():
    """Import questions from JSON. Accepts an array of question objects."""
    global QUESTIONS

    data = request.get_json()
    if not data:
        return {"error": "No JSON data provided"}, 400

    if not isinstance(data, list):
        return {"error": "Expected a JSON array of questions"}, 400

    # Validate required fields
    required_fields = ["source", "question", "options", "answer"]
    for i, q in enumerate(data):
        missing = [f for f in required_fields if f not in q]
        if missing:
            return {"error": f"Question {i}: missing fields {missing}"}, 400

    # Add questions to the list
    added_count = len(data)
    QUESTIONS.extend(data)
    save_questions()

    return {"message": f"Successfully imported {added_count} questions", "total": len(QUESTIONS)}


@app.route("/api/questions/<int:question_id>", methods=["DELETE"])
def delete_question(question_id):
    """Delete a question by its index."""
    global QUESTIONS

    if question_id < 0 or question_id >= len(QUESTIONS):
        return {"error": "Question not found"}, 404

    # Remove the question
    deleted = QUESTIONS.pop(question_id)

    # Clean up attempts for this question and re-index higher IDs
    db = get_db()
    db.execute("DELETE FROM attempts WHERE question_id = ?", (question_id,))
    # Decrement question_id for all attempts with higher IDs
    db.execute(
        "UPDATE attempts SET question_id = question_id - 1 WHERE question_id > ?",
        (question_id,),
    )
    db.commit()

    save_questions()

    return {
        "message": f"Deleted question: {deleted['question'][:50]}...",
        "remaining": len(QUESTIONS),
    }


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
