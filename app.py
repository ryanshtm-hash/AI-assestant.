from flask import Flask, render_template, request, Response, stream_with_context
import requests
import json

app = Flask(__name__)

# Change this to a model name shown by: ollama list
MODEL = "gemma3:1b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

SYSTEM_PROMPT = """
You are StudyMate AI, an excellent study tutor for Class 9 students.

Explain concepts in clear, simple English.
For Maths and Physics, show formulas and step-by-step solutions.
For Chemistry and Biology, give accurate explanations and examples.
Create quizzes, revision notes, and practice questions when requested.
Use readable formatting and ask a clarifying question when necessary.
Never invent facts when you are uncertain.
"""

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    history = data.get("history", [])

    if not question:
        return Response(
            json.dumps({"error": "Please enter a question."}) + "\n",
            status=400, mimetype="application/x-ndjson"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Keep a limited amount of recent conversation context.
    if isinstance(history, list):
        for item in history[-10:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({
                    "role": role,
                    "content": content[:5000]
                })

    messages.append({"role": "user", "content": question[:5000]})

    def generate():
        upstream = None
        try:
            upstream = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": True
                },
                stream=True,
                timeout=(10, 600)
            )
            upstream.raise_for_status()

            for line in upstream.iter_lines():
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if item.get("error"):
                    yield json.dumps({
                        "error": item["error"]
                    }) + "\n"
                    return

                token = item.get("message", {}).get("content", "")
                if token:
                    yield json.dumps({"token": token}) + "\n"

                if item.get("done"):
                    yield json.dumps({"done": True}) + "\n"
                    return

        except requests.exceptions.ConnectionError:
            yield json.dumps({
                "error": "Cannot connect to Ollama. Open Ollama and check port 11434."
            }) + "\n"
        except requests.exceptions.Timeout:
            yield json.dumps({
                "error": "Ollama timed out. Try again or use a smaller model."
            }) + "\n"
        except requests.exceptions.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                detail = exc.response.text[:500]
            yield json.dumps({
                "error": "Ollama request failed: " + (detail or str(exc))
            }) + "\n"
        except GeneratorExit:
            raise
        finally:
            if upstream is not None:
                upstream.close()

    return Response(
        stream_with_context(generate()),
        content_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
