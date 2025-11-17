from flask import Flask, request, jsonify
from utils import generate_from_retrieval

app = Flask(__name__)

# Optional safety caps
MAX_TOP_K = 100

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/query", methods=["POST"])
def query():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON body"}), 400

    q = payload.get("query")
    if not q or not isinstance(q, str) or q.strip() == "":
        return jsonify({"error": "query is required and must be a non-empty string"}), 400

    try:
        top_k = int(payload.get("top_k", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400

    if top_k <= 0:
        return jsonify({"error": "top_k must be a positive integer"}), 400
    if top_k > MAX_TOP_K:
        top_k = MAX_TOP_K

    method = payload.get("method", "semantic")

    try:
        gen = generate_from_retrieval(q, top_k=top_k, method=method)
    except Exception as e:
        return jsonify({"error": f"generation failed: {str(e)}"}), 500

    response = {
        "query": q,
        "method": method,
        "top_k": top_k,
        "answer": gen.get("answer"),
        "retrieved": gen.get("retrieved", []),
        "paraphrase_used": gen.get("paraphrase_used", True)
    }
    if "paraphrase_error" in gen:
        response["paraphrase_error"] = gen["paraphrase_error"]

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)