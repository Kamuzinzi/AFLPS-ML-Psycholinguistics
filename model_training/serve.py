import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from deployment import load_bundle, predict_text


def create_handler(bundle):
    class PredictionHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._write_json(200, {"status": "ok"})
            else:
                self._write_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                self._write_json(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                text = payload["text"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("text must be a non-empty string")
                result = predict_text(text, bundle)
                self._write_json(
                    200,
                    {"dataset": bundle["dataset"], "predictions": result},
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                self._write_json(400, {"error": str(error)})

        def log_message(self, format_string, *args):
            print(f"{self.client_address[0]} - {format_string % args}")

        def _write_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return PredictionHandler


def main():
    parser = argparse.ArgumentParser(description="Serve personality predictions.")
    parser.add_argument("bundle", help="Path to a saved .pt model bundle.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        create_handler(bundle),
    )
    print(f"Serving predictions at http://{args.host}:{args.port}")
    print("POST JSON to /predict with: {\"text\": \"...\"}")
    server.serve_forever()


if __name__ == "__main__":
    main()
