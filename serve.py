"""Run the API with Waitress, bound to localhost unless explicitly overridden."""

import argparse

from waitress import serve

from predict import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9696)
    args = parser.parse_args()
    serve(create_app(), host=args.host, port=args.port, threads=2, max_request_body_size=16384)
