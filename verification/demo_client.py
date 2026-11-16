"""Temporary browser client for live API screenshots, using only the standard library."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', default='http://127.0.0.1:9696')
    parser.add_argument('--port', type=int, default=19876)
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
    target = urlsplit(args.upstream)
    if target.scheme != 'http' or target.hostname != '127.0.0.1' or target.path or target.query or target.username:
        parser.error('Use a plain http://127.0.0.1:PORT upstream.')
    sample = (Path(__file__).resolve().parents[1] / 'sample_request.json').read_text()
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Live Docker API request</title>
<style>body{font:17px system-ui;max-width:1040px;margin:35px auto;padding:0 24px;color:#17202b}
h1{font-size:30px}p{line-height:1.5}textarea,pre{box-sizing:border-box;width:100%;font:15px/1.45 monospace;
padding:15px;border:1px solid #c7d0da;border-radius:6px;background:#f6f8fa}textarea{height:245px}
button{font:inherit;background:#165b9c;color:white;border:0;padding:12px 20px;border-radius:5px;cursor:pointer}
#status{font-weight:650;margin:18px 0 8px}#response{min-height:130px}small{color:#516174}</style>
<h1>Concrete Strength: live Docker API request</h1>
<p>This page sends a real HTTP request to the running Docker container.<br>
Upstream: <code>__UPSTREAM__/predict</code> (localhost; SSH forwarding when checking a remote server).</p>
<label for="payload"><b>POST /predict</b> &middot; Content-Type: application/json</label>
<textarea id="payload" spellcheck="false">__SAMPLE__</textarea>
<button id="send">Send live request</button>
<div id="status">No request sent yet.</div><pre id="response"></pre>
<small>Amounts: kg/m³. Age: days. Predicted strength: MPa. Educational model.</small>
<script>document.querySelector('#send').onclick=async()=>{
const button=document.querySelector('#send');button.disabled=true;
document.querySelector('#status').textContent='Sending request...';
try{const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
body:document.querySelector('#payload').value});const data=await response.json();
document.querySelector('#status').textContent='HTTP '+response.status+' · received '+new Date().toISOString();
document.querySelector('#response').textContent=JSON.stringify(data,null,2);
}catch(error){document.querySelector('#status').textContent='Request failed';
document.querySelector('#response').textContent=String(error);}finally{button.disabled=false;}};</script></html>'''
    # Both substitutions are trusted static project values; response text uses textContent.
    import html
    page = page.replace('__UPSTREAM__', html.escape(args.upstream)).replace('__SAMPLE__', html.escape(sample)).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/':
                self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(page))); self.end_headers(); self.wfile.write(page)

        def do_POST(self):
            if self.path != '/predict':
                self.send_error(404); return
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 8192:
                self.send_error(413); return
            payload = self.rfile.read(length)
            req = Request(args.upstream + '/predict', data=payload, headers={'Content-Type': 'application/json'})
            try:
                with urlopen(req, timeout=15) as response:
                    status, body = response.status, response.read(16384)
            except HTTPError as error:
                status, body = error.code, error.read(16384)
            if args.log:
                try:
                    request_record = json.loads(payload)
                except (ValueError, RecursionError):
                    request_record = payload.decode('utf-8', errors='replace')
                with args.log.open('a') as stream:
                    stream.write(json.dumps({'upstream': args.upstream + '/predict', 'request': request_record,
                                             'status': status, 'response': json.loads(body)}) + '\n')
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    print(f'Live verification client: http://127.0.0.1:{args.port}', flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
