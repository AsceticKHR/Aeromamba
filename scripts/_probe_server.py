"""Serving-time smoke: send the same gray image with different instructions to
the running server_v2 and print the returned action chunk (episode-local poses).
Confirms the deployed proprio-free-query policy responds to instructions."""
import base64, io, json, urllib.request
from PIL import Image

URL = "http://127.0.0.1:5007"


def b64_gray():
    img = Image.new("RGB", (384, 384), (120, 120, 120))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def post(path, payload):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=15).read())


img = b64_gray()
for instr in ["fly straight forward", "turn left", "turn right", "ascend / fly up"]:
    post("/reset", {})
    r = post("/predict", {"image": img, "proprio": [0, 0, 0, 0], "instr": instr})
    act = r.get("action", [])
    first = [round(v, 3) for v in act[0]] if act else None
    last = [round(v, 3) for v in act[-1]] if act else None
    print(f"instr={instr!r:24s} n={len(act)} first={first} last={last}")
