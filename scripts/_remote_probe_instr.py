import base64, json, urllib.request, math

URL = "http://127.0.0.1:5007"
IMG = "/root/autodl-tmp/datasets/stage3_uavflow/2025-04-02_15-17-46/000000.jpg"

with open(IMG, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


INSTRS = [
    "Fly straight forward.",
    "Turn left.",
    "Turn right.",
    "Turn to the direction of the person.",
    "Climb up to a higher altitude.",
    "Descend and fly downward.",
    "Stop and hold position.",
]

print("proprio=[0,0,0,0] cold-start (velocity=0). action=[fwd_cm,right_cm,up_cm,yaw_rad] cumulative from anchor")
print("=" * 100)
for instr in INSTRS:
    post("/reset", {})
    out = post("/predict", {"image": b64, "proprio": [0, 0, 0, 0], "instr": instr})
    act = out.get("action", [])
    if not act:
        print(f"[{instr:42s}] NO ACTION {out}")
        continue
    a0 = act[0]
    aN = act[-1]
    yaw0_deg = a0[3] * 180.0 / math.pi
    yawN_deg = aN[3] * 180.0 / math.pi
    print(f"[{instr:42s}] wp0=({a0[0]:7.1f},{a0[1]:7.1f},{a0[2]:6.1f},{yaw0_deg:6.1f}deg) "
          f"wpEnd=({aN[0]:7.1f},{aN[1]:7.1f},{aN[2]:6.1f},{yawN_deg:6.1f}deg)")

# proprio sensitivity: same neutral instruction, vary current yaw / height in proprio
print("\nPROPRIO sensitivity (instr='Fly to the target.'):")
for name, pro in [("origin", [0, 0, 0, 0]), ("moved+facing90", [500, 0, 0, 90]),
                  ("high", [0, 0, 800, 0])]:
    post("/reset", {})
    out = post("/predict", {"image": b64, "proprio": pro, "instr": "Fly to the target."})
    a0 = out["action"][0]
    print(f"  proprio={name:14s} -> wp0=({a0[0]:7.1f},{a0[1]:7.1f},{a0[2]:6.1f},{a0[3]*180/math.pi:6.1f}deg)")
