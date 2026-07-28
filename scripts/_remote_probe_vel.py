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


def body_out(instr, prev_pose):
    """reset -> call1(prev_pose) sets prev -> call2([0,0,0,0]) => velocity nonzero,
    pose passthrough=0, cyaw=0 so returned action == pure body-frame model output."""
    post("/reset", {})
    post("/predict", {"image": b64, "proprio": prev_pose, "instr": instr})
    out = post("/predict", {"image": b64, "proprio": [0, 0, 0, 0], "instr": instr})
    return out["action"]


def summ(act):
    a0, aN = act[0], act[-1]
    return (f"wp0=({a0[0]:6.1f},{a0[1]:6.1f},{a0[2]:5.1f},{a0[3]*180/math.pi:6.1f}d) "
            f"wpEnd=({aN[0]:6.1f},{aN[1]:6.1f},{aN[2]:5.1f},{aN[3]*180/math.pi:6.1f}d)")


instr = "Turn left."
print("Induced velocity via prev_pose (call2 pose=0 => output is pure body-frame). instr='Turn left.'")
# prev_pose in client units [x_cm,y_cm,z_cm,yaw_deg]; velocity = (0 - prev)/exec_steps
cases = [
    ("vel=0 (prev=0)", [0, 0, 0, 0]),
    ("prev x=+300cm  -> vel -x (backward)", [300, 0, 0, 0]),
    ("prev x=-300cm  -> vel +x (forward)", [-300, 0, 0, 0]),
    ("prev yaw=+40d  -> vel -yaw", [0, 0, 0, 40]),
    ("prev yaw=-40d  -> vel +yaw", [0, 0, 0, -40]),
    ("prev z=+300cm  -> vel -z (down)", [0, 0, 300, 0]),
    ("prev z=-300cm  -> vel +z (up)", [0, 0, -300, 0]),
]
for name, prev in cases:
    print(f"  [{name:38s}] {summ(body_out(instr, prev))}")
