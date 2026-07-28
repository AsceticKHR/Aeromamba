import json, glob, os, math

RES = r"C:\Users\user\学习\UAV source code\UAV-Flow-Eval\results\aerov2_full"
GT = r"C:\Users\user\学习\UAV source code\UAV-Flow-Eval\test_jsons"

files = sorted(glob.glob(os.path.join(RES, "*.json")))
print(f"completed tasks so far: {len(files)}")
for f in files[:6]:
    base = os.path.splitext(os.path.basename(f))[0]
    log = json.load(open(f, encoding="utf-8"))
    task = json.load(open(os.path.join(GT, base + ".json"), encoding="utf-8"))
    instr = task.get("instruction")
    ref = task["reference_path_preprocessed"]
    mxyz = [it["state"][0] for it in log]
    myaw = [it["state"][1][1] for it in log]
    g_end = ref[-1]
    m_end = mxyz[-1]
    gx, gy, gz, gyaw = g_end[0], g_end[1], g_end[2], g_end[4]
    mx, my, mz = m_end[0], m_end[1], m_end[2]
    myaw_e = myaw[-1]
    gmag = math.hypot(gx, gy)
    mmag = math.hypot(mx, my)
    end_xy = math.hypot(mx - gx, my - gy)
    # per-step increments of model (to see if it moves in tiny/constant steps)
    steps = []
    for i in range(1, len(mxyz)):
        d = math.dist(mxyz[i][:2], mxyz[i-1][:2])
        steps.append(d)
    avg_step = sum(steps)/len(steps) if steps else 0
    print(f"== {base} | {instr[:55]}")
    print(f"   steps model={len(mxyz)} gt={len(ref)}  avg_model_xy_step={avg_step:.1f}cm")
    print(f"   GT  end=({gx:.0f},{gy:.0f},{gz:.0f}) |xy|={gmag:.0f} yaw={gyaw:.1f}")
    print(f"   MDL end=({mx:.0f},{my:.0f},{mz:.0f}) |xy|={mmag:.0f} yaw={myaw_e:.1f}")
    print(f"   end_xy_err={end_xy:.0f}cm  |xy|ratio(m/g)={mmag/(gmag+1e-6):.2f}")
