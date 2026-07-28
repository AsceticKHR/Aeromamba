import json, glob, os
cands = ['/root/autodl-tmp/datasets/stage3_uavflow', '/root/autodl-tmp/datasets/uav-flow']
for base in cands:
    fs = glob.glob(os.path.join(base, '**', '*.json'), recursive=True)
    print('DIR', base, 'json_count', len(fs))
    if not fs:
        continue
    f = sorted(fs)[0]
    print(' file', f)
    d = json.load(open(f))
    if isinstance(d, dict):
        print(' top_keys', list(d.keys())[:12])
    traj = d if isinstance(d, list) else None
    fr = traj[0] if isinstance(traj, list) and traj else None
    if fr is None and isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                traj = v
                fr = v[0]
                print(' traj_key', k)
                break
    if isinstance(fr, dict):
        st = fr.get('state')
        print(' frame_keys', list(fr.keys())[:10])
        print(' state_len', len(st) if st else None)
        print(' state[0]', st[0] if st else None)
        print(' state[1]', st[1] if st and len(st) > 1 else None)
        print(' state[2]', st[2] if st and len(st) > 2 else 'ABSENT')
        xs = [abs(t['state'][0][0]) for t in traj[:40] if t.get('state')]
        zs = [abs(t['state'][0][2]) for t in traj[:40] if t.get('state')]
        print(' first40 |x|max', round(max(xs), 2) if xs else None,
              '|z|max', round(max(zs), 2) if zs else None)
    break
