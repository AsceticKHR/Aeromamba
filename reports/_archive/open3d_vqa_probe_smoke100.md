# Open3D-VQA Official-style Probe

> **污染声明**：Stage2 v2 训练已含全部 Open3D-VQA → 下表为 **in-distribution probe**，非干净官方 hold-out。

- checkpoint: `/mnt/c/Users/user/学习/UAV source code/Aeromamba/checkpoints/stage2_v2/best_slim.pth`
- probe: `/mnt/c/Users/user/学习/UAV source code/Aeromamba/data/open3d_vqa_probe/test_official_probe.json`
- n=100, elapsed=200s

## By domain

| domain | n | exact | partial |
|---|---|---|---|
| real | 100 | 36.0% | 40.0% |

## By task bucket

| bucket | n | exact | partial |
|---|---|---|---|
| ALL | 100 | 36.0% | 40.0% |
| direction | 24 | 45.8% | 54.2% |
| distance_quant | 16 | 12.5% | 25.0% |
| other_qual | 36 | 38.9% | 38.9% |
| tf_qual | 24 | 37.5% | 37.5% |

## Examples
- [OK] `direction` RealworldUAV/Lab — GT: Indeed, the white building with vertical windows is positioned on the left side of the curved white railing structure.
  - Pred: Incorrect, the white building with vertical windows is not on the left side of the curved white railing structure.
- [OK] `direction` RealworldUAV/Lab — GT: In fact, the white building with vertical windows is either to the left of or directly aligned with the curved white rai
  - Pred: In fact, the white building with vertical windows is either to the left of or directly aligned with the curved white rai
- [OK] `other_qual` RealworldUAV/Lab — GT: Correct, the white building with vertical windows is located above the curved white railing structure.
  - Pred: Correct, the white building with vertical windows is located above the curved white railing structure.
- [OK] `other_qual` RealworldUAV/Lab — GT: Actually, the white building with vertical windows is either above or at the same elevation as the curved white railing 
  - Pred: Actually, the white building with vertical windows is either above or at the same elevation as the curved white railing 
- [OK] `tf_qual` RealworldUAV/Lab — GT: No, the curved white railing structure is behind the white building with vertical windows.
  - Pred: No, it is not.
- [MISS] `tf_qual` RealworldUAV/Lab — GT: Yes, the white building with vertical windows is behind the curved white railing structure.
  - Pred: No, the curved white railing structure is in front of the white building with vertical windows.
- [OK] `direction` RealworldUAV/Lab — GT: Positioned to the left is white building with vertical windows.
  - Pred: Positioned to the left is white building with vertical windows.
- [OK] `direction` RealworldUAV/Lab — GT: curved white railing structure is more to the right.
  - Pred: curved white railing structure is more to the right.
- [MISS] `other_qual` RealworldUAV/Lab — GT: Positioned higher is white building with vertical windows.
  - Pred: curved white railing structure is higher up.
- [MISS] `other_qual` RealworldUAV/Lab — GT: curved white railing structure is below.
  - Pred: curved white railing structure is lower down.
- [MISS] `other_qual` RealworldUAV/Lab — GT: From the viewer's perspective, curved white railing structure appears closer.
  - Pred: curved white railing structure is more closer to the viewer.
- [MISS] `other_qual` RealworldUAV/Lab — GT: From the viewer's perspective, white building with vertical windows appears further.
  - Pred: curved white railing structure is further to the viewer.