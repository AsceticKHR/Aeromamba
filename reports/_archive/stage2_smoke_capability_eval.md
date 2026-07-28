# Stage 2 能力系统性测评

- checkpoint: `/mnt/c/Users/user/学习/UAV source code/Aeromamba/checkpoints/stage2_v2/best_slim.pth`
- 数据: `stage2_mixed_data_v2.json` 验证集（split_seed=42，与训练一致）
- 时间: 2026-07-14 10:34:08

| source | 能力维度 | n | CLM loss | PPL | 生成精确率 | 部分正确率 |
|---|---|---|---|---|---|---|
| aerial_spatial | 俯视空间推理 | 20 | 0.283 | 1.33 | 20.0% | 20.0% |
| cognitive | 认知推理→指令解析 | 20 | 0.191 | 1.21 | 30.0% | 75.0% |
| general | 通用图文理解 | 20 | 1.702 | 5.48 | — | — |
| uav_motion | 指令→运动语义 | 20 | 0.065 | 1.07 | 90.0% | 85.0% |

## aerial_spatial 生成样例
- ❌ Q: <image>
Does the tall skyscraper with curved top appear on the right side of the tree with purple leaves visible? 
  - GT: In fact, the tall skyscraper with curved top is either to the left of or directly aligned with the tree with purple leav
  - Pred: Incorrect, the tall skyscraper with curved top is not on the right side of the tree with purple leaves visible.
- ❌ Q: <image>
If you are at modern building with grid facade, where will you find white building with irregular windows? 
  - GT: modern building with grid facade will find white building with irregular windows around the 11 o'clock direction.
  - Pred: modern building with grid facade will find white building with irregular windows around the 3 o'clock direction.
- ❌ Q: <image>
Can you confirm if the black curved street lamp posts is positioned below the blue and white billboard?
A.Yes.
B.No. 
  - GT: B.No
  - Pred: A.Yes
- ✅ Q: <image>
Which is below, the black metal fence with vertical bars or the gray abstract building silhouettes?
A:black metal fence with vertical bars
B:gray abstra
  - GT: A.black metal fence with vertical bars
  - Pred: A.black metal fence with vertical bars
- ✅ Q: <image>
Which is below, the distorted white building with windows or the tall rectangular gray structure?
A:distorted white building with windows
B:tall rectang
  - GT: B.tall rectangular gray structure
  - Pred: B.tall rectangular gray structure

## cognitive 生成样例
- ❌ Q: <image>
You control a drone in front of several gates. Task: "Fly through the gate with three" Resolve the task and state the exact flight command.
  - GT: Fly through the big triangle red gate with three
  - Pred: Fly through the big round blue gate with three
- ❌ Q: <image>
You control a drone in front of several gates. Task: "Fly through the gate with a letter D" Resolve the task and state the exact flight command.
  - GT: Fly through the big square red gate with a letter D
  - Pred: Fly through the big square blue gate with a letter D
- ❌ Q: <image>
You control a drone in front of several gates. Task: "Fly through the gate with a letter K" Resolve the task and state the exact flight command.
  - GT: Fly through the small triangle red gate with a letter K
  - Pred: Fly through the big triangle red gate with a letter K
- ❌ Q: <image>
You control a drone in front of several gates. Task: "Fly through the gate with a zebra" Resolve the task and state the exact flight command.
  - GT: Fly through the small square red gate with a zebra
  - Pred: Fly through the small round green gate with a zebra
- ❌ Q: <image>
You control a drone in front of several gates. Task: "Fly through the gate with a letter T" Resolve the task and state the exact flight command.
  - GT: Fly through the small triangle blue gate with a letter T
  - Pred: Fly through the big round red gate with a letter T

## uav_motion 生成样例
- ✅ Q: <image>
You are piloting a UAV. Instruction: "Back up from the tree on the current side" Based on the current view, in which direction should the UAV move overa
  - GT: The UAV should move backward while roughly holding altitude.
  - Pred: The UAV should move backward while roughly holding altitude.
- ❌ Q: <image>
You are piloting a UAV. Instruction: "Please make way to the streetlight from the left side" Based on the current view, in which direction should the UA
  - GT: The UAV should move forward while roughly holding altitude.
  - Pred: The UAV should move forward and to the left while roughly holding altitude.
- ✅ Q: <image>
You are piloting a UAV. Instruction: "Shift to the right side" Based on the current view, in which direction should the UAV move overall?
  - GT: The UAV should move to the right while roughly holding altitude.
  - Pred: The UAV should move to the right while roughly holding altitude.
- ✅ Q: <image>
A UAV pilot received this instruction: "Spin in the direction of the counterclockwise" Restate it as a single concise flight command.
  - GT: Request to spin in counterclockwise direction
  - Pred: Request to spin in counterclockwise direction
- ✅ Q: <image>
You are piloting a UAV. Instruction: "Move to the front goal on the front side" Based on the current view, in which direction should the UAV move overal
  - GT: The UAV should move forward while roughly holding altitude.
  - Pred: The UAV should move forward while roughly holding altitude.
