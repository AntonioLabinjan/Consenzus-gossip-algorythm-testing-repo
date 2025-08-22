"""
Distributed P2P people-detection using OpenCV cameras + gossip + emergent consensus
- Uses real camera indexes via cv2.VideoCapture
- Each node (camera) detects PEOPLE PRESENCE only (no identification)
- Nodes gossip their local observation (present/count + validity) to peers (in-memory P2P)
- Rebroadcast-once prevents infinite loops
- Consensus per tick = majority among VALID contributors (>= 50%) says PRESENT
- No synthetic faulty behaviour: a node is "faulty" only if its camera is broken; broken nodes
  simply don't contribute for that tick (valid=False) and don't bias the vote.

Run examples:
    python distributed_cv_people_detection_gossip.py --cams 0 --steps 300 --show 1
    python distributed_cv_people_detection_gossip.py --cams 0 1 --steps 500 --rounds 2 --show 0

Args:
    --cams <ints...>      Camera indexes (space-separated). If omitted, defaults to [0].
    --steps <int>         Number of ticks to run (default 200).
    --rounds <int>        Gossip rounds per tick (default 2).
    --min_quorum <int>    Min distinct VALID contributors before deciding (default 1).
    --show <0/1>          Show per-camera windows with detections (default 0 = off).

Notes:
- HOG is a simple CPU baseline. Swap detect_people() with your model if you want.
- If a camera fails to open or read, that node marks the reading as invalid for that tick.
"""
