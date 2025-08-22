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

import argparse
from collections import defaultdict
from queue import Queue

import cv2

# -----------------------------
# Message & Node
# -----------------------------

class Msg:
    def __init__(self, kind, payload):
        self.kind = kind  # 'OBS'
        self.payload = payload  # {tick, from, present, count, valid}

class Node:
    def __init__(self, node_id, cap: cv2.VideoCapture, peers=None, show=False):
        self.node_id = node_id
        self.cap = cap
        self.ok = cap is not None and cap.isOpened()
        self.peers = peers or []
        self.inbox = Queue()
        self.show = show

        # OpenCV HOG person detector (simple baseline)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Rebroadcast suppression
        self.seen = set()  # {(tick, from_id)}
        # Aggregates per tick: {tick: {from_id: (present, count, valid)}}
        self.agg = defaultdict(dict)

    def connect(self, peers):
        self.peers = peers

    def receive(self, msg: Msg):
        self.inbox.put(msg)

    def gossip(self, msg: Msg):
        for p in self.peers:
            if p.node_id != self.node_id:
                p.receive(msg)

    def detect_people(self, frame):
        # Resize for speed (tune as needed)
        h, w = frame.shape[:2]
        scale = 640.0 / max(1, w)
        if scale < 1.0:
            frame_rs = cv2.resize(frame, (int(w*scale), int(h*scale)))
        else:
            frame_rs = frame
        # HOG detect
        rects, _ = self.hog.detectMultiScale(frame_rs, winStride=(8,8), padding=(8,8), scale=1.05)
        count = len(rects)
        present = count > 0
        return present, count, frame_rs, rects

    def measure_and_broadcast(self, tick):
        # Grab a frame
        valid = False
        present, count = False, 0
        frame_rs, rects = None, []
        if self.ok:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                valid = True
                present, count, frame_rs, rects = self.detect_people(frame)
            else:
                # mark camera broken until reopened
                self.ok = False

        key = (tick, self.node_id)
        if key not in self.seen:
            self.seen.add(key)
            self.agg[tick][self.node_id] = (present, count, valid)
            self.gossip(Msg('OBS', {
                'tick': tick,
                'from': self.node_id,
                'present': present,
                'count': count,
                'valid': valid,
            }))

        # Optional visualize
        if self.show and frame_rs is not None:
            disp = frame_rs.copy()
            for (x,y,w,h) in rects:
                cv2.rectangle(disp, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(disp, f"Node {self.node_id} | present={present} count={count} valid={valid}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow(f"Node {self.node_id}", disp)

    def process_messages(self):
        outbox = []
        while not self.inbox.empty():
            m = self.inbox.get()
            if m.kind == 'OBS':
                t = m.payload['tick']
                f = m.payload['from']
                key = (t, f)
                if key in self.seen:
                    continue
                self.seen.add(key)
                self.agg[t][f] = (m.payload['present'], m.payload['count'], m.payload['valid'])
                outbox.append(m)
        for m in outbox:
            self.gossip(m)

    def local_decision(self, tick, min_quorum=1):
        # Returns (present_majority, present_ratio, contributors, total_count_est)
        if tick not in self.agg:
            return None
        entries = [(p,c,v) for (p,c,v) in self.agg[tick].values() if v]
        contributors = len(entries)
        if contributors < min_quorum or contributors == 0:
            return None
        yes = sum(1 for p,_,_ in entries if p)
        total_count = sum(c for _,c,_ in entries)
        ratio = yes / contributors
        return (yes >= (contributors - yes), ratio, contributors, total_count)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cams', type=int, nargs='*', default=[0])
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--min_quorum', type=int, default=1)
    ap.add_argument('--show', type=int, default=0)
    args = ap.parse_args()

    # Create caps
    caps = []
    for idx in args.cams:
        cap = cv2.VideoCapture(idx)
        # set smaller res for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        caps.append(cap)

    # Build nodes
    nodes = []
    for i, cap in enumerate(caps):
        n = Node(i, cap, show=bool(args.show))
        nodes.append(n)
    # fully connect
    for n in nodes:
        n.connect(nodes)

    print(f"Running with {len(nodes)} nodes (cams={args.cams})\n")

    # Loop
    for tick in range(args.steps):
        # 1) capture+broadcast
        for n in nodes:
            n.measure_and_broadcast(tick)
        # 2) gossip rounds
        for _ in range(args.rounds):
            for n in nodes:
                n.process_messages()
        # 3) print a representative view (node 0)
        d = nodes[0].local_decision(tick, min_quorum=args.min_quorum)
        if d is not None:
            present_majority, ratio, contrib, total_count = d
            print(f"tick={tick:04d} | contributors={contrib:02d} present_ratio={ratio:.2f} total_count={total_count:02d} => majority={'PRESENT' if present_majority else 'ABSENT'}")
        # UI
        if any(n.show for n in nodes):
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    # Final consensus on last tick
    last_t = args.steps - 1
    votes = []
    for n in nodes:
        d = n.local_decision(last_t, min_quorum=args.min_quorum)
        if d is not None:
            votes.append('P' if d[0] else 'A')
    tally = {k: votes.count(k) for k in set(votes)}
    if tally:
        winner = max(tally, key=tally.get)
        print(f"\nFinal emergent consensus @tick {last_t}: {'PRESENT' if winner=='P' else 'ABSENT'} (votes={tally})")
    else:
        print("\nNo quorum on last tick.")

    # Cleanup
    for c in caps:
        try:
            c.release()
        except Exception:
            pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
