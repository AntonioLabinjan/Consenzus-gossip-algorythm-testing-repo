"""
Distributed P2P people-detection using OpenCV cameras + gossip + emergent consensus
- Each node detects PEOPLE PRESENCE via motion (no HOG/identification)
- Nodes gossip local observations to peers (in-memory P2P)
- Consensus per tick = majority among VALID contributors (>= 50%) says PRESENT
- Broken cameras = invalid for that tick, don't bias vote
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

        # EBV state
        self.prev_frame_gray = None
        self.motion_accumulator = 0

        # Rebroadcast suppression
        self.seen = set()  # {(tick, from_id)}
        self.agg = defaultdict(dict)

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return False, 0, frame  # first frame, no motion
        diff = cv2.absdiff(gray, self.prev_frame_gray)
        self.prev_frame_gray = gray
        motion_mask = (diff > 30).astype('uint8') * 255
        motion_pixels = cv2.countNonZero(motion_mask)
        self.motion_accumulator += motion_pixels

        motion_threshold = 5000  # tweakable
        present = False
        count = 0
        if self.motion_accumulator >= motion_threshold:
            present = True
            count = self.motion_accumulator
            self.motion_accumulator = 0
        return present, count, frame

    def measure_and_broadcast(self, tick):
        valid = False
        present, count = False, 0
        frame_disp = None

        if self.ok:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                valid = True
                present, count, frame_disp = self.detect_motion(frame)
            else:
                self.ok = False  # camera broken

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

        if self.show and frame_disp is not None:
            disp = frame_disp.copy()
            text = f"Node {self.node_id} | present={present} count={count} valid={valid}"
            cv2.putText(disp, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow(f"Node {self.node_id}", disp)

    def connect(self, peers):
        self.peers = peers

    def receive(self, msg: Msg):
        self.inbox.put(msg)

    def gossip(self, msg: Msg):
        for p in self.peers:
            if p.node_id != self.node_id:
                p.receive(msg)

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
    ap.add_argument('--show', type=int, default=1)
    args = ap.parse_args()

    print("Setting up camera captures...")
    caps = []
    for idx in args.cams:
        print(f"  Initializing camera {idx}...")
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        if cap.isOpened():
            print(f"    Camera {idx} opened successfully.")
        else:
            print(f"    WARNING: Camera {idx} failed to open!")
        caps.append(cap)

    print("Creating nodes...")
    nodes = []
    for i, cap in enumerate(caps):
        print(f"  Node {i} created for camera {args.cams[i]}")
        n = Node(i, cap, show=bool(args.show))
        nodes.append(n)
    for n in nodes:
        n.connect(nodes)
    print("All nodes connected.\n")

    print(f"Running with {len(nodes)} nodes (cams={args.cams})\n")

    for tick in range(args.steps):
        # Capture + broadcast
        for n in nodes:
            n.measure_and_broadcast(tick)
        # Gossip rounds
        for _ in range(args.rounds):
            for n in nodes:
                n.process_messages()
        # Local decision
        d = nodes[0].local_decision(tick, min_quorum=args.min_quorum)
        if d is not None:
            present_majority, ratio, contrib, total_count = d
            print(f"tick={tick:04d} | contributors={contrib:02d} present_ratio={ratio:.2f} total_count={total_count:02d} => majority={'PRESENT' if present_majority else 'ABSENT'}")
        # UI quit
        if any(n.show for n in nodes):
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    # Final consensus
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
