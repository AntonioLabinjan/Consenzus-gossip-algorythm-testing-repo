import random
import time
from queue import Queue
from collections import defaultdict

# -----------------------------
# P2P Keyword Search via Gossip + Local Aggregation Consensus
# -----------------------------
# Each node holds a local "document" (bag of words).
# A query (query_id, term) is gossiped. When a node first sees the query,
# it computes its local match statistics and gossips a RESULT message.
# RESULT messages also get gossiped. Since everyone rebroadcasts once,
# eventually all nodes aggregate the same global totals (in a fully-connected net),
# reaching de facto consensus without a central coordinator.
# -----------------------------

class Message:
    def __init__(self, kind, payload):
        self.kind = kind  # 'QUERY' or 'RESULT'
        self.payload = payload

class Node:
    def __init__(self, node_id, peers=None, faulty=False):
        self.node_id = node_id
        self.peers = peers or []
        self.inbox = Queue()
        self.seen_queries = set()       # query_ids we've already processed (to avoid rebroadcast loops)
        self.rebroadcasted_results = set()  # (query_id, from_node) pairs we've already rebroadcast
        self.docs = []                  # local words
        self.faulty = faulty            # intentionally faulty reporter (for demo)

        # Aggregates per query_id
        # stores dict with keys: 'nodes_total', 'nodes_hit', 'hit_count', 'contributors'
        self.aggregates = defaultdict(lambda: {
            'nodes_total': set(),
            'nodes_hit': set(),
            'hit_count': 0,
            'contributors': set()  # node_ids that already contributed a RESULT
        })

    # --- P2P plumbing ---
    def connect(self, peers):
        self.peers = peers

    def receive(self, msg: Message):
        self.inbox.put(msg)

    def gossip(self, msg: Message):
        # broadcast to all peers (simple full-mesh demo)
        for p in self.peers:
            if p.node_id != self.node_id:
                p.receive(msg)

    # --- Local helpers ---
    def local_keyword_stats(self, term: str):
        # count occurrences in our doc (bag of words)
        count = sum(1 for w in self.docs if w == term)
        if self.faulty:
            # Misreport: add a fixed offset to count (demo only)
            count = max(0, count + 3)
        has = 1 if count > 0 else 0
        return has, count

    # --- Protocol logic ---
    def handle_query(self, query_id: str, term: str):
        if query_id in self.seen_queries:
            return
        self.seen_queries.add(query_id)

        # compute local stats and emit RESULT
        has, count = self.local_keyword_stats(term)

        # record that we've contributed (so we don't double-report)
        self.aggregates[query_id]['contributors'].add(self.node_id)
        self.aggregates[query_id]['nodes_total'].add(self.node_id)
        if has:
            self.aggregates[query_id]['nodes_hit'].add(self.node_id)
            self.aggregates[query_id]['hit_count'] += count

        result_msg = Message('RESULT', {
            'query_id': query_id,
            'from': self.node_id,
            'has': has,
            'count': count
        })
        self.gossip(result_msg)

        # Also forward the QUERY once (classic rumor spread)
        self.gossip(Message('QUERY', {'query_id': query_id, 'term': term}))

    def handle_result(self, query_id: str, from_node: int, has: int, count: int):
        # Aggregate if we haven't already accounted for this contributor
        agg = self.aggregates[query_id]
        if from_node in agg['contributors']:
            # we've already tallied this contributor
            return

        agg['contributors'].add(from_node)
        agg['nodes_total'].add(from_node)
        if has:
            agg['nodes_hit'].add(from_node)
            agg['hit_count'] += count

        # Rebroadcast each unique contributor result exactly once (simple flood)
        key = (query_id, from_node)
        if key not in self.rebroadcasted_results:
            self.rebroadcasted_results.add(key)
            self.gossip(Message('RESULT', {
                'query_id': query_id,
                'from': from_node,
                'has': has,
                'count': count
            }))

    def process_messages(self):
        while not self.inbox.empty():
            m = self.inbox.get()
            if m.kind == 'QUERY':
                self.handle_query(m.payload['query_id'], m.payload['term'])
            elif m.kind == 'RESULT':
                self.handle_result(m.payload['query_id'], m.payload['from'], m.payload['has'], m.payload['count'])

    # Local view of global result (consensus emerges when all see same totals)
    def local_decision(self, query_id: str, min_contributors: int = 1, hit_ratio_threshold: float = 0.3):
        agg = self.aggregates.get(query_id)
        if not agg:
            return None
        total = len(agg['contributors'])
        if total < min_contributors:
            return None
        hit_nodes = len(agg['nodes_hit'])
        hit_ratio = hit_nodes / max(1, total)
        total_hits = agg['hit_count']
        decision = 'PRESENT' if hit_ratio >= hit_ratio_threshold else 'RARE/ABSENT'
        return {
            'contributors': total,
            'hit_nodes': hit_nodes,
            'hit_ratio': hit_ratio,
            'total_hits': total_hits,
            'decision': decision
        }


# -----------------------------
# Demo simulation
# -----------------------------

def build_vocab():
    # tiny vocab for demo
    return [
        'ai','ml','cv','nlp','robot','vision','cloud','edge','data','graph',
        'index','search','clip','faiss','kmeans','hash','raft','gossip'
    ]


def make_random_doc(vocab, length=80, keyword=None, keyword_bias=0.15):
    # Each word is sampled uniformly; with some probability we inject the keyword
    doc = []
    for _ in range(length):
        if keyword and random.random() < keyword_bias:
            doc.append(keyword)
        else:
            doc.append(random.choice(vocab))
    return doc


def simulate(num_nodes=25, faulty_ratio=0.12, query_term='gossip', steps=8,
             keyword_bias=0.20, doc_len=80, hit_ratio_threshold=0.3):
    random.seed(42)
    vocab = build_vocab()

    # Create nodes
    nodes = []
    for i in range(num_nodes):
        faulty = (random.random() < faulty_ratio)
        n = Node(i, faulty=faulty)
        n.docs = make_random_doc(vocab, length=doc_len, keyword=query_term, keyword_bias=keyword_bias)
        nodes.append(n)

    # Fully connect (full mesh) for simplicity
    for n in nodes:
        n.connect(nodes)

    # Initiator fires the query
    query_id = 'q1'
    initiator = nodes[0]
    print(f"Query '{query_term}' initiated by Node {initiator.node_id} (q={query_id})\n")
    initiator.receive(Message('QUERY', {'query_id': query_id, 'term': query_term}))

    # Run a few gossip steps
    for step in range(steps):
        for n in nodes:
            n.process_messages()
        time.sleep(0.05)

    # Print local decisions
    print("--- Local decisions per node ---")
    decisions = []
    for n in nodes:
        d = n.local_decision(query_id, min_contributors=max(3, num_nodes//4), hit_ratio_threshold=hit_ratio_threshold)
        if d is None:
            print(f"Node {n.node_id}: decision=UNKNOWN (insufficient data), faulty={n.faulty}")
        else:
            print(f"Node {n.node_id}: decision={d['decision']}, contributors={d['contributors']}, "
                  f"hit_nodes={d['hit_nodes']}, hit_ratio={d['hit_ratio']:.2f}, total_hits={d['total_hits']}, "
                  f"faulty={'YES' if n.faulty else 'NO'}")
            decisions.append(d['decision'])

    # Global tally (emergent consensus)
    if decisions:
        tally = defaultdict(int)
        for dec in decisions:
            tally[dec] += 1
        winner = max(tally, key=tally.get)
        print("\n=== EMERGENT CLUSTER CONSENSUS ===")
        print(f"Decision: {winner}  (votes={dict(tally)})")
    else:
        print("\nNo node had enough info to form a decision.")


if __name__ == '__main__':
    simulate()
