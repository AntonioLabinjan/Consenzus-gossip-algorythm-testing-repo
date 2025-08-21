import random
import matplotlib.pyplot as plt
from queue import Queue

class Node:
    def __init__(self, node_id, peers, faulty=False):
        self.node_id = node_id
        self.peers = peers
        self.inbox = Queue()
        self.seen_queries = set()
        self.results = {}  # query_term -> count
        self.faulty = faulty

    def receive(self, message):
        self.inbox.put(message)

    def gossip(self, message):
        for peer in self.peers:
            if peer.node_id != self.node_id:
                # prevent infinite rebroadcast loops by not spamming everyone blindly
                if message not in peer.seen_queries:
                    peer.receive(message)

    def process_messages(self):
        outbox = []
        while not self.inbox.empty():
            msg_type, payload = self.inbox.get()

            if msg_type == "QUERY":
                query_term = payload
                if query_term not in self.seen_queries:
                    self.seen_queries.add(query_term)
                    # count occurrences in local doc
                    count = self.local_doc.count(query_term)
                    if self.faulty:
                        count += 3  # intentionally wrong boost
                    self.results[query_term] = count

                    # gossip my result
                    outbox.append(("RESULT", (query_term, self.node_id, count)))

                    # forward query only once
                    outbox.append(("QUERY", query_term))

            elif msg_type == "RESULT":
                query_term, sender_id, count = payload
                if (query_term, sender_id) not in self.seen_queries:
                    self.seen_queries.add((query_term, sender_id))
                    if query_term not in self.results:
                        self.results[query_term] = 0
                    # aggregate
                    self.results[query_term] += count
                    outbox.append(("RESULT", payload))

        # gossip all new messages
        for m in outbox:
            self.gossip(m)

    def set_doc(self, words):
        self.local_doc = words


# --- Simulation ---
def simulate(num_nodes=20, faulty_ratio=0.1, query_term="clip", steps=5):
    nodes = []
    for i in range(num_nodes):
        faulty = random.random() < faulty_ratio
        nodes.append(Node(i, [], faulty=faulty))

    for node in nodes:
        node.peers = nodes

    # generate random documents (biased with keyword)
    vocab = ["ai", "ml", "data", "face", "faiss", "gossip", "clip"]
    for node in nodes:
        doc = [random.choice(vocab) for _ in range(100)]
        # inject query_term occasionally
        for _ in range(random.randint(0, 5)):
            doc.append(query_term)
        node.set_doc(doc)

    # bootstrap query
    nodes[0].receive(("QUERY", query_term))

    consensus_over_time = []

    for step in range(steps):
        for node in nodes:
            node.process_messages()

        # count how many nodes think keyword is present
        present_votes = 0
        for node in nodes:
            count = node.local_doc.count(query_term)
            if node.faulty:
                count += 3
            if count > 0:
                present_votes += 1
        consensus_over_time.append(present_votes)

    # final tally
    final_counts = [node.local_doc.count(query_term) + (3 if node.faulty else 0) for node in nodes]

    # --- Visualization ---
    plt.figure(figsize=(12,5))

    # Histogram of keyword counts per node
    plt.subplot(1,2,1)
    plt.hist(final_counts, bins=range(max(final_counts)+2), edgecolor="black")
    plt.xlabel("Keyword count in node's doc")
    plt.ylabel("# of nodes")
    plt.title("Distribution of keyword hits across nodes")

    # Line chart of consensus formation
    plt.subplot(1,2,2)
    plt.plot(range(steps), consensus_over_time, marker="o")
    plt.xlabel("Gossip step")
    plt.ylabel("Nodes voting PRESENT")
    plt.title("Consensus formation over time")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate()
