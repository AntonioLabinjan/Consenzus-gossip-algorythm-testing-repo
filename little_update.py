import random
import time
from queue import Queue

# Simple Node class for gossip + consensus testing (with noisy inputs)
class Node:
    def __init__(self, node_id, peers):
        self.node_id = node_id
        self.peers = peers  # list of other Node references
        self.inbox = Queue()
        self.known_numbers = {}  # number -> [votes]

    def receive(self, message):
        self.inbox.put(message)

    def gossip(self, message):
        for peer in self.peers:
            if peer.node_id != self.node_id:
                peer.receive(message)

    def process_messages(self):
        while not self.inbox.empty():
            number, vote = self.inbox.get()
            if number not in self.known_numbers:
                self.known_numbers[number] = []
            self.known_numbers[number].append(vote)

    def vote(self, true_number):
        # Each node sees a noisy version of the true number
        noisy_number = true_number + random.randint(-3, 3)
        # Classification: 0–33 = LOW, 34–66 = MEDIUM, 67–100 = HIGH
        if noisy_number <= 33:
            vote = "LOW"
        elif noisy_number <= 66:
            vote = "MEDIUM"
        else:
            vote = "HIGH"

        self.known_numbers.setdefault(true_number, []).append(vote)
        self.gossip((true_number, vote))

    def check_consensus(self, number, threshold=3):
        if number not in self.known_numbers:
            return None
        votes = self.known_numbers[number]
        if len(votes) >= threshold:
            # Majority vote among categories
            counts = {cat: votes.count(cat) for cat in set(votes)}
            majority = max(counts, key=counts.get)
            return majority
        return None


# Demo simulation
def simulate():
    # Create nodes
    nodes = [Node(i, []) for i in range(20)]  # broj nodeova
    for node in nodes:
        node.peers = nodes

    # Jedan hardkodirani broj za klasifikaciju
    number = 42
    print(f"\nTrue number to classify: {number}")

    # Each node votes & gossips
    for node in nodes:
        node.vote(number)

    # Simulate gossip propagation
    for _ in range(3):
        for node in nodes:
            node.process_messages()
        time.sleep(0.1)

    # Check consensus at each node
    global_results = []
    for node in nodes:
        result = node.check_consensus(number)
        print(f"Node {node.node_id} consensus on {number}: {result}")
        if result:
            global_results.append(result)

    # Global cluster consensus
    if global_results:
        counts = {cat: global_results.count(cat) for cat in set(global_results)}
        global_majority = max(counts, key=counts.get)
        print("\n=== GLOBAL CONSENSUS ===")
        print(f"Cluster decision on {number}: {global_majority} (votes: {counts})")
    else:
        print("\nNo consensus reached in cluster.")


if __name__ == "__main__":
    simulate()
