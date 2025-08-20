import random
import threading
import time
from queue import Queue

# Simple Node class for gossip + consensus testing
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

    def vote(self, number):
        # Consensus rule: even=1, odd=0
        vote = 1 if number % 2 == 0 else 0
        self.known_numbers.setdefault(number, []).append(vote)
        self.gossip((number, vote))

    def check_consensus(self, number, threshold=3):
        if number not in self.known_numbers:
            return None
        votes = self.known_numbers[number]
        if len(votes) >= threshold:
            majority = 1 if votes.count(1) > votes.count(0) else 0
            return "even" if majority == 1 else "odd"
        return None


# Demo simulation
def simulate():
    # Create nodes
    nodes = [Node(i, []) for i in range(5)]
    for node in nodes:
        node.peers = nodes

    # Random number to classify
    number = random.randint(1, 100)
    print(f"Number to classify: {number}")

    # Each node votes & gossips
    for node in nodes:
        node.vote(number)

    # Simulate gossip propagation
    for _ in range(3):
        for node in nodes:
            node.process_messages()
        time.sleep(0.2)

    # Check consensus at each node
    for node in nodes:
        result = node.check_consensus(number)
        print(f"Node {node.node_id} consensus on {number}: {result}")


if __name__ == "__main__":
    simulate()
