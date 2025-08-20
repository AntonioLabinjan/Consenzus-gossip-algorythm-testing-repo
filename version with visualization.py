import random
import time
from queue import Queue
import matplotlib.pyplot as plt


# Node class for gossip + consensus testing (with noisy + faulty sensor data)
class Node:
    def __init__(self, node_id, peers, faulty=False):
        self.node_id = node_id
        self.peers = peers  # list of other Node references
        self.inbox = Queue()
        self.known_measurements = {}  # reading_id -> [values]
        self.faulty = faulty  # if True, node always reports wrong values
        self.last_measurement = None

    def receive(self, message):
        self.inbox.put(message)

    def gossip(self, message):
        for peer in self.peers:
            if peer.node_id != self.node_id:
                peer.receive(message)

    def process_messages(self):
        while not self.inbox.empty():
            reading_id, value = self.inbox.get()
            if reading_id not in self.known_measurements:
                self.known_measurements[reading_id] = []
            self.known_measurements[reading_id].append(value)

    def measure(self, reading_id, true_temp):
        if self.faulty:
            # Faulty sensor: adds large offset
            noisy_temp = true_temp + 20
        else:
            # Normal sensor: true temp + small random noise
            noisy_temp = true_temp + random.randint(-3, 3)

        self.last_measurement = noisy_temp

        print(
            f"Node {self.node_id} measured {noisy_temp}°C"
            + ("  <-- intentionally faulty node!" if self.faulty else "")
        )

        self.known_measurements.setdefault(reading_id, []).append(noisy_temp)
        self.gossip((reading_id, noisy_temp))

    def check_consensus(self, reading_id, threshold=5):
        if reading_id not in self.known_measurements:
            return None
        values = self.known_measurements[reading_id]
        if len(values) >= threshold:
            avg_temp = sum(values) / len(values)
            if avg_temp >= 30:
                return f"HEAT ALERT (avg={avg_temp:.1f}°C)"
            else:
                return f"OK (avg={avg_temp:.1f}°C)"
        return None


# Demo simulation
def simulate():
    # Create nodes (mark a few as faulty)
    nodes = []
    for i in range(20):  # broj nodeova
        faulty = (i % 7 == 0)  # svaki 7. node je faulty
        nodes.append(Node(i, [], faulty=faulty))

    for node in nodes:
        node.peers = nodes

    # Jedan hardkodirani ground truth (npr. realna temp = 28°C)
    true_temp = 28
    reading_id = "temp_001"
    print(f"\nTrue temperature: {true_temp}°C\n")

    # Each node measures & gossips
    for node in nodes:
        node.measure(reading_id, true_temp)

    # Simulate gossip propagation
    for _ in range(3):
        for node in nodes:
            node.process_messages()
        time.sleep(0.1)

    # Check consensus at each node
    global_results = []
    print("\n--- CONSENSUS CHECK PER NODE ---")
    for node in nodes:
        result = node.check_consensus(reading_id)
        print(f"Node {node.node_id} consensus: {result}")
        if result:
            global_results.append(result)

    # Global cluster consensus
    if global_results:
        counts = {res: global_results.count(res) for res in set(global_results)}
        global_majority = max(counts, key=counts.get)
        print("\n=== GLOBAL CONSENSUS ===")
        print(f"Cluster decision: {global_majority} (votes: {counts})")
    else:
        print("\nNo consensus reached in cluster.")

    # ==== VISUALIZATION ====
    temps = [node.last_measurement for node in nodes]
    colors = ["red" if node.faulty else "blue" for node in nodes]
    labels = ["Faulty" if node.faulty else "Normal" for node in nodes]

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(nodes)), temps, c=colors, label="Node readings", s=100)

    plt.axhline(true_temp, color="green", linestyle="--", label="True temperature")
    if global_results:
        avg_temp = sum([node.last_measurement for node in nodes]) / len(nodes)
        plt.axhline(avg_temp, color="orange", linestyle=":", label="Cluster average")

    for i, (t, lbl) in enumerate(zip(temps, labels)):
        plt.text(i, t + 0.5, lbl, fontsize=8, ha="center")

    plt.title("Distributed Temperature Measurements with Faulty Nodes")
    plt.xlabel("Node ID")
    plt.ylabel("Measured Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    simulate()
