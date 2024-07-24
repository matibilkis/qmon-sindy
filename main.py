import pennylane as qml
from pennylane import numpy as np 
import os
import sys
import argparse
sys.path.insert(0, os.getcwd())


dev = qml.device("default.qubit", wires=1, shots=1000)
@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0],wires=0)
    qml.RZ(params[1], wires=0)
    return qml.sample(qml.PauliZ(0))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--p1", type=float, default=0)
    parser.add_argument("--p2", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    p1 = args.p1
    p2 = args.p2
    seed = args.seed
    np.random.seed(seed)
    
    params = np.array([p1,p2])
    data = circuit(params)
    
    save_dir = "/data/cvcqml/common/matias/tutorial_pic/cost_reconstruction/{}/".format(params)
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir,data)