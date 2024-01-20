import xmlrpc.client
import numpy as np
import torch
import io
import base64
import time


class Client:
    """
    Client for TorchRing.
    """
    def __init__(self):
        # main_node is just so that you don't have to repeatedly enter the same IP
        self.main_node = None

    def main(self):
        """
        Main loop for the client requests.
        """
        print("What would you like to do?")
        print("1. Train a model")
        print("2. Get training status")
        print("3. Get model")
        print("4. Add nodes")
        print("5. Remove nodes")
        choice = input("Enter your choice: ")
        if choice == "1":
            self.train_model()
        elif choice == "2":
            self.get_training_status()
        elif choice == "3":
            self.get_model()
        elif choice == "4":
            self.add_node()
        elif choice == "5":
            self.remove_node()

    def train_model(self):
        """
        Train using nodes in the ring.
        """
        node_ips = input("Enter the path to the node IPs file: ")
        with open(node_ips, "r") as f:
            node_ips = f.readlines()
        node_ips = [node.strip() for node in node_ips]
        if len(node_ips) < 2:
            print("Please provide at least 2 nodes.")
            return
        model_path = input("Enter the path to the PyTorch model file: ")
        data_x_path = input("Enter the path to the x data npy file: ")
        data_y_path = input("Enter the path to the y data npy file: ")
        num_epochs = int(input("Enter the number of epochs: "))
        n_size = int(input("Enter the size of the successor lists: "))
        with open(model_path, "r") as f:
            model = f.read()
        data_x = np.load(data_x_path)
        data_y = np.load(data_y_path)

        self.main_node = node_ips[0]
        initial_node = xmlrpc.client.ServerProxy(f"http://{node_ips[0]}")
        initial_node.clear_state()
        initial_node.specify_model(model)
        initial_node.specify_data([data_x.tolist(), data_y.tolist()])
        initial_node.specify_num_epochs(num_epochs)
        initial_node.specify_succ_size(n_size)
        for i, ip in enumerate(node_ips[1:]):
            try:
                print(f"Adding node {ip}, {i + 1} / {len(node_ips) - 1}")
                node = xmlrpc.client.ServerProxy(f"http://{ip}")
                node.clear_state()
                initial_node.add_node(ip)
            except:
                print(f"Could not connect to node {ip}.")
                continue
        initial_node.begin_training()
        print("Began training.")


    def get_training_status(self):
        """
        Get the training status.
        """
        if self.main_node is None:
            ip = input("Enter the ip:port to any node: ")
            self.main_node = ip
        else:
            ip = self.main_node
        node = xmlrpc.client.ServerProxy(f"http://{ip}")
        _, num_nodes, starting_nodes, removed, curr_epoch, num_epochs, total_time, finish_time, epoch_times, time_portions = node.get_status()
        if curr_epoch == num_epochs:
            _, num_nodes, starting_nodes, removed, curr_epoch, num_epochs, total_time, finish_time, epoch_times, time_portions = node.get_status()
            print("Training complete.")
            print(f"Total training time: {round(finish_time / 60, 4)} minutes")
            print(f"Epoch times: {epoch_times}")
            print(f"Time portions: {time_portions}")
            return
        print(f"Number of nodes: {num_nodes} / {starting_nodes}")
        print(f"Removed nodes: {removed}")
        print(f"Current epoch: {curr_epoch} / {num_epochs}")
        print(f"Total training time: {round(total_time / 60, 2)} minutes")
        if curr_epoch != 0:
            print(f"Estimated time remaining: {round((total_time / curr_epoch) * (num_epochs - curr_epoch) / 60, 2)} minutes")

    def get_model(self):
        """
        Get the model once training is complete.
        """
        if self.main_node is None:
            ip = input("Enter the ip:port to any node: ")
            self.main_node = ip
        else:
            ip = self.main_node
        out = input("Enter the path to the .pt output file: ")
        node = xmlrpc.client.ServerProxy(f"http://{ip}")
        model_string = node.get_status()[0]
        model = torch.load(io.BytesIO(base64.b64decode(model_string)))
        torch.save(model, out)
        print("Model saved.")

    def add_node(self):
        """
        Add a node to the ring.
        """
        if self.main_node is None:
            ip = input("Enter the ip:port to any node: ")
            self.main_node = ip
        else:
            ip = self.main_node
        node_ips = input("Enter the path to the node IPs file: ")
        node = xmlrpc.client.ServerProxy(f"http://{ip}")
        with open(node_ips, "r") as f:
            node_ips = f.readlines()
        node_ips = [node.strip() for node in node_ips]
        for ip in node_ips:
            try:
                temp = xmlrpc.client.ServerProxy(f"http://{ip}")
                temp.change_index(0)
            except:
                print(f"Could not connect to node {ip}.")
                continue
            node.add_node(ip)
        print("Nodes added. They will be included in the training process when the next epoch begins.")

    def remove_node(self):
        """
        Remove a node from the ring.
        """
        node_ips = input("Enter the path to the node IPs file: ")
        with open(node_ips, "r") as f:
            node_ips = f.readlines()
        node_ips = [node.strip() for node in node_ips]
        for ip in node_ips:
            time.sleep(0.3)
            node = xmlrpc.client.ServerProxy(f"http://{ip}")
            node.remove_self()
        print("Nodes removed.")


if __name__ == "__main__":
    client = Client()
    while True:
        client.main()
        print("\n")