from xmlrpc.server import SimpleXMLRPCServer
import logging
import sys
import messager
import numpy as np
import torch
import io
import base64
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import threading
import time as t
import datetime

logging.basicConfig(level=logging.INFO)

TIMEOUT_SECS = 10

class Data(Dataset):
    """
    A simple dataset class for PyTorch dataloaders.
    """
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]
    
    def __len__(self):
        return len(self.x)

class Server:
    """
    The server class for TorchRing, run on each node.
    """
    def __init__(self, ip):
        self.ip = ip
        self.port = int(ip.split(":")[1])
        self.server = SimpleXMLRPCServer(("0.0.0.0", self.port), logRequests=False)
        self.register_functions()
        
        self.start_time = datetime.datetime.now()
        self.finish_time = datetime.datetime.now()
        self.index = 0
        self.num_nodes = 1
        self.starting_num_nodes = 1
        self.successors = []
        self.weights = None
        self.model = None
        self.raw_data = None
        self.new_model_dict = None
        self.num_new_models = 0
        self.last_time = 0
        self.last_total_time = 1
        self.last_preceding_time = 0
        self.curr_time = 0
        self.curr_total_time = 1
        self.curr_preceding_time = 0
        self.num_epochs = None
        self.curr_epoch = -1
        self.removed = []

        self.epoch_times = []
        self.time_portions = []

    def start(self):
        """
        Starts the listening server.
        """
        logging.info(f"Starting on port {self.port}...")
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logging.info("Server stopped")

    def register_functions(self):
        """
        Registers all XML-RPC functions.
        """
        self.server.register_function(self.change_index, 'change_index')
        self.server.register_function(self.change_num_nodes, 'change_num_nodes')
        self.server.register_function(self.add_node, 'add_node')
        self.server.register_function(self.init_self, 'init_self')
        self.server.register_function(self.change_weights, 'change_weights')
        self.server.register_function(self.clear_state, 'clear_state')
        self.server.register_function(self.specify_model, 'specify_model')
        self.server.register_function(self.specify_data, 'specify_data')
        self.server.register_function(self.specify_num_epochs, 'specify_num_epochs')
        self.server.register_function(self.begin_training, 'begin_training')
        self.server.register_function(self.store_model, 'store_model')
        self.server.register_function(self.remove_self, 'remove_self')
        self.server.register_function(self.remove_predecessor, 'remove_predecessor')
        self.server.register_function(self.remove_node, 'remove_node')
        self.server.register_function(self.add_to_training, 'add_to_training')
        self.server.register_function(self.get_status, 'get_status')
        self.server.register_function(self.specify_succ_size, 'specify_succ_size')

    def change_index(self, index):
        """
        Changes the index of the node.
        Args:
            index (int): The new index of the node.
        """
        self.index = index
        logging.info(f"Index changed to {self.index}.")
        return 1
    
    def change_num_nodes(self, num_nodes):
        """
        Changes the number of nodes in the ring.
        Args:
            num_nodes (int): The new number of nodes in the ring.
        """
        self.num_nodes = num_nodes
        logging.info(f"Number of nodes changed to {self.num_nodes}.")
        return 1
    
    def change_weights(self, weights):
        """
        Changes the weights of the model.
        Args:
            weights (dict): The new weights of the model.
        """
        self.model.load_state_dict(weights)
        logging.info(f"Weights changed.")
        return 1
    
    def specify_model(self, model_string):
        """
        Specifies the model.
        Args:
            model_string (str): The stringified PyTorch model class.
        """
        with open("model.py", "w") as f:
            f.write(model_string)
        import model
        self.model = model.Model()
        logging.info(f"Model changed.")
        return 1
    
    def specify_data(self, data):
        """
        Specifies the data.
        Args:
            data (tuple): The data.
        """
        self.raw_data = np.array(data[0]).astype(np.float32), np.array(data[1]).astype(np.float32)
        logging.info(f"Data changed.")
        return 1
    
    def specify_num_epochs(self, num_epochs):
        """
        Specifies the number of epochs.
        Args:
            num_epochs (int): The number of epochs.
        """
        self.num_epochs = num_epochs
        logging.info(f"Number of epochs changed.")
        return 1
    
    def specify_succ_size(self, succ_size):
        """
        Specifies the size of the successor list.
        Args:
            succ_size (int): The number of successors.
        """
        self.succ_size = succ_size
        logging.info(f"Successor size changed.")
        return 1
    
    def init_self(self, index, successors, succ_size, model_string, serialized_model, data, num_epochs, curr_epoch):
        """
        A function for nodes to initialize themselves
        Args:
            index (int): The index of the node.
            successors (list): The successor list for the node.
            succ_size (int): The size of the successor list.
            model_string (str): The stringified PyTorch model class.
            serialized_model (str): The serialized PyTorch model.
            data (tuple): The data.
            num_epochs (int): The number of epochs.
            curr_epoch (int): The current epoch.
        """
        self.index = index
        self.num_nodes = index + 1
        self.successors = successors
        self.succ_size = succ_size
        with open("model.py", "w") as f:
            f.write(model_string)
        import model
        self.model = model.Model()
        self.model.load_state_dict(self.deserialize_model(serialized_model).state_dict())
        self.raw_data = np.array(data[0]).astype(np.float32), np.array(data[1]).astype(np.float32)
        self.num_epochs = num_epochs
        self.curr_epoch = curr_epoch
        logging.info(f"Initialized self with index {self.index} and successors {self.successors}.")
        return 1

    def clear_state(self):
        """
        Clears the state of the node to prepare for a new job.
        """
        self.start_time = datetime.datetime.now()
        self.finish_time = datetime.datetime.now()
        self.succ_size = 2
        self.index = 0
        self.model_send_index = 0
        self.num_nodes = 1
        self.starting_num_nodes = 1
        self.successors = []
        self.weights = None
        self.model = None
        self.raw_data = None
        self.new_model_dict = None
        self.num_new_models = 0
        self.last_time = 0
        self.last_total_time = 1
        self.last_preceding_time = 0
        self.curr_time = 0
        self.curr_total_time = 1
        self.curr_preceding_time = 0
        self.num_epochs = None
        self.curr_epoch = -1
        self.removed = []

        self.epoch_times = []
        self.time_portions = []
        logging.info(f"State cleared.")
        return 1
    
    def serialize_model(self):
        """
        Serializes the model so it can be sent over XML-RPC.
        """
        buffer = io.BytesIO()
        torch.save(self.model, buffer)
        serialized_model = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return serialized_model
    
    def deserialize_model(self, serialized_model):
        """
        Deserializes the model so it can be used by PyTorch.
        Args:
            serialized_model (str): The serialized PyTorch model.
        """
        buffer = io.BytesIO(base64.b64decode(serialized_model))
        model = torch.load(buffer)
        return model
    
    def train(self, index=None):
        """
        Trains the model for one epoch.
        Args:
            index (int): The index to train on.
        """
        logging.info(f"Beginning new epoch.")
        self.new_model_dict = None
        self.num_new_models = 0
        self.model_send_index = self.index
        train_index = index if index is not None else self.index
        data_x = torch.from_numpy(self.raw_data[0])
        data_y = torch.from_numpy(self.raw_data[1])
        # find the split
        if self.curr_epoch == 0:
            self.last_time = 1 / self.num_nodes
            self.last_preceding_time = self.index * self.last_time
        if train_index == 0:
            split_start = 0
        else:
            split_start = int(len(data_x) * self.last_preceding_time / self.last_total_time)
        split_end = split_start + int(len(data_x) * self.last_time / self.last_total_time)
        if split_end - split_start == 0:
            split_start = 0
            split_end = 1
        logging.info(f"Data subset = {(split_end - split_start) / len(data_x)}")
        data_subset = Data(data_x[split_start:split_end], data_y[split_start:split_end])
        dl = DataLoader(data_subset, batch_size=32, shuffle=True)
        loss_fn = self.model.loss()
        optimizer = self.model.optimizer()
        start = datetime.datetime.now()
        for batch in dl:
            x, y = batch
            self.train_batch(x, y, loss_fn, optimizer)
        time = (datetime.datetime.now() - start).total_seconds() / len(data_subset)
        self.add_new_model(self.model)
        self.curr_time = 1 / time
        self.curr_total_time += self.curr_time
        messager.send_model(self.successors, train_index, self.serialize_model(), self.curr_epoch, self.index, time)
        self.check_completion(False)
        return 1
        
    def train_batch(self, x, y, loss_fn, optimizer):
        """
        Trains one batch.
        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The labels.
            loss_fn (torch.nn.modules.loss): The loss function.
            optimizer (torch.optim): The optimizer.
        """
        self.model.train()
        optimizer.zero_grad()
        loss = loss_fn(self.model(x), y)
        loss.backward()
        optimizer.step()
    
    def store_model(self, index, model, epoch, start_index, time):
        """
        Stores the model and sends it to the next node.
        Args:
            index (int): The index of the model.
            model (str): The serialized model.
            epoch (int): The epoch it was sent from.
            start_index (int): The index it was sent from.
            time (float): The training rate.
        """
        if self.curr_epoch != epoch or self.model_send_index == index:
            return 1
        self.add_new_model(self.deserialize_model(model))
        if index < self.index:
            self.curr_preceding_time += 1 / time
        self.curr_total_time += 1 / time
        if (start_index - 1) % self.num_nodes != self.index:
            messager.send_model(self.successors, index, model, epoch, start_index, time)
        self.check_completion(True)
        return 1
    
    def add_new_model(self, model):
        """
        Aggregates the new model with the current average.
        Args:
            model (torch.nn.Module): The new model.
        """
        if self.new_model_dict is None:
            self.new_model_dict = model.state_dict()
            self.num_new_models = 1
        else:
            self.new_model_dict = self.average_weights(model.state_dict(), self.num_new_models)
            self.num_new_models += 1
    
    def check_completion(self, new_thread):
        """
        Checks if the epoch is complete using the number of new models.
        Args:
            new_thread (bool): Whether it needs to start a new thread for the next epoch.
        """
        logging.info(f"Recieved models {self.num_new_models} / {self.num_nodes}")
        if self.num_new_models == self.num_nodes:
            self.model.load_state_dict(self.new_model_dict)
            self.epoch_times.append((datetime.datetime.now() - self.start_time).total_seconds())
            if self.curr_epoch == 0:
                self.time_portions.append(1 / self.num_nodes)
            else:
                self.time_portions.append(self.last_time / self.last_total_time)
            self.curr_epoch += 1
            if self.curr_epoch < self.num_epochs:
                self.last_time = self.curr_time
                self.last_total_time = self.curr_total_time
                self.last_preceding_time = self.curr_preceding_time
                self.curr_total_time = 0
                self.curr_preceding_time = 0
                if new_thread:
                    threading.Thread(target=self.train).start()
                else:
                    self.train()
            else:
                self.finish_time = datetime.datetime.now()
                logging.info(f"Training complete.")
                logging.info(f"Training time: {datetime.datetime.now() - self.start_time}")

    def average_weights(self, newest_model, num_avged):
        """
        Averages the weights of the new model with the current average.
        Args:
            newest_model (OrderedDict): The new model's weights.
            num_avged (int): The number of models that have been averaged.
        Returns:
            OrderedDict: The averaged weights.
        """
        averaged_weights = OrderedDict()
        old_ratio = (num_avged - 1) / num_avged
        new_ratio = 1 / num_avged
        for k in self.new_model_dict.keys():
            averaged_weights[k] = old_ratio * self.new_model_dict[k] + new_ratio * newest_model[k]
        return averaged_weights
    
    def begin_training(self, start_index=None):
        """
        Begins the training process.
        Args:
            start_index (int): The index to start training from.
        """
        logging.info(f"Beginning training.")
        messager.set_timeout(TIMEOUT_SECS)
        self.curr_epoch = 0
        self.starting_num_nodes = self.num_nodes
        self.start_time = datetime.datetime.now()
        if start_index is None:
            start_index = self.index
        if (start_index - 1) % self.num_nodes != self.index:
            messager.train_ping(self.successors[0], start_index)
        threading.Thread(target=self.train).start()
        return 1
    
    def add_to_training(self, num_new_models, time):
        """
        Adds a node during the training process.
        Args:
            num_new_models (int): The number of models the initialzing node has.
            time (float): The training rate of the initializing node.
        """
        messager.set_timeout(TIMEOUT_SECS)
        self.start_time = datetime.datetime.now()
        self.num_new_models = num_new_models
        self.new_model_dict = self.model.state_dict()
        messager.send_model(self.successors, self.index, self.serialize_model(), self.curr_epoch, self.index, time)
        return 1

    def add_node(self, ip):
        """
        Adds a node to the network.
        Args:
            ip (str): The IP address of the node to add.
        """
        started = False if self.curr_epoch <= 0 else True
        self.starting_num_nodes += 1
        if ip in self.removed:
            self.removed.remove(ip)
        if self.index == self.num_nodes - 1: # If this is the last node
            new_successors = self.successors.copy()
            if len(new_successors) < self.succ_size: # needs to add itself if the list is not full
                new_successors.append(self.ip)
            with open("model.py", "r") as f:
                model_string = f.read()
            messager.init_node(ip, self.index + 1, new_successors, self.succ_size, model_string, self.serialize_model(), (self.raw_data[0].tolist(), self.raw_data[1].tolist()), self.num_epochs, self.curr_epoch)
            if started:
                messager.add_to_training(ip, self.num_new_models + 1, 1 / self.last_time)
            self.num_nodes += 1
            self.successors.insert(0, ip)
            if len(self.successors) > self.succ_size:
                self.successors.pop()
            logging.info(f"Added node {ip} to the ring.")
            logging.info(f"Successors: {self.successors}")
        else: # This is not the last node, pass the message along
            if self.num_nodes - self.index <= self.succ_size:
                self.successors.insert(self.num_nodes - self.index - 1, ip)
                if len(self.successors) > self.succ_size:
                    self.successors.pop()
            self.num_nodes += 1
            messager.add_node(self.successors, ip, started)
            logging.info(f"Passed add_node message for {ip}")
            logging.info(f"Successors: {self.successors}")
        return 1
    
    def shutdown(self):
        """
        Shuts down the node.
        """
        t.sleep(1)
        self.server.shutdown()
    
    def remove_self(self):
        """
        Removes the node from the network.
        """
        messager.remove_node(self.successors, self.ip, self.index, self.successors, 0)
        threading.Thread(target=self.shutdown).start()
        return 1

    def remove_predecessor(self, dead_ip):
        """
        Removes a predecessor from the network.
        Args:
            dead_ip (str): The IP address of the predecessor to remove.
        """
        successors = self.successors.copy()[:-1]
        successors.insert(0, self.ip)
        dead_index = self.index - 1
        if dead_index < 0:
            dead_index = self.num_nodes - 1
        self.remove_node(dead_ip, dead_index, successors, 0)
        return 1

    def remove_node(self, dead_ip, index, successors, starting_ip):
        """
        Removes a node from the network.
        Args:
            dead_ip (str): The IP address of the node to remove.
            index (int): The index of the node to remove.
            successors (list): The reconstructed successor list.
            starting_ip (str): The IP address of the node that started the removal process.
        """
        if dead_ip in self.removed:
            return 1
        self.removed.append(dead_ip)
        if self.ip == starting_ip:
            return 1
        if starting_ip == 0:
            starting_ip = self.ip
        self.num_nodes -= 1
        if self.index > index:
            self.index -= 1
        if dead_ip in self.successors:
            self.successors = self.successors[:self.successors.index(dead_ip)]
            self.successors.extend(successors)
            if len(self.successors) > self.succ_size:
                self.successors = self.successors[:self.succ_size]
            if self.ip in self.successors:
                self.successors = self.successors[:self.successors.index(self.ip)]
        for dead_ip in self.removed:
            if dead_ip in self.successors:
                self.successors.remove(dead_ip)
        logging.info(f"Removed node {dead_ip} from the ring.")
        logging.info(f"Successors: {self.successors}")
        messager.remove_node(self.successors, dead_ip, index, successors, starting_ip)
        self.check_completion(True)
        return 1
    
    def get_status(self):
        """
        Gets the status of the training process.
        """
        return self.serialize_model(), self.num_nodes, self.starting_num_nodes, self.removed, self.curr_epoch, self.num_epochs, (datetime.datetime.now() - self.start_time).total_seconds(), (self.finish_time - self.start_time).total_seconds(), self.epoch_times, self.time_portions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Correct usage: python3 server.py [ip:port]")
        sys.exit(1)
    server = Server(sys.argv[1])
    server.start()
