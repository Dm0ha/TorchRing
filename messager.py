import xmlrpc.client
import threading
import datetime
import socket


def set_timeout(timeout):
    """
    Set the timeout for the socket.
    Args:
        timeout (int): The timeout in seconds.
    """
    socket.setdefaulttimeout(timeout)

def task_remove_node(successors, removed_ip, index, rep_succesors, starting_ip):
    """
    Pass the message to remove a node.
    Args:
        successors (list): The list of successors.
        removed_ip (str): The IP of the node to remove.
        index (int): The index of the node to remove.
        rep_succesors (list): The reconstructed successor list.
        starting_ip (str): The IP of the node that started the removal.
    """
    for ip in successors:
        try:
            node = xmlrpc.client.ServerProxy(f'http://{ip}')
            node.remove_node(removed_ip, index, rep_succesors, starting_ip)
            break
        except Exception as _:
            pass

def task_send_model(successors, index, model, epoch, start_index, time):
    """
    Pass a model around the ring.
    Args:
        successors (list): The list of successors.
        index (int): The index of the node who trained the model.
        model (torch.nn.Module): The model to pass.
        epoch (int): The epoch number.
        start_index (int): The index the model was sent from.
        time (datetime.datetime): The training rate.
    """
    for ip in successors:
        try:
            node = xmlrpc.client.ServerProxy(f'http://{ip}')
            node.store_model(index, model, epoch, start_index, time)
            break
        except Exception as _:
            call_remove_predecessor(successors[1:], ip)

def init_node(ip, index, successors, succ_size, model_string, model, data, num_epochs, curr_epoch):
    """
    Pass the message to initialize a node.
    Args:
        ip (str): The IP of the node to initialize.
        index (int): The index of the node to initialize.
        successors (list): The list of successors.
        succ_size (int): The size of the successor list.
        model_string (str): The stringified PyTorch model.
        model (str): The serialized PyTorch model.
        data (tuple): The data.
        num_epochs (int): The number of epochs to train.
        curr_epoch (int): The current epoch.
    """
    try:
        node = xmlrpc.client.ServerProxy(f'http://{ip}')
        node.init_self(index, successors, succ_size, model_string, model, data, num_epochs, curr_epoch)
    except Exception as e:
        print(e)

def task_add_node(successors, new_ip):
    """
    Pass the message to add a node.
    Args:
        successors (list): The list of successors.
        new_ip (str): The IP of the new node.
    """
    for ip in successors:
        try:
            node = xmlrpc.client.ServerProxy(f'http://{ip}')
            node.add_node(new_ip)
            break
        except Exception as _:
            pass


def add_node(successors, new_ip, threaded=False):
    """
    Pass the message to add a node.
    Args:
        successors (list): The list of successors.
        new_ip (str): The IP of the new node.
    """
    if threaded:
        threading.Thread(target=task_add_node, args=(successors, new_ip)).start()
    else:
        task_add_node(successors, new_ip)

def train_ping(ip, start_index):
    """
    Pass the message to begin training.
    Args:
        ip (str): The next IP in the ring.
        start_index (int): The index that started the ping.
    """
    try:
        node = xmlrpc.client.ServerProxy(f'http://{ip}')
        node.begin_training(start_index)
    except Exception as _:
        pass

def add_to_training(ip, num_new_models, time):
    """
    Pass the message to add node while training.
    Args:
        ip (str): The IP of the node.
        num_new_models (int): The number of models at the initial node.
        time (datetime.datetime): The training rate of the initial node.
    """
    node = xmlrpc.client.ServerProxy(f'http://{ip}')
    node.add_to_training(num_new_models, time)

def send_model(successors, index, model, epoch, start_index, time):
    """
    Pass a model around the ring.
    Args:
        successors (list): The list of successors.
        index (int): The index of the node who trained the model.
        model (torch.nn.Module): The model to pass.
        epoch (int): The epoch number.
        start_index (int): The index the model was sent from.
        time (datetime.datetime): The training rate.
    """
    threading.Thread(target=task_send_model, args=(successors, index, model, epoch, start_index, time)).start()

def remove_node(successors, removed_ip, index, rep_succesors, starting_ip):
    """
    Pass the message to remove a node.
    Args:
        successors (list): The list of successors.
        removed_ip (str): The IP of the node to remove.
        index (int): The index of the node to remove.
        rep_succesors (list): The reconstructed successor list.
        starting_ip (str): The IP of the node that started the removal.
    """
    threading.Thread(target=task_remove_node, args=(successors, removed_ip, index, rep_succesors, starting_ip)).start()
    

def call_remove_predecessor(successors, removed_ip):
    """
    Pass the message to remove a predecessor.
    Args:
        successors (list): The list of successors.
        removed_ip (str): The IP of the node to remove.
    """
    for succ in successors:
        try:
            node = xmlrpc.client.ServerProxy(f'http://{succ}')
            node.remove_predecessor(removed_ip)
            break
        except Exception as _:
            pass