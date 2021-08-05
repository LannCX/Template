import importlib


def find_network_using_name(network_name):
    network_filename = "arch." + network_name.lower()
    networklib = importlib.import_module(network_filename)

    network = None
    target_network_name = network_name.replace('_', '')
    for name, cls in networklib.__dict__.items():
        if name.lower() == target_network_name.lower():
            network = cls
            break

    if network is None:
        print("In %s.py, there should be a class with class name that matches %s in lowercase." %
              (network_filename, target_network_name))
        exit(0)

    return network


def customer_net_loader(cfg):
    network_class = find_network_using_name(cfg.MODEL.NAME)
    network_obj = network_class(cfg)

    return network_obj
