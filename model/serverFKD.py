import logging

from .fkd_server_manager import FedMLServerManager
from .fkd_server_aggregator import FedMLAggregator
from .default_aggregator import DefaultServerAggregator


def fedavg_cross_device(args, process_id, worker_number, comm, device, test_dataloader, model, server_aggregator=None):
    logging.info("test_data_global.iter_number = {}".format(test_dataloader.iter_number))

    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, test_dataloader, server_aggregator)

def init_server(args, device, comm, rank, size, model, test_dataloader, aggregator):
    if aggregator is None:
        aggregator = DefaultServerAggregator(model, args)
    aggregator.set_id(-1)

    td_id = id(test_dataloader)
    logging.info("test_dataloader = {}".format(td_id))
    logging.info("test_data_global.iter_number = {}".format(test_dataloader.iter_number))

    worker_num = size
    aggregator = FedMLAggregator(test_dataloader, worker_num, device, args, aggregator)

    # start the distributed training
    backend = args.backend
    server_manager = FedMLServerManager(args, aggregator, comm, rank, size, backend)
    if not args.using_mlops:
        server_manager.start_train()
    server_manager.run()

class ServerFKD():
    def __init__(self, args, device, test_dataloader, model, server_aggregator=None):
        if args.federated_optimizer == "FedAvg":
            logging.info("test_data_global.iter_number = {}".format(test_dataloader.iter_number))

            fedavg_cross_device(
                args, 0, args.worker_num, None, device, test_dataloader, model, server_aggregator=server_aggregator
            )
        else:
            raise Exception("Exception")
    
    def run(self):
        pass
