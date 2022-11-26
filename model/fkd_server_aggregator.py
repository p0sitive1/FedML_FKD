import time

import MNN
import numpy as np
import wandb

import fedml
from fedml import mlops
import torch
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm

from .utils import Models

F = MNN.expr
nn = MNN.nn

from .utils import read_mnn_as_tensor_dict
import logging
from FedGen.FedGen import LinearNet, Linear


class FedMLAggregator(object):
    def __init__(
        self, test_dataloader, worker_num, device, args, aggregator,
    ):
        self.aggregator = aggregator

        self.args = args
        self.test_global = test_dataloader

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.aggregator.get_model_params()

    # TODO: refactor MNN-related file processing
    def get_global_model_params_file(self):
        return self.aggregator.get_model_params_file()

    def set_global_model_params(self, model_parameters):
        self.aggregator.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        logging.info(f"{index} worker's model: {self.model_dict[index]}")
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        logging.info("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            logging.info("self.model_dict[idx] = {}".format(self.model_dict[idx]))
            mnn_file_path = self.model_dict[idx]
            tensor_params_dict = read_mnn_as_tensor_dict(mnn_file_path)  # Read model file here
            logging.info(f"worker {idx} model: ")
            for key, value in tensor_params_dict.items():
                logging.info('{}: {}'.format(key, value.shape))
            # logging.info(f"{idx} worker's model: {tensor_params_dict}")
            model_list.append((self.sample_num_dict[idx], tensor_params_dict))
            training_num += self.sample_num_dict[idx]
        logging.info("training_num = {}".format(training_num))
        logging.info(f"Number of models {len(model_list)}")
        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        ######################FKD here##################
                    
        # generate temp dataset
        temp_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        temp_train = DataLoader(temp_trainset)
        client_models = []


        for i in range(0, len(model_list)):
            local_sample_number, local_model_params = model_list[i]
            local_model_dict = dict()
            actual_keys = ["fc2.bias", "fc2.weight", "fc1.bias", "fc1.weight", "conv2.weight", "conv2.bias", "conv1.weight", "conv1.bias"]
            for key, value in local_model_params.items():
                if key == 0 or key == 2:
                    value = value.reshape(-1)
                local_model_dict[actual_keys[key]] = value
            temp_model = Linear()
            temp_model.load_state_dict(local_model_dict)
            temp_model.eval()
            temp_client = Models(temp_model)
            client_models.append(temp_client)
            print("models constructed")

        logging.info(f"Global knowledge distillation starts")
        for t in range(3):
            logging.info(f"Distillation round {t}")
            for _, (images_pub, labels_pub) in enumerate(temp_train):
                images_pub, labels_pub = images_pub.to(self.device), labels_pub.to(self.device)
                total_logit = None
                for i in range(0, len(client_models)):
                    client_models[i].model.eval()
                    output = client_models[i].model(images_pub)
                    if total_logit is None:
                        total_logit = output.detach().clone()
                    else:
                        total_logit += output.detach().clone()

                for i in range(0, len(client_models)):
                    client_models[i].model.train()
                    client_models[i].optimizer.zero_grad()
                    output_pub = client_models[i].model(images_pub)

                    avg_logit_except_self = (total_logit.clone() - output_pub) / (len(model_list))
                    # only one client 
                    avg_prob_except_self = torch.nn.functional.softmax(avg_logit_except_self / 1)
                    loss_kd = torch.nn.KLDivLoss(reduction="batchmean")(
                        f.log_softmax(output_pub / 1),
                        avg_prob_except_self.detach())
                    loss = 1 * loss_kd
                    loss.backward()
                    client_models[i].optimizer.step()

        logging.info(f"Global knowledge distillation completes")

        tmp_model = client_models[0]
        state_dict = tmp_model.model.state_dict()

        send_dict = dict()
        actual_keys = ["fc2.bias", "fc2.weight", "fc1.bias", "fc1.weight", "conv2.weight", "conv2.bias", "conv1.weight", "conv1.bias"]
        for key, value in state_dict.items():
            if actual_keys.index(key) == 0 or actual_keys.index(key) == 2:
                tmp = value.shape[0]
                value = value.reshape([1, tmp])
            send_dict[actual_keys.index(key)] = value.cpu()
        send_dict = dict(sorted(send_dict.items()))

        logging.info(f"send model: ")
        for key, value in send_dict.items():
            logging.info('{}: {}'.format(key, value.shape))
        
        # torch.save(client_models[0], self.args.global_model_file_path)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return send_dict
    
    def data_silo_selection(self, round_idx, data_silo_num_in_total, client_num_in_total):
        """

        Args:
            round_idx: round index, starting from 0
            data_silo_num_in_total: this is equal to the users in a synthetic data,
                                    e.g., in synthetic_1_1, this value is 30
            client_num_in_total: the number of edge devices that can train

        Returns:
            data_silo_index_list: e.g., when data_silo_num_in_total = 30, client_num_in_total = 3,
                                        this value is the form of [0, 11, 20]

        """
        logging.info(
            "data_silo_num_in_total = %d, client_num_in_total = %d" % (data_silo_num_in_total, client_num_in_total)
        )
        assert data_silo_num_in_total >= client_num_in_total
        if client_num_in_total == data_silo_num_in_total:
            return [i for i in range(data_silo_num_in_total)]
        else:
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            data_silo_index_list = np.random.choice(range(data_silo_num_in_total), client_num_in_total, replace=False)
        return data_silo_index_list

    def client_selection(self, round_idx, client_id_list_in_total, client_num_per_round):
        """
        Args:
            round_idx: round index, starting from 0
            client_id_list_in_total: this is the real edge IDs.
                                    In MLOps, its element is real edge ID, e.g., [64, 65, 66, 67];
                                    in simulated mode, its element is client index starting from 0, e.g., [0, 1, 2, 3]
            client_num_per_round:

        Returns:
            client_id_list_in_this_round: sampled real edge ID list, e.g., [64, 66]
        """
        if client_num_per_round == len(client_id_list_in_total) or len(client_id_list_in_total) == 1:  # for debugging
            return client_id_list_in_total
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_id_list_in_this_round = np.random.choice(client_id_list_in_total, client_num_per_round, replace=False)
        return client_id_list_in_this_round

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def test_on_server_for_all_clients(self, mnn_file_path, round_idx):
        # load global model from MNN
        var_map = F.load_as_dict(mnn_file_path)
        input_dicts, output_dicts = F.get_inputs_and_outputs(var_map)
        input_names = [n for n in input_dicts.keys()]
        output_names = [n for n in output_dicts.keys()]
        input_vars = [input_dicts[n] for n in input_names]
        output_vars = [output_dicts[n] for n in output_names]
        module = MNN.nn.load_module(input_vars, output_vars, False)

        module.train(False)
        self.test_global.reset()

        correct = 0
        for i in range(self.test_global.iter_number):
            example = self.test_global.next()
            input_data = example[0]
            output_target = example[1]
            data = input_data[0]  # which input, model may have more than one inputs
            label = output_target[0]  # also, model may have more than one outputs

            result = module.forward(data)
            predict = F.argmax(result, 1)
            predict = np.array(predict.read())

            label_test = np.array(label.read())
            correct += np.sum(label_test == predict)

            target = F.one_hot(F.cast(label, F.int), 10, 1, 0)
            loss = nn.loss.cross_entropy(result, target)

        test_accuracy = correct * 100.0 / self.test_global.size
        test_loss = loss.read()
        fedml.logging.info("test acc = {}".format(test_accuracy))
        fedml.logging.info("test loss = {}".format(test_loss))

        mlops.log(
            {
                "round_idx": round_idx,
                "accuracy": round(np.round(test_accuracy, 4), 4),
                "loss": round(np.round(test_loss, 4)),
            }
        )

        if self.args.enable_wandb:
            wandb.log(
                {"round idx": round_idx, "test acc": test_accuracy, "test loss": test_loss,}
            )
