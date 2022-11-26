import copy
import os

import MNN
import numpy as np
import torch

F = MNN.expr


def read_mnn_as_tensor_dict(mnn_file_path):
    var_map = F.load_as_dict(mnn_file_path)
    input_dicts, output_dicts = F.get_inputs_and_outputs(var_map)
    input_names = [n for n in input_dicts.keys()]
    output_names = [n for n in output_dicts.keys()]
    input_vars = [input_dicts[n] for n in input_names]
    output_vars = [output_dicts[n] for n in output_names]
    module = MNN.nn.load_module(input_vars, output_vars, False)

    tensor_params_tensor_dict = {}
    for idx_layer in range(len(module.parameters)):
        module.parameters[idx_layer].fix_as_const()
        mnn_layer_weights_np_arr = copy.deepcopy(module.parameters[idx_layer].read())
        tensor_params_tensor_dict[idx_layer] = torch.from_numpy(
            mnn_layer_weights_np_arr
        ).detach()

    return tensor_params_tensor_dict


def write_tensor_dict_to_mnn(mnn_file_path, tensor_params_tensor_dict):
    var_map = F.load_as_dict(mnn_file_path)
    input_dicts, output_dicts = F.get_inputs_and_outputs(var_map)
    input_names = [n for n in input_dicts.keys()]
    output_names = [n for n in output_dicts.keys()]
    input_vars = [input_dicts[n] for n in input_names]
    output_vars = [output_dicts[n] for n in output_names]
    module = MNN.nn.load_module(input_vars, output_vars, False)
    input_shape = F.shape(input_vars[0])

    mnn_params_list = []
    for idx_layer in range(len(tensor_params_tensor_dict)):
        pt_layer_weights_np_arr = tensor_params_tensor_dict[idx_layer].numpy()
        tmp = F.const(pt_layer_weights_np_arr, list(pt_layer_weights_np_arr.shape))
        tmp.fix_as_trainable()
        mnn_params_list.append(tmp)

    module.load_parameters(mnn_params_list)
    predict = module.forward(F.placeholder(input_shape.read(), F.NCHW))
    F.save([predict], mnn_file_path)


def transform_list_to_tensor(model_params_list, enable_cuda_rpc):
    if enable_cuda_rpc:
        return model_params_list
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(
            np.asarray(model_params_list[k])
        ).float()
    return model_params_list


def transform_tensor_to_list(model_params, enable_cuda_rpc):
    if enable_cuda_rpc:
        return model_params
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def post_complete_message_to_sweep_process(args):
    pipe_path = "./tmp/fedml"
    os.system("mkdir ./tmp/; touch ./tmp/fedml")
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))


# store temp model
import os, sys
import collections
from functools import reduce
from torch.autograd import Variable
import copy
import torch.nn as nn

LOSS_ACC_BATCH_SIZE = 128   # When computing loss and accuracy, use blocks of LOSS_ACC_BATCH_SIZE


class Models():
    def __init__(self, model, rand_seed=None, learning_rate=0.001, num_classes=10, model_name='LeNet5', channels=1, img_size=32, device=torch.device('cuda'), flatten_weight=False, optimizer='Adam'):
        super(Models, self).__init__()
        if rand_seed is not None:
            torch.manual_seed(rand_seed)
        self.model = None
        self.loss_fn = None
        self.weights_key_list = None
        self.weights_size_list = None
        self.weights_num_list = None
        self.optimizer = None
        self.channels = channels
        self.img_size = img_size
        self.flatten_weight = flatten_weight
        self.learning_rate = learning_rate
        self.device = device

        self.model = model

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.model.to(device)
        self.loss_fn = nn.CrossEntropyLoss().to(device)
        self._get_weight_info()

    def weight_variable(self, tensor, mean, std):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

    def bias_variable(self, shape):
        return torch.ones(shape) * 0.1

    def init_variables(self):

        self._get_weight_info()

        weight_dic = collections.OrderedDict()

        for i in range(len(self.weights_key_list)):
            if i%2 == 0:
                tensor = torch.zeros(self.weights_size_list[i])
                sub_weight = self.weight_variable(tensor, 0, 0.1)
            else:
                sub_weight = self.bias_variable(self.weights_size_list[i])
            weight_dic[self.weights_key_list[i]] = sub_weight

        self.model.load_state_dict(weight_dic)

    def _get_weight_info(self):
        self.weights_key_list = []
        self.weights_size_list = []
        self.weights_num_list = []
        state = self.model.state_dict()
        for k, v in state.items():
            shape = list(v.size())
            self.weights_key_list.append(k)
            self.weights_size_list.append(shape)
            if len(shape) > 0:
                num_w = reduce(lambda x, y: x * y, shape)
            else:
                num_w=0
            self.weights_num_list.append(num_w)
        self.grad_key_list = []  # For the different part of weight compared to gradient
        for k, _ in self.model.named_parameters():
            self.grad_key_list.append(k)
        self.diff_index_list = []
        j = 0
        for i in range(len(self.weights_key_list)):
            if self.weights_key_list[i] == self.grad_key_list[j]:
                j += 1
            else:
                self.diff_index_list.append(i)

    def get_weight_dimension(self):
        dim = sum(self.weights_num_list)
        return dim

    def get_weight(self):
        with torch.no_grad():
            state = self.model.state_dict()
            if self.flatten_weight:
                weight_flatten_tensor = torch.Tensor(sum(self.weights_num_list)).to(state[self.weights_key_list[0]].device)
                start_index = 0
                for i,[_, v] in zip(range(len(self.weights_num_list)), state.items()):
                    weight_flatten_tensor[start_index:start_index+self.weights_num_list[i]] = v.view(1, -1)
                    start_index += self.weights_num_list[i]

                return weight_flatten_tensor
            else:
                return copy.deepcopy(state)

    def assign_weight(self, w):
        if self.flatten_weight:
            self.assign_flattened_weight(w)
        else:
            self.model.load_state_dict(w)

    def assign_flattened_weight(self, w):

        weight_dic = collections.OrderedDict()
        start_index = 0

        for i in range(len(self.weights_key_list)):
            sub_weight = w[start_index:start_index+self.weights_num_list[i]]
            if len(sub_weight) > 0:
                weight_dic[self.weights_key_list[i]] = sub_weight.view(self.weights_size_list[i])
            else:
                weight_dic[self.weights_key_list[i]] = torch.tensor(0)
            start_index += self.weights_num_list[i]
        self.model.load_state_dict(weight_dic)

    def _data_reshape(self, imgs, labels=None):
        if len(imgs.size()) < 3:
            x_image = imgs.view([-1, self.channels, self.img_size, self.img_size])
            if labels is not None:
                _, y_label = torch.max(labels.data, 1)  # From one-hot to number
            else:
                y_label = None
            return x_image, y_label
        else:
            return imgs, labels

    global start_index
    global grad_flatten_tensor
    global i

    def get_flattened_gradient(self):
        global start_index
        global grad_flatten_tensor
        global i
        grad_flatten_tensor = torch.Tensor(sum(self.weights_num_list)).to(self.device)
        start_index = 0
        with torch.no_grad():
            i = 0
            for _, param in self.model.named_parameters():
                def add_diff():
                    global start_index
                    global grad_flatten_tensor
                    global i
                    if i in self.diff_index_list:
                        grad_flatten_tensor[start_index:start_index + self.weights_num_list[i]] = torch.zeros(self.weights_num_list[i])
                        start_index += self.weights_num_list[i]
                        i += 1
                        add_diff()
                    else:
                        return 1
                if i in self.diff_index_list:
                    add_diff()
                grad_flatten_tensor[start_index:start_index + self.weights_num_list[i]] = param.grad.clone().view([1, -1])
                start_index += self.weights_num_list[i]
                i += 1
        return grad_flatten_tensor

    def accuracy(self, data_test_loader, w, device):
        if w is not None:
            self.assign_weight(w)

        self.model.eval()
        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_test_loader):
                images, labels = Variable(images).to(device), Variable(labels).to(device)
                output = self.model(images)
                avg_loss += self.loss_fn(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        avg_loss /= len(data_test_loader.dataset)
        acc = float(total_correct) / len(data_test_loader.dataset)

        return avg_loss.item(), acc

    def predict(self, img, w, device):

        self.assign_weight(w)
        img, _ = self._data_reshape(img)
        with torch.no_grad():
            self.model.eval()
            _, pred = torch.max(self.model(img.to(device)).data, 1)

        return pred

    def train_one_epoch(self, data_train_loader, device):
        self.model.train()
        for i, (images, labels) in enumerate(data_train_loader):
            images, labels = Variable(images).to(device), Variable(labels).to(device)
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimizer.step()

    def get_feature_dimension(self, images, labels, device):
        self.model.eval()
        images, labels = Variable(images).to(device), Variable(labels).to(device)
        _, feature = self.model(images, out_feature=True)
        return feature.size()[1]
    