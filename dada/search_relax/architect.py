import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.args = args
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.device = model.device
        if args.dada_fix_m and args.dada_fix_m:  # exclude m and p (pos 0 and 2)
            params_to_update = [self.model.augment_parameters()[1]] + self.model.augment_parameters()[3:]
        elif args.dada_fix_p:  # exclude p (pos 0)
            params_to_update = self.model.augment_parameters()[1:]
        elif args.dada_fix_m:  # exclude m (pos 2)
            params_to_update = self.model.augment_parameters()[:2] + self.model.augment_parameters()[3:]
        else:  # update all params
            params_to_update = self.model.augment_parameters()
        self.optimizer = torch.optim.Adam(params_to_update,  #self.model.augment_parameters(), #[self.model.augment_parameters()[1]] + self.model.augment_parameters()[3:],
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        #self.optimizer = torch.optim.SGD([self.model.augment_parameters()[0]] + self.model.augment_parameters()[3:], lr=args.arch_learning_rate,
        #                           momentum=0.9, nesterov=args.nesterov)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data.detach()
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data.detach() + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(moment + dtheta, alpha=eta))  #sub(eta, moment + dtheta))
        return unrolled_model


    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):

        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            raise NotImplementedError("does not include returning accuracies")
            self._backward_step(input_valid, target_valid)

        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_model.set_augmenting(False)
        unrolled_loss, self.unrolled_acc = unrolled_model._loss(input_valid, target_valid, return_acc=True)


        # unrolled_loss.backward()
        # dalpha = [v.grad for v in unrolled_model.augment_parameters()]
        # vector = [v.grad.data for v in unrolled_model.parameters()]
        # implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
        #
        # for g, ig in zip(dalpha, implicit_grads):
        #     g.data.sub_(eta, ig.data)
        #print("unrolled_loss", unrolled_loss)
        unrolled_loss.backward()

        dalpha = []
        vector = [v.grad.data.detach() for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
        for ig in implicit_grads:
            dalpha += [-ig]
        for v, g in zip(self.model.augment_parameters(), dalpha):
            if v.grad is None:
                if not (g is None):
                    v.grad = Variable(g.data)
            else:
                if not (g is None):
                    v.grad.data.copy_(g.data)


    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        if self.args.parallel:
            model_new.model = torch.nn.DataParallel(model_new.model, device_ids=self.args.device_ids)
            #model_new._criterion = torch.nn.DataParallel(model_new._criterion, device_ids=self.args.device_ids)
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.to(self.device)

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        #if torch.any(torch.isinf(self.model.model.layer1._modules['0'].conv1.weight)):
        #    print("ERROR: architect.Model contains NaN")
        # The nominator of the following expression can get zero leading to inf values -> added + epsilon
        R = r / (_concat(vector).data.detach().norm() + 1e-10)
        for p, v in zip(self.model.parameters(), vector):
            if torch.any(torch.isinf(p.data)):
                print("ERROR: parameter is inf")
            p.data.add_(v, alpha=R)  # (R, v)
            if torch.any(torch.isinf(p.data)):
                print("ERROR: parameter is inf")
        loss = self.model._loss(input, target)
        # grads_p = torch.autograd.grad(loss, self.model.augment_parameters(), retain_graph=True, allow_unused=True)
        grads_p = self.model.relax(loss)
        m_grads_p = torch.autograd.grad(loss, [self.model.augment_parameters()[2]], retain_graph=True, allow_unused=True)[0]
        if m_grads_p is None:
            m_grads_p = torch.zeros_like(self.model.augment_parameters()[2])
        grads_p.insert(2, m_grads_p)

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(v, alpha=2 * R)  # (2 * R, v)
        loss = self.model._loss(input, target)
        grads_n = self.model.relax(loss)
        m_grads_n = torch.autograd.grad(loss, [self.model.augment_parameters()[2]], retain_graph=True, allow_unused=True)[0]
        if m_grads_n is None:
            m_grads_n = torch.zeros_like(self.model.augment_parameters()[2])
        grads_n.insert(2, m_grads_n)
        # for x in grads_n:
        #     print(x.shape)

        for p, v in zip(self.model.parameters(), vector):
            if torch.any(torch.isinf(p.data)):
                print("ERROR: parameter is inf")
            p.data.add_(v, alpha=R)  # (R, v)
            if torch.any(torch.isinf(p.data)):
                print("ERROR: parameter is inf")
        for i in range(len(grads_p)):
            if torch.any(torch.isnan([(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)][i])):
                print("NaN in Hess. vector product", i)
        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
