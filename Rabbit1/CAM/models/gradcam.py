import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# models/gradcam.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAMBase:
    def __init__(self, model, target_layers):
        """
        model: SimSiamBackbone
        target_layers: list of strings (e.g., ['encoder.5', 'encoder.6', 'encoder.7'])
        """
        self.model = model
        self.target_layers = target_layers
        self.handles = []
        self.activations = {}
        self.gradients = {}
        self._register_hooks()

    def _get_layer(self, name):
        module = self.model
        for part in name.split('.'):
            module = getattr(module, part)
        return module

    def _register_hooks(self):
        for idx, layer_name in enumerate(self.target_layers):
            layer = self._get_layer(layer_name)

            def fwd_hook(module, input, output, layer_idx=idx):
                self.activations[layer_idx] = output.detach()

            def bwd_hook(module, grad_input, grad_output, layer_idx=idx):
                self.gradients[layer_idx] = grad_output[0].detach()

            self.handles.append(layer.register_forward_hook(fwd_hook))
            self.handles.append(layer.register_backward_hook(bwd_hook))  # warning: may deprecate in future

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()


class GradCAM(GradCAMBase):
    def __init__(self, model, target_layers):
        super().__init__(model, target_layers)

    def generate(self, input_tensor, target_score):
        """
        input_tensor: [1, 3, H, W]
        target_score: scalar tensor with requires_grad=True
        """
        self.model.zero_grad()
        if isinstance(target_score, torch.Tensor):
            target_score.backward(retain_graph=True)
        else:
            raise ValueError("target_score must be a scalar tensor with requires_grad=True")

        heatmaps = []
        for idx in range(len(self.target_layers)):
            act = self.activations.get(idx)
            grad = self.gradients.get(idx)

            if act is None or grad is None:
                raise RuntimeError(f"Missing activation/gradient for layer index {idx}")

            weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
            cam = (weights * act).sum(dim=1).squeeze(0)    # [H, W]
            cam = F.relu(cam)

            cam -= cam.min()
            cam /= cam.max() + 1e-8
            cam = cam.cpu().numpy()
            cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
            heatmaps.append(cam)

        return heatmaps


class GradCAMpp(GradCAMBase):
    def __init__(self, model, target_layers):
        super().__init__(model, target_layers)

    def generate(self, input_tensor, target_score):
        self.model.zero_grad()
        if isinstance(target_score, torch.Tensor):
            target_score.backward(retain_graph=True)
        else:
            raise ValueError("target_score must be a scalar tensor with requires_grad=True")

        heatmaps = []
        for idx in range(len(self.target_layers)):
            act = self.activations.get(idx)
            grad = self.gradients.get(idx)

            if act is None or grad is None:
                raise RuntimeError(f"Missing activation/gradient for layer index {idx}")

            grad2 = grad ** 2
            grad3 = grad ** 3
            sum_act_grad2 = (act * grad2).sum(dim=(2, 3), keepdim=True)

            eps = 1e-8
            alpha = grad2 / (2 * grad2 + sum_act_grad2 * grad3 + eps)
            alpha = torch.nan_to_num(alpha)

            weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1).squeeze(0)
            cam = F.relu(cam)

            cam -= cam.min()
            cam /= cam.max() + 1e-8
            cam = cam.cpu().numpy()
            cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
            heatmaps.append(cam)

        return heatmaps
