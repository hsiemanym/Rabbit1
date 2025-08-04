import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
                print(f"[DEBUG] fwd_hook triggered for layer {layer_idx}, activation shape: {output.shape}")
                self.activations[layer_idx] = output.detach()

            def bwd_hook(module, grad_input, grad_output, layer_idx=idx):
                print(f"[DEBUG] bwd_hook called for layer {layer_idx}, grad mean: {grad_output[0].mean().item():.10f}")
                self.gradients[layer_idx] = grad_output[0].detach()

            self.handles.append(layer.register_forward_hook(fwd_hook))
            self.handles.append(layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()


class GradCAM(GradCAMBase):
    def __init__(self, model, target_layers):
        super().__init__(model, target_layers)

    def generate(self, input_tensor, target_score):
        self.model.zero_grad()
        if isinstance(target_score, torch.Tensor):
            print("[DEBUG] Backward 시작 - target_score.requires_grad:", target_score.requires_grad)
            target_score.backward(retain_graph=True)
            print("[DEBUG] Backward 완료")
        else:
            raise ValueError("target_score must be a scalar tensor with requires_grad=True")

        heatmaps = []
        for idx in range(len(self.target_layers)):
            act = self.activations.get(idx)
            grad = self.gradients.get(idx)

            if act is None or grad is None:
                print(f"[ERROR] Activation or gradient missing for layer {idx}")
                raise RuntimeError(f"Missing activation/gradient for layer index {idx}")
            else:
                print(f"[DEBUG] GradCAM Layer {idx} → Activation shape: {act.shape}, Grad shape: {grad.shape}")
                print(f"[DEBUG] Grad mean @ layer {idx}: {grad.mean().item():.10f}")

            weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
            cam = (weights * act).sum(dim=1).squeeze(0)    # [H, W]
            cam = F.relu(cam)

            cam -= cam.min()
            cam /= cam.max() + 1e-8
            cam = cam.cpu().numpy()
            cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
            heatmaps.append(cam)

        return heatmaps

    def generate_from_features(self, feature_map, grad_tensor, input_tensor):
        print("[DEBUG] generate_from_features() called")
        print(f"[DEBUG] feature_map shape: {feature_map.shape}, grad_tensor shape: {grad_tensor.shape}")

        weights = grad_tensor.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * feature_map).sum(dim=1).squeeze(0)  # [H, W]
        cam = F.relu(cam)

        cam -= cam.min()
        cam /= cam.max() + 1e-8
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        return cam
