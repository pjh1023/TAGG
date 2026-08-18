import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from src.metrics.abstract_metrics import CrossEntropyMetric

"""
PDM LOSS
"""
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance


def degree_filtration_barcode_0d(adj):
    """ Compute the 0-dimensional persistence barcode of a graph under the
        degree-based sublevel set filtration (Eq. 2-3 of the paper).
        adj : (n, n) tensor -- (weighted) adjacency matrix. Entries of adj may carry
              gradients; the returned barcode is built by indexing the normalized
              degree tensor, so it stays differentiable.
        Returns a (num_barcodes, 2) tensor of (birth, death) pairs where essential
        components have death = inf, or None if the graph has no simplices. """
    deg = torch.sum(adj, dim=0)
    max_deg = torch.max(deg)
    deg_norm = deg / max_deg if max_deg != 0 else deg

    edge_index = adj.nonzero().t().contiguous()
    simplices = gd.SimplexTree()
    for idx in torch.where(deg != 0)[0]:
        simplices.insert([idx.item()], filtration=deg_norm[idx])
    for i, j in zip(edge_index[0], edge_index[1]):
        simplices.insert([i.item(), j.item()], filtration=max(deg_norm[i], deg_norm[j]))

    simplices.persistence(min_persistence=0, persistence_dim_max=1)
    persistence_pairs = simplices.persistence_pairs()

    bcs = []
    for b, d in persistence_pairs:
        if len(b) == 1 and len(d) == 0:
            # essential connected component (never dies)
            bc = torch.cat([deg_norm[b], torch.tensor([torch.inf], device=adj.device)])
        elif len(b) == 1 and len(d) == 1:
            bc = torch.cat([deg_norm[b], deg_norm[d]])
        elif len(b) == 1 and len(d) == 2:
            # component killed by an edge: death = filtration value of the edge
            bc = torch.cat([deg_norm[b], deg_norm[[d[torch.argmax(deg_norm[d])]]]])
        else:
            # 1-dimensional feature (loop): PDM loss only uses 0-dim barcodes
            continue
        bcs.append(bc)

    if len(bcs) == 0:
        return None
    return torch.stack(bcs)


def pdm_loss(adj, score_adj):
    """ Persistence Diagram Matching loss (Eq. 11 of the paper).
        1-Wasserstein distance between the 0-dim persistence diagrams of the
        reference graphs and the predicted (probabilistic) graphs.
        adj       : (bs, n, n) tensor -- reference adjacency matrices.
        score_adj : (bs, n, n) tensor -- predicted edge probabilities. """
    loss = 0
    for a, score in zip(adj, score_adj):
        adj_b0 = degree_filtration_barcode_0d(a)

        # binarize the predicted edge probabilities (1_{x > 0.5}) while keeping gradients
        score = torch.where(score > 0.5, score, torch.zeros_like(score))
        if torch.max(score) != 0:
            score = score / torch.max(score)
        score_b0 = degree_filtration_barcode_0d(score)

        if adj_b0 is None or score_b0 is None:
            continue
        if adj_b0.shape[0] == 1:
            continue
        loss += wasserstein_distance(adj_b0, score_b0, enable_autodiff=True, keep_essential_parts=False)

    return loss

""" PDM LOSS END """


class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy + Persistence Diagram Matching loss (Sec. 4.3):
        L_final = L_CE^V + alpha_1 * L_CE^E + alpha_2 * L_PDM """
    def __init__(self, model_cfg):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = model_cfg.lambda_train      # [alpha_1, y-weight (unused, 0)]
        self.use_pdm_loss = model_cfg.use_pdm_loss      # alpha_2

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """

        # PDM loss on the estimated probability vector \hat{p}_{G_0} (edge channel 0 = "no edge")
        if self.use_pdm_loss > 0:
            pred_probs_E = F.softmax(masked_pred_E, dim=-1)
            true_adjs = 1 - true_E[:, :, :, 0]
            pred_adjs = 1 - pred_probs_E[:, :, :, 0]

            diag_mask = torch.eye(true_adjs.size(-1), device=true_adjs.device, dtype=torch.bool)
            true_adjs = true_adjs.masked_fill(diag_mask, 0)
            pred_adjs = pred_adjs.masked_fill(diag_mask, 0)

            loss_PDM = pdm_loss(true_adjs, pred_adjs)

        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)

        if self.use_pdm_loss > 0:
            return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y + self.use_pdm_loss * loss_PDM
        else:
            return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log
