import gudhi as gd
import torch


def get_tree(a):
    """ Build the simplex tree of the degree-based sublevel set filtration (Eq. 2-3).
        a : (n, n) adjacency matrix. Isolated (degree-0) nodes are excluded. """
    deg = torch.sum(a, dim=0)
    deg_norm = deg / torch.max(deg)

    edge_index = a.nonzero().t().contiguous()
    simplices = gd.SimplexTree()
    for idx in torch.where(deg != 0)[0]:
        simplices.insert([idx.item()], filtration=deg_norm[idx])
    for i, j in zip(edge_index[0], edge_index[1]):
        simplices.insert([i.item(), j.item()], filtration=max(deg_norm[i], deg_norm[j]))

    return simplices, deg, deg_norm


def get_barcode(a):
    """ Compute the 0-dim persistence barcode of a graph under the degree-based
        sublevel set filtration.
        Returns a (num_barcodes, 2) tensor of (birth, death) pairs where essential
        components have death = inf, or None if the graph has no barcodes. """
    simplices, deg, deg_norm = get_tree(a)

    simplices.persistence(min_persistence=0, persistence_dim_max=1)
    persistence_pairs = simplices.persistence_pairs()

    bcs = []
    for pair in persistence_pairs:
        b, d = pair
        if len(b) == 1 and len(d) == 0:
            bc = torch.cat([deg_norm[b], torch.tensor([torch.inf]).to(a.device)])
        elif len(b) == 1 and len(d) == 1:
            bc = torch.cat([deg_norm[b], deg_norm[d]])
        elif len(b) == 1 and len(d) == 2:
            bc = torch.cat([deg_norm[b], deg_norm[[d[torch.argmax(deg_norm[d])]]]])
        else:
            # 1-dimensional feature (loop): not used
            continue
        bcs.append(bc)

    if len(bcs) == 0:
        return None
    return torch.stack(bcs)
