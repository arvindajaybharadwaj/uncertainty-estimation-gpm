"""GPM inference with cached walk encodings, using GPM's own modules unchanged.

A GPM forward pass on a node's walks has two parts (GPM/GPM/model):
  1. PatternEncoder.encode_node turns each walk into one vector. A walk's
     vector depends only on its own nodes and its length, so with the
     per-length layout (layout.py) it never changes between stages.
  2. Model.encode_node then prepends a CLS token (on in every dataset's
     config), runs the pattern Transformer across it and all of the node's walk
     vectors, and applies the head to the CLS output. Attention mixes every
     walk, so this part reruns at every stage.
`encode` calls part 1 on the new walks only. `classify` repeats the rest of
Model.encode_node line for line. `predict` is the original full forward pass,
used for fixed K and to check that caching changes nothing.
"""
import numpy as np

import layout
from paths import use_gpm


def load(dataset, checkpoint, device=None):
    """Data, split and model, set up in the same order as GPM/final_inference.py."""
    use_gpm()
    import torch
    from data.pyg_data_loader import load_data
    from model.model import Model
    from model.random_walk import get_patterns
    from sampling_experiment import load_baseline_params, set_sampling_seed
    from utils.utils import mask2idx, seed_everything

    seed_everything(42)
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params = load_baseline_params(dataset)
    params['device'] = device
    graph, splits = load_data(params)
    params['input_dim'] = graph.x.size(1)
    params['edge_dim'] = graph.edge_attr.size(1) if graph.edge_attr is not None else 0
    params['output_dim'] = params.get('num_tasks') or int(graph.y.max()) + 1
    model = Model(params=params).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    ids = [mask2idx(splits[0][s]).cpu() for s in ('val', 'test')]
    return CachedGPM(model, graph, params, get_patterns), ids, set_sampling_seed


class CachedGPM:
    def __init__(self, model, graph, params, get_patterns):
        import torch
        self.torch = torch
        self.model, self.graph, self.params = model, graph, params
        self.get_patterns = get_patterns
        self.device = params['device']
        self.groups = len(params['multiscale'])
        if params['use_vq']:
            raise ValueError('use_vq is on; classify() only covers the configuration used in the runs.')
        if params['task'] != 'node':
            raise ValueError('Only node classification is supported.')

    def sample(self, batch, count):
        return self.get_patterns(self.graph, dict(self.params, pre_sample_pattern_num=count), nodes=batch)

    def predict(self, batch, pool):
        """The original forward pass (final_inference.py), class probabilities [n, class]."""
        run = dict(self.params, pattern_set={'pattern': pool[0].contiguous(), 'eid': pool[1].contiguous()})
        logits = self.model(self.graph, self.torch.arange(len(batch)), run, mode='eval')[0]
        # The model drops the batch dimension for a single node; keep it so stopping masks stay 1-D.
        return logits.softmax(-1).reshape(len(batch), -1)

    def arranged(self, pool, k):
        """The first k walks of a pool in sampling order, grouped by length (layout.order)."""
        index = self.torch.from_numpy(layout.order(k, groups=self.groups)).to(pool[0].device)
        return tuple(x.index_select(0, index.to(x.device)) for x in pool)

    def encode(self, walks, eids):
        """Walk vectors [h, n, hidden] for walks placed in equal length groups, as in Model.encode_node."""
        graph = self.graph
        e_feat = graph.edge_attr[eids].to(self.device) if graph.edge_attr is not None else None
        node_pe = graph.pe if graph.get('pe') is not None else None
        return self.model.pattern_encoder.encode_node(walks, graph.x, node_pe, e_feat, self.params)

    def classify(self, vectors, n):
        """The rest of Model.encode_node: CLS token, pattern Transformer, instance embedding, head."""
        if self.params['use_cls_token']:
            cls_token = self.torch.ones(1, vectors.size(1), vectors.size(2), device=vectors.device)
            vectors = self.torch.cat([cls_token, vectors], dim=0)
        pattern_emb = self.model.transformer_encode(vectors)
        instance_emb = self.model.get_instance_emb(pattern_emb, self.params)
        return self.model.head(instance_emb).softmax(-1).reshape(n, -1)

    def adaptive(self, batch, steps, threshold, check=False):
        """The live nested rule of final_inference.py, with cached walk encodings.

        Sampling, costs and stopping are as in the original: each stage samples
        k - previous walks for the nodes still active, every stage is charged k
        pattern-passes, and a node stops once its confidence reaches the
        threshold. Only the new walks are encoded; their vectors join each
        length group's cache, and stopped nodes leave the cache.
        With check=True the walks are also kept, and every stage's cached
        probabilities are compared with a fresh pass on the same walks.
        Returns final_k, predicted, pattern_passes, sampled and the largest
        difference seen (None without check).
        """
        torch = self.torch
        n = len(batch)
        final_k = np.zeros(n, dtype=np.int64)
        predicted = np.full(n, -1, dtype=np.int64)
        passes = np.zeros(n, dtype=np.int64)
        sampled = np.zeros(n, dtype=np.int64)
        active, cache, pool, previous, largest = np.arange(n), None, None, 0, 0.
        for k in steps:
            if not len(active):
                break
            active_nodes = batch[torch.from_numpy(active)]
            extra = self.sample(active_nodes, k - previous)
            sampled[active] += k - previous
            new = self.encode(*extra)
            parts = [new[s] for s in layout.split_new(k - previous, self.groups)]
            cache = parts if cache is None else [torch.cat((old, part), 0) for old, part in zip(cache, parts)]
            probs = self.classify(torch.cat(cache, 0), len(active))
            if check:
                pool = extra if pool is None else tuple(torch.cat((old, x), 0) for old, x in zip(pool, extra))
                fresh = self.predict(active_nodes, self.arranged(pool, k))
                largest = max(largest, float((probs - fresh).abs().max()))
            probs = probs.cpu().numpy()
            passes[active] += k
            stop = (probs.max(-1) >= threshold) | (k == steps[-1])
            final_k[active[stop]] = k
            predicted[active[stop]] = probs[stop].argmax(-1)
            active = active[~stop]
            keep = torch.from_numpy(np.flatnonzero(~stop))
            cache = [x[:, keep.to(x.device)].contiguous() for x in cache]
            if pool is not None:
                pool = tuple(x[:, keep.to(x.device)].contiguous() for x in pool)
            previous = k
        return final_k, predicted, passes, sampled, (largest if check else None)
