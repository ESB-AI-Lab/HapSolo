import numpy as np

ABSENT = 0
COMPLETE = 1
FRAGMENTED = 2


class ScoringData:
    """Integer-indexed ortholog scoring data, GPU-transferable.

    Encodes busco2contigdict/contigs2buscodict as a numpy matrix
    ortho_contig_matrix[ortho_id, contig_id] with status codes:
      0 = absent, 1 = Complete, 2 = Fragmented
    """

    def __init__(self):
        self.contig_to_id = {}
        self.id_to_contig = []
        self.ortho_to_id = {}
        self.id_to_ortho = []
        self.n_orthologs = 0
        self.n_contigs = 0
        self.ortho_contig_matrix = None

    @classmethod
    def from_busco_dicts(cls, busco2contig, contigs2busco,
                         contig_to_id=None, id_to_contig=None):
        """Build from legacy dict format.

        If contig_to_id/id_to_contig are provided, uses that encoding
        (must cover all contigs in contigs2busco). Otherwise builds its
        own from contigs2busco.keys().
        """
        sd = cls()
        if contig_to_id is not None and id_to_contig is not None:
            sd.contig_to_id = contig_to_id
            sd.id_to_contig = id_to_contig
            sd.n_contigs = len(id_to_contig)
        else:
            all_contigs = sorted(contigs2busco.keys())
            sd.id_to_contig = all_contigs
            sd.contig_to_id = {name: i for i, name in enumerate(all_contigs)}
            sd.n_contigs = len(all_contigs)

        all_orthos = sorted(busco2contig.keys())
        sd.id_to_ortho = all_orthos
        sd.ortho_to_id = {name: i for i, name in enumerate(all_orthos)}
        sd.n_orthologs = len(all_orthos)

        sd.ortho_contig_matrix = np.zeros(
            (sd.n_orthologs, sd.n_contigs), dtype=np.uint8)

        for contig_name, busco_types in contigs2busco.items():
            cid = sd.contig_to_id.get(contig_name)
            if cid is None:
                continue
            for buscoid in busco_types.get('C', []):
                oid = sd.ortho_to_id.get(buscoid)
                if oid is not None:
                    sd.ortho_contig_matrix[oid, cid] = COMPLETE
            for buscoid in busco_types.get('F', []):
                oid = sd.ortho_to_id.get(buscoid)
                if oid is not None and sd.ortho_contig_matrix[oid, cid] != COMPLETE:
                    sd.ortho_contig_matrix[oid, cid] = FRAGMENTED

        return sd

    def calculate_scores(self, contig_names):
        """Vectorized ortholog scoring. Returns dict with keys C, S, D, F, M."""
        ids = [self.contig_to_id[n] for n in contig_names if n in self.contig_to_id]
        if not ids:
            return {'C': 0, 'S': 0, 'D': 0, 'F': 0, 'M': self.n_orthologs}

        sub = self.ortho_contig_matrix[:, ids]
        complete_counts = np.sum(sub == COMPLETE, axis=1)
        has_frag = np.any(sub == FRAGMENTED, axis=1)

        single = int(np.sum(complete_counts == 1))
        dup = int(np.sum(complete_counts > 1))
        frag = int(np.sum((complete_counts == 0) & has_frag))
        missing = self.n_orthologs - single - dup - frag

        return {
            'C': single + dup,
            'S': single,
            'D': dup,
            'F': frag,
            'M': missing,
        }

    def calculate_scores_from_mask(self, good_mask):
        """Score orthologs using a boolean mask. Works with numpy or cupy arrays."""
        xp = type(self.ortho_contig_matrix)
        try:
            import cupy as cp
            if isinstance(self.ortho_contig_matrix, cp.ndarray):
                xp = cp
            else:
                xp = np
        except ImportError:
            xp = np

        ids = xp.where(good_mask[:self.n_contigs])[0]
        if len(ids) == 0:
            return {'C': 0, 'S': 0, 'D': 0, 'F': 0, 'M': self.n_orthologs}

        sub = self.ortho_contig_matrix[:, ids]
        complete_counts = xp.sum(sub == COMPLETE, axis=1)
        has_frag = xp.any(sub == FRAGMENTED, axis=1)

        single = int(xp.sum(complete_counts == 1))
        dup = int(xp.sum(complete_counts > 1))
        frag = int(xp.sum((complete_counts == 0) & has_frag))
        missing = self.n_orthologs - single - dup - frag

        return {
            'C': single + dup,
            'S': single,
            'D': dup,
            'F': frag,
            'M': missing,
        }

    def calculate_scores_batch(self, good_masks):
        """Score N threshold combos simultaneously on GPU.

        good_masks: cupy array of shape (N, n_contigs), bool
        Returns: (singles, dups, frags, missings) — cupy arrays of shape (N,)
        """
        import cupy as cp
        N = good_masks.shape[0]
        matrix = self.ortho_contig_matrix[None, :, :]  # (1, n_ortho, n_contigs)
        masks_3d = good_masks[:, None, :].astype(cp.uint8)  # (N, 1, n_contigs)
        masked = matrix * masks_3d  # (N, n_ortho, n_contigs) — zeros out non-good
        complete_counts = cp.sum(masked == COMPLETE, axis=2)  # (N, n_ortho)
        has_frag = cp.any(masked == FRAGMENTED, axis=2)  # (N, n_ortho)
        singles = cp.sum(complete_counts == 1, axis=1)  # (N,)
        dups = cp.sum(complete_counts > 1, axis=1)  # (N,)
        frags = cp.sum((complete_counts == 0) & has_frag, axis=1)  # (N,)
        missings = self.n_orthologs - singles - dups - frags  # (N,)
        return singles, dups, frags, missings

    def prepare_batch_fast(self):
        """Pre-compute float32 complete/frag matrices for matmul scoring.

        Replaces the 3D tensor approach in calculate_scores_batch with two
        matrix multiplies, eliminating the (N, n_ortho, n_contigs) temporary.
        """
        import cupy as cp
        self._complete_f32 = (self.ortho_contig_matrix == COMPLETE).astype(cp.float32)
        self._frag_f32 = (self.ortho_contig_matrix == FRAGMENTED).astype(cp.float32)

    def calculate_scores_batch_fast(self, good_masks):
        """Score N combos via matmul — no 3D tensor, no sync points.

        good_masks: cupy (N, n_contigs) bool
        Returns: (singles, dups, frags, missings) cupy (N,) int
        """
        import cupy as cp
        masks_f = good_masks[:, :self.n_contigs].astype(cp.float32)
        cc = (self._complete_f32 @ masks_f.T).T
        fc = (self._frag_f32 @ masks_f.T).T
        singles = cp.sum(cc == 1.0, axis=1)
        dups = cp.sum(cc > 1.0, axis=1)
        frags = cp.sum((cc == 0.0) & (fc > 0.0), axis=1)
        missings = self.n_orthologs - singles - dups - frags
        return singles, dups, frags, missings

    def to_gpu(self):
        """Convert numpy arrays to cupy for GPU-accelerated scoring."""
        try:
            import cupy as cp
            self.ortho_contig_matrix = cp.asarray(self.ortho_contig_matrix)
        except ImportError:
            raise RuntimeError('cupy is required for GPU acceleration')
