"""Hierarchical cluster structure extracted from HDBSCAN's condensed tree.

HDBSCAN's output is usually consumed as a *flat* labelling: each point
lands in cluster ``0..K-1`` or gets ``-1`` for outlier. That flat view
discards the rich hierarchy that HDBSCAN actually computes: density-
based clusters live on a condensed tree whose edges carry a ``lambda``
value (inverse of the density threshold at which a node forms). Two
sibling clusters always merge into the same parent at some finite
``lambda``, and the paper's "two-run" workflow is, in effect, the
astronomer's way of walking one level deeper in that tree.

This module turns the tree into first-class data:

* :class:`HierarchicalStructure` bundles the condensed-tree edge list
  together with a ``flat_to_node`` mapping so downstream users can jump
  from a flat cluster id to its tree node without re-implementing the
  lookup.
* :meth:`HierarchicalStructure.merge_lambda` answers "at what density
  threshold do clusters ``i`` and ``j`` join?", which is a direct
  measure of how far apart they sit in density space.
* :meth:`HierarchicalStructure.subcluster_labels` returns a refined
  labelling on the subset of points that currently sit in a given
  flat cluster, equivalent to the paper's two-run workflow but
  computed in one line without a second noise baseline.

The module is backend-aware: when the fit was done on ``cuml``, no
tree object is available and the :class:`HierarchicalStructure` gracefully
reports ``available=False``. CPU-fitted runs get the full interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import hdbscan as _hdbscan_typing
    from numpy.typing import NDArray

__all__ = [
    "HierarchicalStructure",
    "extract_hierarchy",
]


@dataclass(frozen=True)
class HierarchicalStructure:
    """Condensed-tree view of an HDBSCAN fit.

    The attributes are populated from ``hdbscan.HDBSCAN.condensed_tree_``
    when the CPU backend was used. On the cuml backend the tree is not
    exposed so the instance reports ``available=False`` and all the
    query methods raise :class:`RuntimeError`.

    Parameters
    ----------
    available
        ``True`` when the fit carries a condensed tree (CPU backend);
        ``False`` on cuml.
    edges
        Structured array of the condensed tree with dtype
        ``[('parent', int), ('child', int), ('lambda_val', float),
        ('child_size', int)]``. ``parent``/``child`` are *node ids*
        (leaves: single points, internal nodes: subclusters). Rows
        where ``child_size == 1`` carry an individual point; rows where
        ``child_size > 1`` carry a subcluster.
    flat_to_node
        Length-``n_flat_clusters`` mapping: for flat cluster
        ``k`` (non-negative label in ``PipelineResult.labels``),
        ``flat_to_node[k]`` is the tree node id that cluster corresponds
        to. The mapping is recovered from ``condensed_tree_`` by taking
        the node whose member-point set matches the flat label; if the
        lookup fails (pathological single-point "clusters") the entry
        is ``-1``.
    labels
        The flat labelling the hierarchy was computed from, retained so
        the object stands on its own.
    n_samples
        Total number of input samples (``labels.size``).
    raw_tree
        Opaque handle to the original ``hdbscan`` condensed-tree object,
        kept so callers can also use its ``plot`` / ``to_pandas`` /
        ``to_networkx`` helpers directly. ``None`` on cuml.
    """

    available: bool
    edges: NDArray[np.void]
    flat_to_node: NDArray[np.intp]
    labels: NDArray[np.intp]
    n_samples: int
    raw_tree: _hdbscan_typing.plots.CondensedTree | None = field(default=None, repr=False)

    # -- sibling / merge-lambda -------------------------------------------------

    def merge_lambda(self, cluster_i: int, cluster_j: int) -> float:
        r"""Return the :math:`\lambda` value at which two flat clusters merge.

        Walks the condensed tree upward from each cluster's node until
        the two walks meet; the ``lambda_val`` on the edge leaving the
        common ancestor is the merge threshold (smaller ``lambda`` =
        lower density = earlier merge). A large merge ``lambda`` means
        the two clusters live in the same density regime; a small
        ``lambda`` means one survives to a much sparser density level
        than the other.

        Raises
        ------
        RuntimeError
            If :attr:`available` is ``False``.
        ValueError
            If either cluster id is out of range.
        """
        self._require_available()
        self._check_cluster_id(cluster_i)
        self._check_cluster_id(cluster_j)
        if cluster_i == cluster_j:
            return 0.0
        node_i = int(self.flat_to_node[cluster_i])
        node_j = int(self.flat_to_node[cluster_j])
        if node_i < 0 or node_j < 0:
            return float("nan")
        parents_i, lambdas_i = self._ancestor_chain(node_i)
        parents_j, lambdas_j = self._ancestor_chain(node_j)
        set_j = dict(zip(parents_j, lambdas_j, strict=True))
        for p, lam in zip(parents_i, lambdas_i, strict=True):
            if p in set_j:
                return float(max(lam, set_j[p]))
        return float("nan")

    def sibling(self, cluster_id: int) -> int | None:
        """Return the flat cluster id that merges with ``cluster_id`` first.

        If the sibling is not itself a flat cluster (it is a
        condensed-tree subcluster that HDBSCAN's flat selection did not
        pick), returns ``None``.
        """
        self._require_available()
        self._check_cluster_id(cluster_id)
        node = int(self.flat_to_node[cluster_id])
        if node < 0:
            return None
        lambdas = np.full(self.flat_to_node.size, np.inf, dtype=np.float64)
        for k in range(self.flat_to_node.size):
            if k == cluster_id:
                continue
            lambdas[k] = self.merge_lambda(cluster_id, k)
        if not np.any(np.isfinite(lambdas)):
            return None
        return int(np.argmin(lambdas))

    # -- subcluster ----------------------------------------------------------

    def subcluster_on(
        self,
        embedding: NDArray[np.floating[Any]],
        cluster_id: int,
        *,
        min_cluster_size: int,
        min_samples: int | None = None,
        metric: str = "euclidean",
    ) -> NDArray[np.intp]:
        """Refit HDBSCAN on the rows of ``embedding`` inside ``cluster_id``.

        The caller is expected to pass the same 2-D embedding the outer
        pipeline clustered on (``PipelineResult.embedding``). The returned
        label vector is aligned to the full sample: rows outside the
        parent cluster receive ``-1``, rows inside receive the sub-fit's
        labels (with ``-1`` still denoting outliers inside the parent).
        """
        self._require_available()
        self._check_cluster_id(cluster_id)
        from starfold.clustering import run_hdbscan  # noqa: PLC0415

        emb = np.asarray(embedding, dtype=np.float64)
        if emb.ndim != 2 or emb.shape[0] != self.n_samples:
            msg = (
                "embedding must be 2-D with the same n_samples the "
                "hierarchy was built from "
                f"(got shape {emb.shape}, expected ({self.n_samples}, d))."
            )
            raise ValueError(msg)
        mask = self.labels == cluster_id
        if not np.any(mask):
            return np.full(self.n_samples, -1, dtype=np.intp)
        inner = run_hdbscan(
            emb[mask],
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )
        out = np.full(self.n_samples, -1, dtype=np.intp)
        out[mask] = np.asarray(inner.labels, dtype=np.intp)
        return out

    # -- private helpers -----------------------------------------------------

    def _require_available(self) -> None:
        if not self.available:
            msg = (
                "HierarchicalStructure is unavailable (the fit used a "
                "backend that does not expose the condensed tree, e.g. cuml). "
                "Refit with engine='cpu' to get hierarchical queries."
            )
            raise RuntimeError(msg)

    def _check_cluster_id(self, cluster_id: int) -> None:
        if not (0 <= cluster_id < self.flat_to_node.size):
            msg = (
                f"cluster_id {cluster_id} out of range for "
                f"{self.flat_to_node.size} flat clusters."
            )
            raise ValueError(msg)

    def _ancestor_chain(self, node: int) -> tuple[list[int], list[float]]:
        """Walk ``node`` up to the root, recording (parent, lambda_val)."""
        parents: list[int] = []
        lambdas: list[float] = []
        current = int(node)
        child_col = self.edges["child"]
        parent_col = self.edges["parent"]
        lambda_col = self.edges["lambda_val"]
        # An HDBSCAN condensed tree has at most ~2*n edges, so this
        # linear scan per step is fine for realistic n (<< 10^6).
        while True:
            hit = np.flatnonzero(child_col == current)
            if hit.size == 0:
                break
            idx = int(hit[0])
            parent = int(parent_col[idx])
            parents.append(parent)
            lambdas.append(float(lambda_col[idx]))
            current = parent
        return parents, lambdas


def extract_hierarchy(
    model: _hdbscan_typing.HDBSCAN | None,
    labels: NDArray[np.intp],
) -> HierarchicalStructure:
    """Build a :class:`HierarchicalStructure` from a fitted HDBSCAN model.

    Parameters
    ----------
    model
        The fitted ``hdbscan.HDBSCAN`` model (CPU backend). Pass
        ``None`` for cuml fits; the returned structure will have
        ``available=False`` but is still safe to pass through
        ``PipelineResult``.
    labels
        The flat labelling the model produced, with ``-1`` for
        outliers.

    Returns
    -------
    HierarchicalStructure
    """
    labels_arr = np.asarray(labels, dtype=np.intp)
    n_samples = int(labels_arr.size)
    positive = labels_arr[labels_arr >= 0]
    n_flat = int(positive.max() + 1) if positive.size else 0

    if model is None or not hasattr(model, "condensed_tree_"):
        return HierarchicalStructure(
            available=False,
            edges=np.zeros(0, dtype=_EMPTY_EDGE_DTYPE),
            flat_to_node=np.full(n_flat, -1, dtype=np.intp),
            labels=labels_arr,
            n_samples=n_samples,
            raw_tree=None,
        )

    tree = model.condensed_tree_
    edges = np.asarray(tree.to_numpy())
    flat_to_node = _recover_flat_to_node(edges, labels_arr, n_flat, n_samples)
    return HierarchicalStructure(
        available=True,
        edges=edges,
        flat_to_node=flat_to_node,
        labels=labels_arr,
        n_samples=n_samples,
        raw_tree=tree,
    )


_EMPTY_EDGE_DTYPE = np.dtype(
    [("parent", "<i8"), ("child", "<i8"), ("lambda_val", "<f8"), ("child_size", "<i8")]
)


def _build_children_map(
    parent_col: NDArray[np.int64],
    child_col: NDArray[np.int64],
) -> dict[int, list[int]]:
    children_of: dict[int, list[int]] = {}
    for p, c in zip(parent_col.tolist(), child_col.tolist(), strict=True):
        children_of.setdefault(int(p), []).append(int(c))
    return children_of


def _leaves_under(
    node: int, children_of: dict[int, list[int]], n_samples: int,
) -> set[int]:
    stack = [node]
    leaves: set[int] = set()
    while stack:
        n = stack.pop()
        if n < n_samples:
            leaves.add(n)
            continue
        stack.extend(children_of.get(n, ()))
    return leaves


def _recover_flat_to_node(
    edges: NDArray[np.void],
    labels: NDArray[np.intp],
    n_flat: int,
    n_samples: int,
) -> NDArray[np.intp]:
    """Map each flat cluster to the condensed-tree node with the same leaf set.

    Recovery works by point-set matching: every condensed-tree
    node below a point-index leaf set is reconstructed by walking the
    subtree rooted at that node; the node whose subtree matches a flat
    cluster's point set is the one recorded. Ambiguities (multiple
    nodes with the same point set, which can happen if a node has only
    one child surviving EOM selection) are resolved by picking the
    deepest (highest lambda) match so ``merge_lambda`` walks the
    minimum necessary tree distance.
    """
    out = np.full(n_flat, -1, dtype=np.intp)
    if edges.size == 0 or n_flat == 0:
        return out
    child_col = edges["child"].astype(np.int64)
    parent_col = edges["parent"].astype(np.int64)
    node_ids = np.unique(np.concatenate([parent_col, child_col]))
    internal = node_ids[node_ids >= n_samples]
    children_of = _build_children_map(parent_col, child_col)

    cluster_points: list[set[int]] = [
        set(np.flatnonzero(labels == k).tolist()) for k in range(n_flat)
    ]
    parent_lambda: dict[int, float] = {}
    lambda_col = edges["lambda_val"].astype(np.float64)
    for p, lam in zip(parent_col.tolist(), lambda_col.tolist(), strict=True):
        parent_lambda.setdefault(int(p), float(lam))
    internal_sorted = sorted(internal.tolist(), key=lambda n: -parent_lambda.get(n, 0.0))
    for node in internal_sorted:
        points = _leaves_under(node, children_of, n_samples)
        for k, pts in enumerate(cluster_points):
            if out[k] == -1 and points == pts:
                out[k] = int(node)
    return out
