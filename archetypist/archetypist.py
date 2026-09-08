"""ArcheTypist - sample-level celltype composition analysis tools for single-cell datasets.

Workflow
--------
1. ``make_celltype_fraction_adata`` collapses a cell-level AnnData into a
   sample x cell-type fraction AnnData.
2. ``map_query_to_reference_embedding`` places new samples onto a reference
   composition embedding and transfers reference labels.
3. ``combine_reference_query_embedding`` merges reference and mapped query for
   joint plotting.
4. ``plot_reference_mapping`` draws query samples over the reference embedding.
5. ``predict_cmv`` applies a trained CMV classifier to composition fractions.
6. ``map_sample_obs_to_cells`` writes sample-level annotations back to cells.
"""

from importlib.resources import files
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import celltypist
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def make_celltype_fraction_adata(
    adata,
    sample_col="sample_id",
    celltype_col="predicted_labels",
    exclude_meta=None,
):
    """Build a sample-by-cell-type fraction AnnData from a cell-level AnnData.

    Each observation in the returned object is one sample and each variable is
    one cell type, with ``X`` holding the fraction of that sample's cells
    assigned to that cell type (rows sum to 1). Cell-level metadata is carried
    over only where it is constant within a sample.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell-level object. ``adata.obs`` must contain ``sample_col`` and
        ``celltype_col``.
    sample_col : str, default "sample_id"
        Column in ``adata.obs`` identifying the sample each cell came from.
    celltype_col : str, default "predicted_labels"
        Column in ``adata.obs`` holding cell type annotations.
    exclude_meta : iterable of str, optional
        Additional ``adata.obs`` columns to leave out of the sample-level
        metadata. ``celltype_col`` is always excluded.

    Returns
    -------
    anndata.AnnData
        Shape ``(n_samples, n_cell_types)``. ``obs`` holds the sample-constant
        metadata, ``obs_names`` are sample IDs (as strings) and ``var_names``
        are cell type names (as strings).

    Notes
    -----
    Columns that vary within any sample become ``pd.NA`` for that sample, and
    columns that end up entirely missing are dropped.
    """
    obs = adata.obs.copy()

    exclude_meta = set(exclude_meta or [])
    exclude_meta.add(celltype_col)

    # Cell type fractions per sample
    celltype_fractions = (
        obs
        .groupby([sample_col, celltype_col])
        .size()
        .groupby(level=sample_col, group_keys=False)
        .apply(lambda x: x / x.sum())
        .unstack(fill_value=0)
    )
    celltype_fractions.index = celltype_fractions.index.astype(str)
    celltype_fractions.columns = celltype_fractions.columns.astype(str)

    # Sample-level metadata (only keep columns constant within sample)
    candidate_meta = [c for c in obs.columns if c not in exclude_meta]
    sample_meta = (
        obs[candidate_meta]
        .groupby(sample_col)
        .agg(lambda x: x.dropna().iloc[0] if x.dropna().nunique() <= 1 else pd.NA)
    )
    sample_meta = sample_meta.loc[
        celltype_fractions.index,
        sample_meta.notna().any(axis=0)
    ]
    sample_meta.index = sample_meta.index.astype(str)

    # Create AnnData
    adata_frac = sc.AnnData(
        X=celltype_fractions.to_numpy(),
        obs=sample_meta,
        var=pd.DataFrame(index=celltype_fractions.columns),
    )
    adata_frac.obs_names.name = sample_col
    adata_frac.var_names.name = "cell_type"

    return adata_frac


def map_query_to_reference_embedding(
    adata_ref,
    adata_query,
    embedding_key="X_draw_graph_fa",
    n_neighbors=5,
    metric="cosine",
    label_keys=("archetype", "composition_cluster"),
    scale=False,
    eps=1e-8,
    copy=True,
):
    """Project query samples into a reference embedding and transfer labels.

    Query samples are matched to reference samples by k-nearest neighbours over
    the variables shared by both objects (typically cell-type fractions). Query
    embedding coordinates are the inverse-distance-weighted mean of their
    neighbours' reference coordinates, and categorical reference labels are
    transferred by weighted vote.

    Parameters
    ----------
    adata_ref : anndata.AnnData
        Reference object, e.g. from :func:`make_celltype_fraction_adata`. Must
        contain ``embedding_key`` in ``.obsm``.
    adata_query : anndata.AnnData
        Query object with variables overlapping the reference.
    embedding_key : str, default "X_draw_graph_fa"
        Key in ``adata_ref.obsm`` holding the reference embedding to map into.
    n_neighbors : int, default 5
        Number of reference neighbours per query sample.
    metric : str, default "cosine"
        Distance metric passed to :class:`sklearn.neighbors.NearestNeighbors`.
    label_keys : sequence of str, default ("archetype", "composition_cluster")
        Reference ``obs`` columns to transfer. Missing keys are skipped.
    scale : bool, default False
        If True, z-score features using statistics fitted on the reference.
    eps : float, default 1e-8
        Added to distances before inverting, to avoid division by zero.
    copy : bool, default True
        If True, operate on copies rather than modifying the inputs in place.

    Returns
    -------
    anndata.AnnData
        The query object with:

        - ``obsm[embedding_key]`` — mapped coordinates.
        - ``obs["mapped_ref_sample"]`` — nearest reference sample name.
        - ``obs["mapping_distance"]`` / ``obs["mapping_similarity"]`` —
          nearest-neighbour distance and ``1 - distance``.
        - ``obs["transferred_<key>"]`` and ``obs["transferred_<key>_confidence"]``
          for each transferred label, plus matching ``uns`` colour palettes
          where the reference defines them.
        - ``uns["reference_mapping"]`` — parameters used for the mapping.

    Raises
    ------
    ValueError
        If the reference and query share no variables.
    KeyError
        If ``embedding_key`` is absent from ``adata_ref.obsm``.

    Notes
    -----
    ``mapping_similarity`` is only meaningful for bounded metrics such as
    ``"cosine"``.
    """
    ref = adata_ref.copy() if copy else adata_ref
    query = adata_query.copy() if copy else adata_query

    shared_vars = ref.var_names.intersection(query.var_names)
    if len(shared_vars) == 0:
        raise ValueError("No shared variables between reference and query.")

    if embedding_key not in ref.obsm:
        raise KeyError(f"{embedding_key!r} not found in adata_ref.obsm.")

    ref_X = ref[:, shared_vars].X
    query_X = query[:, shared_vars].X
    ref_X = ref_X.toarray() if hasattr(ref_X, "toarray") else np.asarray(ref_X)
    query_X = query_X.toarray() if hasattr(query_X, "toarray") else np.asarray(query_X)

    if scale:
        scaler = StandardScaler()
        ref_X = scaler.fit_transform(ref_X)
        query_X = scaler.transform(query_X)

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
    )
    nn.fit(ref_X)
    distances, indices = nn.kneighbors(query_X)

    weights = 1 / (distances + eps)
    weights = weights / weights.sum(axis=1, keepdims=True)

    ref_embedding = np.asarray(ref.obsm[embedding_key])
    query.obsm[embedding_key] = np.sum(
        ref_embedding[indices] * weights[:, :, None],
        axis=1,
    )

    query.obs["mapped_ref_sample"] = ref.obs_names[indices[:, 0]].astype(str)
    query.obs["mapping_distance"] = distances[:, 0]
    query.obs["mapping_similarity"] = 1 - distances[:, 0]

    for label_key in label_keys:
        if label_key not in ref.obs:
            continue

        neighbor_labels = (
            ref.obs[label_key]
            .iloc[indices.ravel()]
            .values
            .reshape(indices.shape)
        )

        transferred = []
        confidence = []
        for labs, w in zip(neighbor_labels, weights):
            scores = pd.Series(w, index=labs).groupby(level=0).sum()
            transferred.append(scores.idxmax())
            confidence.append(scores.max())

        if pd.api.types.is_categorical_dtype(ref.obs[label_key]):
            categories = ref.obs[label_key].cat.categories
        else:
            categories = pd.Index(pd.unique(ref.obs[label_key].dropna()))

        query.obs[f"transferred_{label_key}"] = pd.Categorical(
            transferred,
            categories=categories,
        )
        query.obs[f"transferred_{label_key}_confidence"] = confidence

        color_key = f"{label_key}_colors"
        if color_key in ref.uns:
            query.uns[f"transferred_{label_key}_colors"] = list(ref.uns[color_key])

    query.uns["reference_mapping"] = {
        "embedding_key": embedding_key,
        "n_neighbors": n_neighbors,
        "metric": metric,
        "scale": scale,
        "shared_vars": list(shared_vars),
        "label_keys": list(label_keys),
    }

    return query


def combine_reference_query_embedding(
    adata_ref,
    adata_query_mapped,
    embedding_key="X_draw_graph_fa",
    label_keys=("archetype", "composition_cluster"),
    dataset_col="dataset",
    ref_label="reference",
    query_label="query",
):
    """Concatenate reference and mapped query objects for joint plotting.

    Adds a column marking each observation's origin, stacks the reference and
    query embeddings into a single ``obsm`` entry, and copies the reference
    colour palettes across so original and transferred labels are drawn with
    the same colours.

    Parameters
    ----------
    adata_ref : anndata.AnnData
        Reference object carrying ``embedding_key`` in ``.obsm``.
    adata_query_mapped : anndata.AnnData
        Query object returned by :func:`map_query_to_reference_embedding`.
    embedding_key : str, default "X_draw_graph_fa"
        ``obsm`` key holding the shared embedding.
    label_keys : sequence of str, default ("archetype", "composition_cluster")
        Label columns whose palettes should be preserved, both as ``<key>`` and
        as ``transferred_<key>``.
    dataset_col : str, default "dataset"
        Name of the ``obs`` column added to record the origin of each sample.
    ref_label, query_label : str, defaults "reference" / "query"
        Values written into ``dataset_col``.

    Returns
    -------
    anndata.AnnData
        Concatenated object (reference rows first) with the stacked embedding
        in ``obsm[embedding_key]`` and palettes in ``uns``.
    """
    ref_plot = adata_ref.copy()
    query_plot = adata_query_mapped.copy()

    ref_plot.obs[dataset_col] = ref_label
    query_plot.obs[dataset_col] = query_label

    combined = ref_plot.concatenate(
        query_plot,
        join="outer",
        batch_key=None,
    )

    combined.obsm[embedding_key] = np.vstack([
        ref_plot.obsm[embedding_key],
        query_plot.obsm[embedding_key],
    ])

    for key in label_keys:
        color_key = f"{key}_colors"
        if key in combined.obs:
            combined.obs[key] = combined.obs[key].astype("category")
        if color_key in adata_ref.uns:
            combined.uns[color_key] = list(adata_ref.uns[color_key])

        transferred_key = f"transferred_{key}"
        transferred_color_key = f"{transferred_key}_colors"
        if transferred_key in combined.obs:
            combined.obs[transferred_key] = combined.obs[transferred_key].astype("category")
        if color_key in adata_ref.uns:
            combined.uns[transferred_color_key] = list(adata_ref.uns[color_key])

    return combined


def plot_reference_mapping(
    adata,
    feature,
    embedding="draw_graph_fa",
    embedding_key="X_draw_graph_fa",
    reference_color="composition_cluster",
    dataset_col="dataset",
    reference_label="reference",
    query_label="query",
    background_alpha=0.1,
    background_size=12,
    query_size=40,
    edgecolor="black",
    linewidth=0.4,
    cmap="tab20",
    seed=42,
    figsize=(5, 4),
    title="Reference mapping",
    ax=None,
):
    """Plot mapped query samples over a faded reference embedding.

    The reference is drawn as a translucent background coloured by
    ``reference_color``; query samples are overlaid as larger outlined points
    coloured by a categorical query metadata ``feature``.

    Parameters
    ----------
    adata : anndata.AnnData
        Combined object from :func:`combine_reference_query_embedding`.
    feature : str
        Categorical ``obs`` column used to colour the query points. A palette in
        ``adata.uns[f"{feature}_colors"]`` is reused if present.
    embedding : str, default "draw_graph_fa"
        Basis name passed to :func:`scanpy.pl.embedding` for the background.
    embedding_key : str, default "X_draw_graph_fa"
        ``obsm`` key used to read query coordinates directly.
    reference_color : str, default "composition_cluster"
        ``obs`` column colouring the reference background.
    dataset_col : str, default "dataset"
        Column separating reference from query observations.
    reference_label, query_label : str, defaults "reference" / "query"
        Values of ``dataset_col`` identifying each group.
    background_alpha, background_size : float, defaults 0.1 / 12
        Opacity and point size of the reference background.
    query_size : float, default 40
        Point size for query samples.
    edgecolor : str, default "black"
        Outline colour for query points and legend markers.
    linewidth : float, default 0.4
        Outline width for query points and legend markers.
    cmap : str, default "tab20"
        Qualitative colormap used when ``feature`` has no stored palette.
    seed : int, default 42
        Seed for shuffling the fallback palette.
    figsize : tuple of float, default (5, 4)
        Figure size, used only when ``ax`` is None.
    title : str, default "Reference mapping"
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. A new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    ref_mask = adata.obs[dataset_col] == reference_label
    query_mask = adata.obs[dataset_col] == query_label

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Reference background
    sc.pl.embedding(
        adata[ref_mask],
        basis=embedding,
        color=reference_color,
        alpha=background_alpha,
        size=background_size,
        frameon=False,
        ax=ax,
        show=False,
        legend_loc=None,
    )

    coords = adata[query_mask].obsm[embedding_key]
    vals = adata.obs.loc[query_mask, feature].astype("category")

    # Use existing palette if available
    color_key = f"{feature}_colors"
    if color_key in adata.uns:
        palette = dict(zip(vals.cat.categories, adata.uns[color_key]))
    else:
        colors = list(plt.get_cmap(cmap).colors)
        rng = np.random.default_rng(seed)
        rng.shuffle(colors)
        palette = {
            cat: mpl.colors.to_hex(colors[i % len(colors)])
            for i, cat in enumerate(vals.cat.categories)
        }

    point_colors = vals.map(palette)

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=point_colors,
        s=query_size,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=20,
    )

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=palette[cat],
            markeredgecolor=edgecolor,
            markeredgewidth=linewidth,
            markersize=6,
            label=str(cat),
        )
        for cat in vals.cat.categories
    ]
    ax.legend(
        handles=handles,
        title=feature,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    return ax


def predict_cmv(
    adata_frac,
    artifact,
    threshold=0.5,
):
    """Predict CMV serostatus from sample cell-type fractions.

    Loads a serialised classifier bundle, reindexes the fraction matrix onto the
    feature order the model was trained with, and writes predictions into
    ``adata_frac.obs``.

    Parameters
    ----------
    adata_frac : anndata.AnnData
        Sample-by-cell-type fraction object, e.g. from
        :func:`make_celltype_fraction_adata`. Modified in place.
    artifact : joblib bundle
        A loaded joblib bundle with keys ``"model"`` (a fitted estimator
        exposing ``predict_proba``), ``"features"`` (training feature order) and
        optionally ``"fill_value"`` (default 0.0) for cell types absent here.
        One is distributed with the package and can be imported via
        :func:`cmv_rf_model`.
    threshold : float, default 0.5
        Probability above which a sample is called ``"pos"``.

    Returns
    -------
    anndata.AnnData
        The same object, with ``obs["cmv_prob_pred"]`` (probability of class 1)
        and ``obs["cmv_pred"]`` (``"pos"`` / ``"neg"``).

    Notes
    -----
    Cell types present here but unseen during training are dropped; missing ones
    are filled with ``fill_value``. Wide divergence between the query and
    training cell-type panels will degrade predictions silently.
    """
    # Load model
    model = artifact["model"]
    features = artifact["features"]
    fill_value = artifact.get("fill_value", 0.0)

    # Extract and align cell-type fractions
    X = adata_frac.to_df().astype(float)
    X = X.reindex(columns=features, fill_value=fill_value)

    # Predict probability of class 1
    cmv_prob = model.predict_proba(X)[:, 1]

    # Call class 1 "pos" and class 0 "neg"
    cmv_pred = np.where(cmv_prob > threshold, "pos", "neg")

    # Store predictions
    adata_frac.obs["cmv_prob_pred"] = cmv_prob
    adata_frac.obs["cmv_pred"] = cmv_pred

    return adata_frac


def map_sample_obs_to_cells(
    adata,
    adata_frac,
    sample_col="sample_id",
    columns=None,
    inplace=True,
    copy_colors=True,
):
    """Broadcast sample-level annotations back onto individual cells.

    Joins columns from ``adata_frac.obs`` (indexed by sample) onto
    ``adata.obs`` by matching each cell's sample ID. Existing columns of the
    same name in ``adata.obs`` are replaced.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell-level object whose ``obs`` contains ``sample_col``.
    adata_frac : anndata.AnnData
        Sample-level object whose ``obs_names`` are sample IDs.
    sample_col : str, default "sample_id"
        Column in ``adata.obs`` used to match samples.
    columns : list of str, optional
        Columns of ``adata_frac.obs`` to transfer. Defaults to all of them.
    inplace : bool, default True
        If False, work on and return a copy of ``adata``.
    copy_colors : bool, default True
        Also copy any ``uns[f"{col}_colors"]`` palettes for the transferred
        columns.

    Returns
    -------
    anndata.AnnData
        The cell-level object with the sample annotations added. Cells whose
        sample is absent from ``adata_frac`` receive missing values.
    """
    if not inplace:
        adata = adata.copy()

    if columns is None:
        columns = adata_frac.obs.columns.tolist()

    meta = adata_frac.obs[columns].copy()
    meta.index = meta.index.astype(str)
    meta.index.name = sample_col

    obs = adata.obs.copy()
    obs[sample_col] = obs[sample_col].astype(str)
    obs = obs.drop(
        columns=[c for c in columns if c in obs.columns],
        errors="ignore",
    )

    adata.obs = obs.join(
        meta,
        on=sample_col,
        how="left",
    )

    if copy_colors:
        for col in columns:
            color_key = f"{col}_colors"
            if color_key in adata_frac.uns:
                adata.uns[color_key] = adata_frac.uns[color_key].copy()

    return adata

def cardinal_synthetic_reference():
    """
    Load the distributed Cardinal synthetic reference AnnData.
    """
    return sc.read(files("archetypist.data").joinpath("cardinal_synthetic_reference.h5ad"))

def cardinal_archetype_model():
    """
    Load the distributed CellTypist Cardinal archetype model
    """
    #celltypist does not understand PosixPath
    return celltypist.models.Model.load(str(files("archetypist.data").joinpath("cardinal_archetype_model.pkl")))

def cardinal_comp_cluster_model():
    """
    Load the distributed CellTypist Cardinal composition cluster model
    """
    #celltypist does not understand PosixPath
    return celltypist.models.Model.load(str(files("archetypist.data").joinpath("cardinal_comp_cluster_model.pkl")))

def cardinal_flat_genesymbol():
    """
    Load the distributed CellTypist Cardinal PBMC model
    """
    #celltypist does not understand PosixPath
    return celltypist.models.Model.load(str(files("archetypist.data").joinpath("cardinal_flat_genesymbol.pkl")))

def cmv_rf_model():
    """
    Load the distributed joblib cmv rf model
    """
    return joblib.load(files("archetypist.data").joinpath("cmv_rf_model_v1.joblib"))
