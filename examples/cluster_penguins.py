#!/usr/bin/env python3
"""tidy3 → NumPy → scikit-learn → plot3: k-means on the Palmer penguins.

tidy3 does the pandas-style work (load, clean, scale), ``to_numpy()`` hands
the matrix to scikit-learn, and the cluster labels come straight back into
the pipe for counting and plotting. Needs network access for the data::

    pip install "tidy3[plot3]" scikit-learn
    python examples/cluster_penguins.py

Figures are written to examples/output/ as interactive HTML and PNG.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.cluster import KMeans

from plot3 import aes, coord_3d, geom_point, geom_point3d, ggplot, ggsave, labs
from tidy3 import across, compute, count, drop_na, group_by, mean, mutate, scan_csv, sd, summarise

URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
OUT = Path(__file__).parent / "output"

# 1. Load and clean. The pipe is lazy; compute() runs it once, so the file
#    downloads once and every step below reuses the result.
penguins = scan_csv(URL, null_values="NA") >> drop_na(*FEATURES) >> compute()

# 2. Scale each feature to mean 0, sd 1 (R's scale()), then hand off to NumPy.
X = (
    penguins
    >> mutate(across(FEATURES, lambda x: (x - mean(x)) / sd(x)))
).to_numpy(columns=FEATURES)

# 3. scikit-learn sees an ordinary (n, 3) float array.
kmeans = KMeans(n_clusters=3, n_init=10, random_state=1).fit(X)

# 4. The labels come back as a column; everything after is tidy3 again.
clustered = penguins >> mutate(cluster=kmeans.labels_ + 1)
print(clustered >> count("species", "cluster"))

# Cluster centres in the original units: a group_by, not an inverse transform.
centres = clustered >> group_by("cluster") >> summarise(across(FEATURES, mean))
print(centres)

# 5. Plot the clusters, with each penguin's true species as its shape.
flat = (
    clustered
    >> ggplot(aes(x="flipper_length_mm", y="bill_length_mm", colour="factor(cluster)"))
    + geom_point(aes(shape="species"), alpha=0.75)
    + geom_point(data=centres, size=16, shape="cross", colour="black")
    + labs(title="k-means on bill and flipper size", colour="cluster",
           x="Flipper length (mm)", y="Bill length (mm)")
)
orbit = (
    clustered
    >> ggplot(aes(x="bill_length_mm", y="bill_depth_mm", z="flipper_length_mm",
                  colour="factor(cluster)"))
    + geom_point3d()
    + coord_3d(aspect="equal")   # a cube: the features span very different ranges
    + labs(title="The three features k-means used", colour="cluster",
           x="Bill length", y="Bill depth", z="Flipper length")
)

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fig in {"penguin_clusters": flat, "penguin_clusters_3d": orbit}.items():
        ggsave(str(OUT / f"{name}.html"), fig)
        ggsave(str(OUT / f"{name}.png"), fig, width=6, height=4.5, units="in", dpi=150)
