# k-Means Clustering

k-Means splits unlabeled data into $k$ groups so that samples in the same group are close
together. Each group is represented by its mean, the **centroid**. The algorithm alternates
between assigning samples to their nearest centroid and moving each centroid to the mean of
its samples, until nothing changes.

This document covers the theory and how it maps to [k_means.py](k_means.py). The equations
use LaTeX math, which GitHub and the VS Code Markdown preview both render.

## Notation

| Symbol | Meaning | In the code |
|---|---|---|
| $n$ | number of samples | `len(X)` |
| $d$ | number of features | `X.shape[1]` |
| $k$ | number of clusters | `self.k` |
| $x_i \in \mathbb{R}^d$ | sample $i$, with $i = 1, \dots, n$ | `X[i]` |
| $\mu_j \in \mathbb{R}^d$ | centroid of cluster $j$, with $j = 1, \dots, k$ | `centroids[j]` |
| $c_i \in \{1, \dots, k\}$ | cluster assigned to sample $i$ | `labels[i]` |
| $C_j = \{\, i : c_i = j \,\}$ | set of samples in cluster $j$ | `labels == j` |
| $n_j$ | number of samples in cluster $j$ | `counts[j]` |

The code counts from 0, so its labels run from $0$ to $k-1$.

## Distance

k-Means measures closeness with the **Euclidean distance** (L2 norm). For a sample $x$ and a
centroid $\mu$ with $d$ features:

$$
\operatorname{dist}(x, \mu) = \lVert x - \mu \rVert_2 = \sqrt{\sum_{l=1}^{d} (x_l - \mu_l)^2}
$$

For example, with $x = (1, 2)$ and $\mu = (4, 6)$:

$$
\lVert x - \mu \rVert_2 = \sqrt{(1 - 4)^2 + (2 - 6)^2} = \sqrt{9 + 16} = 5
$$

The theory below uses the **squared** distance, $\lVert x - \mu \rVert_2^2$, because it has
no square root and is easy to differentiate. For choosing a nearest centroid the two are
interchangeable: since $f(t) = t^2$ is strictly increasing for $t \ge 0$, both give the same
result.

$$
\arg\min_{j} \lVert x - \mu_j \rVert_2 = \arg\min_{j} \lVert x - \mu_j \rVert_2^2
$$

The code uses the plain distance. Because every feature contributes through a squared
difference, features with large numeric ranges dominate (see [Limitations](#limitations)).

## Objective

k-Means minimizes the **within-cluster sum of squares** (WCSS, also called *inertia*), the
total squared distance from each sample to its centroid:

$$
J(c, \mu) = \sum_{j=1}^{k} \sum_{i \in C_j} \lVert x_i - \mu_j \rVert_2^2
          = \sum_{i=1}^{n} \lVert x_i - \mu_{c_i} \rVert_2^2
$$

Lower is better, and $J = 0$ means every sample sits exactly on its centroid. Finding the
global minimum is NP-hard in general, so k-Means uses a heuristic that finds a good local
minimum. The standard one, described next, is **Lloyd's algorithm**.

## Algorithm

**Input:** data $x_1, \dots, x_n$, number of clusters $k$, iteration limit $T$.

1. **Initialize:** choose $k$ distinct samples at random as the centroids
   $\mu_1, \dots, \mu_k$.
2. **Assign:** give each sample the label of its nearest centroid:

   $$
   c_i = \arg\min_{j \in \{1, \dots, k\}} \lVert x_i - \mu_j \rVert_2
   $$

3. **Update:** move each centroid to the mean of its samples:

   $$
   \mu_j = \frac{1}{n_j} \sum_{i \in C_j} x_i
   $$

4. **Repeat** steps 2 and 3 until the centroids stop changing or $T$ iterations have run.

**Output:** the labels $c_i$ and the centroids $\mu_j$.

The assignment step divides the feature space into one convex region per centroid, a
*Voronoi partition*. The boundary between two clusters is the set of points equally far from
both centroids, which is a straight line in 2D and a hyperplane in higher dimensions.

## Why the update uses the mean

With the assignments fixed, each cluster's part of $J$ can be minimized separately. Take the
gradient with respect to $\mu_j$ and set it to zero:

$$
\frac{\partial}{\partial \mu_j} \sum_{i \in C_j} \lVert x_i - \mu_j \rVert_2^2
= -2 \sum_{i \in C_j} (x_i - \mu_j) = 0
$$

$$
\sum_{i \in C_j} x_i = n_j \, \mu_j
\quad\Longrightarrow\quad
\mu_j = \frac{1}{n_j} \sum_{i \in C_j} x_i
$$

The function is convex in $\mu_j$, so this is the minimum: the mean is the optimal centroid
for squared Euclidean distance. This is where the name comes from. Other distances lead to
other centers; for example, the Manhattan (L1) distance leads to the median (*k-medians*).

## Why it converges

Each step can only lower $J$ or leave it unchanged:

- **Assign** picks the closest centroid for every sample, so no term of $J$ grows.
- **Update** picks the optimal center for every cluster, as shown above.

$$
J^{(t+1)} \le J^{(t)} \quad \text{for every iteration } t
$$

$J$ is bounded below by 0, and there are at most $k^n$ possible assignments, so the algorithm
settles after finitely many steps. It stops at a **local** minimum, which is not necessarily
the global one, so the result depends on the starting centroids.

## Worked example

Six samples in 2D form two obvious groups, and $k = 2$:

$$
x_1 = (1, 1),\; x_2 = (2, 1),\; x_3 = (1, 2),\; x_4 = (8, 8),\; x_5 = (9, 8),\; x_6 = (8, 9)
$$

Deliberately start with a bad initialization, both centroids in the same group:
$\mu_1 = (1, 1)$ and $\mu_2 = (2, 1)$.

| Iteration | Cluster 1 | Cluster 2 | $J$ after assigning | New $\mu_1$ | New $\mu_2$ | $J$ after updating |
|---|---|---|---|---|---|---|
| 1 | $x_1, x_3$ | $x_2, x_4, x_5, x_6$ | 284.00 | $(1, 1.5)$ | $(6.75, 6.5)$ | 72.25 |
| 2 | $x_1, x_2, x_3$ | $x_4, x_5, x_6$ | 20.69 | $(1.33, 1.33)$ | $(8.33, 8.33)$ | 2.67 |
| 3 | $x_1, x_2, x_3$ | $x_4, x_5, x_6$ | 2.67 | unchanged | unchanged | 2.67 |

In iteration 1, $x_4$ belongs to cluster 2 because
$\lVert x_4 - \mu_2 \rVert_2 = \sqrt{6^2 + 7^2} \approx 9.22$ is less than
$\lVert x_4 - \mu_1 \rVert_2 = \sqrt{7^2 + 7^2} \approx 9.90$.

$J$ falls at every step, and iteration 3 changes nothing, so the algorithm stops with the two
groups correctly separated.

## Implementation

The [KMeans](k_means.py) class implements each step as a numpy array operation.

| Step | Method | Notes |
|---|---|---|
| Initialize | `_init_random_centroids` | Draws $k$ distinct samples. Raises `ValueError` if $k > n$. |
| Assign | `_assign_labels` | Computes all $n \times k$ distances at once. Ties go to the lowest index. |
| Update | `_calculate_centroids` | Computes all cluster means at once. |
| Run | `predict` | Runs the loop, stores `self.centroids`, and returns the labels. |

`predict` clusters the data it is given. There is no separate `fit` step, and calling
`predict` again starts over from new random centroids, so it cannot assign new samples to
existing clusters.

**Update step.** Let $R \in \{0, 1\}^{n \times k}$ be the one-hot assignment matrix, with
$R_{ij} = 1$ when $c_i = j$, and let $X \in \mathbb{R}^{n \times d}$ hold the samples as rows.
The cluster sums and sizes are then:

$$
S = R^\top X \in \mathbb{R}^{k \times d},
\qquad
n_j = \sum_{i=1}^{n} R_{ij}
$$

Row $j$ of $S$ divided by $n_j$ is $\mu_j$. In the code, `one_hot` is $R$, `sums` is $S$, and
`counts` holds the $n_j$.

**Empty clusters.** A cluster can lose all its samples, which makes $n_j = 0$ and the mean
undefined. The code keeps the previous centroid in that case:

$$
\mu_j^{(t+1)} =
\begin{cases}
\dfrac{1}{n_j} \displaystyle\sum_{i \in C_j} x_i & \text{if } n_j > 0 \\[2ex]
\mu_j^{(t)} & \text{if } n_j = 0
\end{cases}
$$

An empty cluster adds nothing to $J$, so this does not break the convergence argument.

**Stopping.** The loop ends when `new_centroids` exactly equals `centroids`, or after
`max_iter` updates. Exact comparison of floats is safe here: the same labels always produce
the same means, so the values stop changing as soon as the labels do.

## Vectorized distances

How `_assign_labels` in [k_means.py](k_means.py) computes the distance from every sample to
every centroid in a single line, without Python loops:

```python
distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
labels = np.argmin(distances, axis=1)
```

It works in three steps: **add axes → broadcast → reduce**.

The examples below use the Iris shapes: `X` is `(150, 4)` (150 samples, 4 features) and
`centroids` is `(3, 4)` (k = 3).

### Step 1: Add axes with `None`

`None` (same as `np.newaxis`) inside an index adds a new axis of length 1 at that position.
The data stays the same; only the shape changes.

```
X                      (150, 4)
X[:, None, :]          (150, 1, 4)    "each sample, compared against ? centroids"

centroids              (3, 4)
centroids[None, :, :]  (1, 3, 4)      "? samples, each compared against these centroids"
```

Read the three axes as **(sample, centroid, feature)**:

- `X` gets a length-1 axis in the *centroid* position.
- `centroids` gets a length-1 axis in the *sample* position.

> `centroids[None, :, :]` can also be written `centroids[None]`. numpy would even add that
> axis automatically, but writing it out makes the intent clear.

### Step 2: Subtract with broadcasting

When two arrays have different shapes, numpy compares them axis by axis. Where one of them
has length 1, it is repeated (without actually copying memory) to match the other:

```
X[:, None, :]          (150, 1, 4)
centroids[None, :, :]  (  1, 3, 4)
                        ───────────
difference             (150, 3, 4)
```

- Axis 0: 150 vs 1 → each centroid is reused for all 150 samples.
- Axis 1: 1 vs 3 → each sample is reused for all 3 centroids.
- Axis 2: 4 vs 4 → the features line up one-to-one.

Result: `diff[i, j, :] = X[i] - centroids[j]`, the difference vector between **sample i**
and **centroid j**, for every combination.

### Step 3: Take the norm over `axis=2`

`np.linalg.norm(..., axis=2)` computes `sqrt(sum of squares)` along the **feature** axis and
removes it:

```
(150, 3, 4)  --norm over features-->  (150, 3)
```

So `distances[i, j]` is the Euclidean distance between sample i and centroid j. Each row
belongs to a sample, and each column to a centroid.

Then `np.argmin(distances, axis=1)` finds the smallest value in each **row**, which is the
closest centroid for each sample. The result has shape `(150,)`. If two centroids are equally
close, `argmin` picks the first one.

### Small example

2 samples, 2 centroids, 2 features:

```python
X         = [[0, 0],      # sample 0
             [5, 5]]      # sample 1
centroids = [[1, 0],      # centroid 0
             [5, 4]]      # centroid 1

X[:, None, :] - centroids[None, :, :]   # shape (2, 2, 2)
# sample 0: [[0-1, 0-0], [0-5, 0-4]] = [[-1,  0], [-5, -4]]
# sample 1: [[5-1, 5-0], [5-5, 5-4]] = [[ 4,  5], [ 0,  1]]

np.linalg.norm(..., axis=2)             # shape (2, 2)
# [[1.0,  6.40],    sample 0: close to centroid 0
#  [6.40, 1.0 ]]    sample 1: close to centroid 1

np.argmin(..., axis=1)                  # -> [0, 1]
```

### The equivalent loop

```python
distances = np.empty((len(X), len(centroids)))
for i in range(len(X)):
    for j in range(len(centroids)):
        distances[i, j] = np.sqrt(np.sum((X[i] - centroids[j]) ** 2))
```

The one-liner does the same work, but the loops run inside numpy's C code instead of in
Python. That's where the speedup comes from.

### Cost

The intermediate `(n_samples, k, n_features)` difference array really is allocated in
memory. That is tiny for Iris (150 × 3 × 4 values), but for millions of samples you would
compute the distances in chunks of samples instead.

### Rule of thumb

To combine **every** element of `A` (shape `(n, d)`) with **every** element of `B`
(shape `(m, d)`):

1. Give `A` a length-1 axis where `B`'s items go: `A[:, None, :]` → `(n, 1, d)`.
2. Give `B` a length-1 axis where `A`'s items go: `B[None, :, :]` → `(1, m, d)`.
3. Combine them: broadcasting produces `(n, m, d)`.
4. Reduce the axis you don't need anymore (here the feature axis, `axis=2`) → `(n, m)`.

## Complexity

| Step | Time |
|---|---|
| Assign | $O(n \cdot k \cdot d)$ |
| Update | $O(n \cdot k \cdot d)$ |
| Full run of $T$ iterations | $O(T \cdot n \cdot k \cdot d)$ |

Memory is also $O(n \cdot k \cdot d)$, because of the temporary difference array in the
assign step. Runs usually converge well before `max_iter`.

## Limitations

**Results depend on the starting centroids.** Over 100 random seeds on Iris ($k = 3$, all four
features), 84 runs reached about 89% accuracy and 16 got stuck in a worse local minimum:

| Runs | $J$ | Accuracy |
|---|---|---|
| 84 | about 78.9 | 88.7% to 89.3% |
| 16 | 142.8 to 145.5 | 66.7% |

Two common fixes are to run several times and keep the result with the lowest $J$ (`n_init`
in scikit-learn), or to start with **k-means++**, which spreads the initial centroids apart.
This implementation does neither yet.

**$k$ must be chosen in advance.** $J$ always falls as $k$ grows, so it can't pick $k$ by
itself. Common approaches are the *elbow method* (plot $J$ against $k$ and look for the bend)
and the *silhouette score*.

**Features need comparable scales.** A feature measured in thousands dominates one measured in
fractions, because distance adds up squared differences. Standardize such features first:

$$
x_l' = \frac{x_l - \bar{x}_l}{\sigma_l}
$$

Iris doesn't need this, since all its features are lengths in centimeters.

**It prefers compact, similarly sized, round clusters.** The straight boundaries between
clusters can't separate shapes like rings or crescents. For those, consider DBSCAN or spectral
clustering.

**Outliers pull centroids,** because the mean is sensitive to extreme values and squared
distance amplifies them.

## Comparing clusters with ground truth

Cluster numbers are arbitrary: cluster 0 may correspond to class 2. Before computing accuracy,
map each cluster to the most common true label within it, which is what
`align_cluster_labels` in [visualization.py](../utils/visualization.py) does:

$$
\hat{y}_i = \arg\max_{y} \; \bigl|\{\, m \in C_{c_i} : y_m = y \,\}\bigr|
$$

$$
\text{accuracy} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(\hat{y}_i = y_i)
$$

Two clusters can map to the same class this way, so treat the result as an evaluation aid. It
requires labels, which real clustering problems usually don't have.

Also keep in mind that a 2D plot of higher-dimensional data shows only part of the picture.
Points can look wrongly assigned in a plot of two Iris features because the clustering used all
four.
