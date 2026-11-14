---
title: k-d trees: finding neighbors without checking everyone
description: How a k-d tree splits space, and how it prunes nearest-neighbor and radius search. A small 3D demo in Python.
date: 2026-11-13
category: demo
status: draft
---

Several applications, from navigation, path planning, up to machine learning keep asking the same question: _which points are close to a given point?_
@book{Papula2018,
In the k-nearest neighbors (kNN) method, this is queried for each prediction, in DBSCAN, it is queried for each point in the dataset [@ester1996dbscan], and when evaluating a vehicle's position within a lane, it is queried for multiple lane markings at each sampling step. The obvious answer is to measure the distance to every stored point. That costs $O(n)$ per query, and $O(n^2)$ when every point asks. Mathematically speaking, the length of the vector $\vec{a}$ between two points $p_1$ and $p_2$ can be calculated using the Euclidean norm. In general, the Euclidean norm [@Papula2018] of an $n$-dimensional vector is defined as:

$$
|\vec{a}| = \sqrt{a_1^2 + a_2^2 + \dots + a_n^2}
= \sqrt{\sum_{i=1}^{n} a_i^2}
$$

Other distance measures, such as the Mahalanobis distance, can also be used depending on the application and the properties of the data.

A k-d tree [@bentley1975kdtree] answers the same question while skipping most of the points as it borrows the idea from a search tree. This post shows the idea with a small implementation and a 3D demo of 125 random points. Hence, the values are given in $\mathbb{R}^3$, spanning the three axes $x$, $y$, and $z$.

## The idea: A binary search tree over k coordinates

A binary search tree sorts numbers along a single axis. A k-d tree does the same for points with $k$ coordinates, by **alternating the axis at each level**:

- The root splits all points at the median of their **x** values. Points with a smaller x go left, points with a larger x go right.
- Both children split their range at the median of **y**, the next level at **z**, and then it starts over with x.

Every node therefore cuts its region of space into two boxes with a plane. Taking the median keeps both halves the same size, so the tree has about $\log_2 n$ levels.

![The top three levels of the tree: the root splits on x, its children on y, the next level on z](kdtree.scene.json#splits)

Building the tree uses recursion. In order to have a performant code, we are using the numpy library. `np.argpartition` finds the median without sorting all points, which keeps the work per level at $O(n)$:

```python
def _build_tree(self, indices, depth):
    n = indices.size
    if n == 0:
        return None
    if n <= self._leaf_size:
        return Leaf(tuple(indices.tolist()))

    axis = depth % self._k  # x, y, z, x, …
    median = n // 2
    order = np.argpartition(self._coordinates[indices, axis], median)
    index = int(indices[order[median]])

    return Node(
        point_index=index,
        axis=axis,
        split=float(self._coordinates[index, axis]),
        left=self._build_tree(indices[order[:median]], depth + 1),
        right=self._build_tree(indices[order[median + 1 :]], depth + 1),
    )
```

Small groups of at most `leaf_size` points are not split further. They are stored as a `Leaf` and searched one by one. In Python this is faster than descending through several more tiny nodes. The demo uses `leaf_size=1` so that every split is visible.

## Why walking down is not enough

The obvious idea for a nearest-neighbour search works like a lookup in a binary search tree. Start at the root, compare the query with the splitting plane, go left or right, and repeat until you reach the query's own cell. The closest point on that path is the answer. This is fast and it took at most $\log_2 n$ comparisons, but it is unfortunately not for all cases correct. The splitting planes ignore how close points are as a huge limitation of the algorithm. Two points can be neighbors in space and still end up on different sides of a plane, far apart in the tree. If the query sits close to a plane, its nearest neighbor may lie behind it in closest proximity.

![The limitation: The walk down finds the best point in the query's own cell, but the true nearest point lies just across the root plane](kdtree.scene.json#pitfall)

In this example the query lies only 0.027 units in front of the root plane at $x = 0.457$. The top-to-bottom-walk compares 7 points and returns one at distance 0.165. However, the true nearest point is at 0.040, on the other side of the very first split. The algorithm never looks there. In this tree, a walk down without looking back returns the wrong point for about **1 in 3** random queries.

The figure also shows how to detect the problem. Draw the sphere around the query that reaches the best point so far. If that sphere crosses a splitting plane, a closer point may be waiting on the other side, and that side has to be searched as well. If the sphere stays on its side, the whole other side can be ruled out.

## Nearest neighbour: Prune whole boxes

The correct search therefore also checks the surroundings. It walks down the tree towards the query and remembers the closest point found so far. The interesting step comes on the way back. A subtree on the **other side** of a splitting plane can only contain a closer point if the plane itself is closer than the best distance so far [@friedman1977nn]. If it isn't, the whole subtree is skipped without looking at a single point inside it.

```python
offset = point[axis] - split
near, far = (left, right) if offset < 0 else (right, left)
stack.append((far, abs(offset)))  # only searched if abs(offset) < best distance
stack.append((near, 0.0))         # popped first
```

`abs(offset)` is the distance from the query to the plane, a lower bound for every point behind it. Because the near side is searched first, the best distance shrinks early and more far sides can be dropped.

![Nearest-neighbour search: the points light up in the order the search compares them](kdtree.scene.json#nearest)

Here the search compared 47 of the 125 points. The sphere around the query reaches up to the nearest point. Every box that this sphere does not touch is skipped.

## Radius search: The same pruning, a fixed sphere

DBSCAN does not need the single nearest point. It needs all points within a defined radius $r$ ($ \hat{=} $ eps). The pruning rule becomes even simpler, because the sphere does not shrink. A side of a plane is only visited if the sphere reaches across the plane:

```python
value = point[axis]
if value - radius <= split:
    stack.append(left)
if value + radius >= split:
    stack.append(right)
```

![Radius search with r = 0.2: 6 points found, 57 compared](kdtree.scene.json#radius)

## How much does it save?

The numbers in the demo are small: 47 and 57 out of 125 points. With so few points, a large share of the tree is always near the query. The gain grows with the size of the data. The number of levels grows only with $\log n$, while a full scan grows with $n$. For a million points in 3D, a nearest-neighbour query compares a few hundred points instead of a million.

The tree has one further known weakness: **many dimensions**. In 20 dimensions, a sphere around the query crosses almost every splitting plane, nearly nothing gets pruned, and the tree only adds overhead. The implementation therefore takes a `max_tree_dims` parameter (default 6). Above it, no tree is built and the search falls back to a vectorized full scan with NumPy:

```python
distances = np.linalg.norm(self._coordinates - np.asarray(point, dtype=float), axis=1)
return self._points[int(np.argmin(distances))]
```
