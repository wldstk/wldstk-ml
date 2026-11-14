"""Unit tests for wldstk_ml.utils.trees."""

import math
import random
import unittest

import numpy as np

from wldstk_ml.utils.trees import KdTree, Leaf, Node, Point

POINTS = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]
LEAF_SIZES = (1, 4, 16)

Subtree = Node | Leaf | None


def random_points(rng: random.Random, n: int, k: int, grid: bool = False) -> list[Point]:
    """Return n random points with k coordinates, rounded to a coarse grid if requested."""
    points: list[Point] = [tuple(rng.gauss(0, 1) for _ in range(k)) for _ in range(n)]
    return [tuple(round(c, 1) for c in p) for p in points] if grid else points


def brute_force_radius(points: list[Point], point: Point, radius: float) -> list[int]:
    """Return the indices of all points within the radius by checking every point."""
    return [i for i, p in enumerate(points) if math.dist(p, point) <= radius]


def root_of(tree: KdTree) -> Subtree:
    """Return the root of the tree, for checking its structure."""
    return tree._root  # pylint: disable=protected-access


def subtree_indices(node: Subtree) -> list[int]:
    """Return the indices of all points stored in a subtree."""
    if node is None:
        return []
    if isinstance(node, Leaf):
        return list(node.indices)
    return [node.point_index, *subtree_indices(node.left), *subtree_indices(node.right)]


def subtree_leaves(node: Subtree) -> list[Leaf]:
    """Return all leaves of a subtree."""
    if node is None:
        return []
    if isinstance(node, Leaf):
        return [node]
    return [*subtree_leaves(node.left), *subtree_leaves(node.right)]


def height(node: Subtree) -> int:
    """Return the number of levels of a subtree, counting a leaf as one level."""
    if node is None:
        return 0
    if isinstance(node, Leaf):
        return 1
    return 1 + max(height(node.left), height(node.right))


class TestConstruction(unittest.TestCase):
    """Tests building the tree."""

    def test_k_is_inferred_from_points(self) -> None:
        """Without k, the tree uses the number of coordinates of the points."""
        self.assertEqual(KdTree(POINTS).k, 2)
        self.assertEqual(KdTree([(1, 2, 3)]).k, 3)

    def test_explicit_k(self) -> None:
        """An explicit k is kept."""
        self.assertEqual(KdTree(POINTS, k=2).k, 2)

    def test_empty_tree(self) -> None:
        """An empty tree has no root, and k defaults to 0 unless given."""
        self.assertIsNone(root_of(KdTree([])))
        self.assertEqual(KdTree([]).k, 0)
        self.assertEqual(KdTree([], k=3).k, 3)

    def test_points_of_different_dimensions_raise(self) -> None:
        """All points must have the same number of coordinates."""
        with self.assertRaises(ValueError):
            KdTree([(1, 2), (1, 2, 3)])

    def test_points_not_matching_explicit_k_raise(self) -> None:
        """All points must have k coordinates."""
        with self.assertRaises(ValueError):
            KdTree(POINTS, k=3)

    def test_points_without_coordinates_raise_value_error(self) -> None:
        """Points need at least one coordinate."""
        with self.assertRaises(ValueError):
            KdTree([()])

    def test_explicit_k_below_one_raises(self) -> None:
        """k must be at least 1 when given, even for an empty tree."""
        for k in (0, -1):
            with self.subTest(k=k), self.assertRaises(ValueError):
                KdTree([], k=k)

    def test_leaf_size_below_one_raises(self) -> None:
        """Leaves must be able to hold at least one point."""
        with self.assertRaises(ValueError):
            KdTree(POINTS, leaf_size=0)

    def test_input_is_not_modified(self) -> None:
        """Building the tree leaves the caller's list unchanged."""
        points = list(reversed(POINTS))
        KdTree(points)
        self.assertEqual(points, list(reversed(POINTS)))

    def test_accepts_lists_and_tuples(self) -> None:
        """Points can be lists or tuples."""
        self.assertEqual(KdTree([[0.0, 1.0], (2.0, 3.0)]).closest_point((2, 2)), (2.0, 3.0))

    def test_accepts_numpy_array(self) -> None:
        """Points can be the rows of a numpy array, also without giving k."""
        tree = KdTree(np.array(POINTS, dtype=float))
        self.assertEqual(tree.k, 2)
        self.assertCountEqual(tree.query_radius((1, 1), 1.0), [0, 1, 2])


class TestStructure(unittest.TestCase):
    """Tests the shape of the built tree."""

    def setUp(self) -> None:
        """Create trees of different sizes and dimensions, with and without duplicates."""
        rng = random.Random(0)
        self.cases = [
            (n, k, grid, random_points(rng, n, k, grid))
            for n in (1, 2, 3, 10, 100)
            for k in (1, 2, 3)
            for grid in (False, True)
        ]

    def test_every_point_is_stored_once(self) -> None:
        """Each input index appears exactly once, in a node or a leaf."""
        for n, k, grid, points in self.cases:
            for leaf_size in LEAF_SIZES:
                with self.subTest(n=n, k=k, grid=grid, leaf_size=leaf_size):
                    root = root_of(KdTree(points, leaf_size=leaf_size))
                    self.assertEqual(sorted(subtree_indices(root)), list(range(n)))

    def test_leaves_respect_leaf_size(self) -> None:
        """Every leaf holds between 1 and leaf_size points."""
        for n, k, grid, points in self.cases:
            for leaf_size in LEAF_SIZES:
                with self.subTest(n=n, k=k, grid=grid, leaf_size=leaf_size):
                    for leaf in subtree_leaves(root_of(KdTree(points, leaf_size=leaf_size))):
                        self.assertTrue(1 <= len(leaf.indices) <= leaf_size)

    def test_small_input_is_a_single_leaf(self) -> None:
        """Up to leaf_size points need no splitting."""
        self.assertEqual(root_of(KdTree(POINTS, leaf_size=5)), Leaf((0, 1, 2, 3, 4)))

    def test_binary_search_tree_property(self) -> None:
        """Left subtrees hold smaller or equal values, right subtrees larger or equal ones."""

        def check(node: Subtree, points: list[Point], depth: int, k: int) -> None:
            if not isinstance(node, Node):
                return
            self.assertEqual(node.axis, depth % k)
            self.assertEqual(node.split, points[node.point_index][node.axis])
            for index in subtree_indices(node.left):
                self.assertLessEqual(points[index][node.axis], node.split)
            for index in subtree_indices(node.right):
                self.assertGreaterEqual(points[index][node.axis], node.split)
            check(node.left, points, depth + 1, k)
            check(node.right, points, depth + 1, k)

        for n, k, grid, points in self.cases:
            for leaf_size in LEAF_SIZES:
                with self.subTest(n=n, k=k, grid=grid, leaf_size=leaf_size):
                    check(root_of(KdTree(points, leaf_size=leaf_size)), points, 0, k)

    def test_tree_is_balanced(self) -> None:
        """Splitting at the median gives the minimal height."""
        for n, k, grid, points in self.cases:
            with self.subTest(n=n, k=k, grid=grid):
                tree = KdTree(points, leaf_size=1)
                self.assertEqual(height(root_of(tree)), math.floor(math.log2(n)) + 1)


class TestDistanceSquared(unittest.TestCase):
    """Tests the distance function."""

    def test_values(self) -> None:
        """Squared Euclidean distance in one, two and three dimensions."""
        self.assertEqual(KdTree.distance_squared((1,), (4,)), 9)
        self.assertEqual(KdTree.distance_squared((1, 2), (4, 6)), 25)
        self.assertEqual(KdTree.distance_squared((1, 2, 3), (1, 2, 3)), 0)

    def test_is_symmetric(self) -> None:
        """The order of the points does not matter."""
        self.assertEqual(
            KdTree.distance_squared((0.5, -2), (3, 1)), KdTree.distance_squared((3, 1), (0.5, -2))
        )

    def test_different_dimensions_raise(self) -> None:
        """Points of different dimensions cannot be compared."""
        with self.assertRaises(ValueError):
            KdTree.distance_squared((1, 2), (1, 2, 3))


class TestClosestPoint(unittest.TestCase):
    """Tests the nearest-neighbor search."""

    def test_example(self) -> None:
        """The closest grid point to (1.49, 1.49) is (1, 1)."""
        self.assertEqual(KdTree(POINTS).closest_point((1.49, 1.49)), (1, 1))

    def test_stored_point_finds_itself(self) -> None:
        """A query at a stored point returns that point."""
        tree = KdTree(POINTS)
        for point in POINTS:
            with self.subTest(point=point):
                self.assertEqual(tree.closest_point(point), point)

    def test_returns_stored_object(self) -> None:
        """The result is the stored point itself, not a copy."""
        points = [(0.0, 0.0), (5.0, 5.0)]
        self.assertIs(KdTree(points).closest_point((4, 4)), points[1])

    def test_query_far_outside(self) -> None:
        """Queries far away from all points still find the nearest one."""
        self.assertEqual(KdTree(POINTS).closest_point((100, 100)), (3, 3))
        self.assertEqual(KdTree(POINTS).closest_point((-100, 1.2)), (1, 1))

    def test_empty_tree_returns_none(self) -> None:
        """An empty tree has no closest point."""
        self.assertIsNone(KdTree([]).closest_point((1, 2)))

    def test_wrong_dimension_raises(self) -> None:
        """The query must have k coordinates."""
        with self.assertRaises(ValueError):
            KdTree(POINTS).closest_point((1, 2, 3))

    def test_matches_brute_force(self) -> None:
        """Random queries find a point at the minimal distance, in several dimensions."""
        rng = random.Random(1)
        for k in (1, 2, 3, 5):
            for grid in (False, True):
                points = random_points(rng, 200, k, grid)
                tree = KdTree(points)
                for _ in range(50):
                    query = tuple(rng.gauss(0, 1.5) for _ in range(k))
                    found = tree.closest_point(query)
                    assert found is not None
                    best = min(KdTree.distance_squared(query, p) for p in points)
                    with self.subTest(k=k, grid=grid, query=query):
                        self.assertEqual(KdTree.distance_squared(query, found), best)


class TestQueryRadius(unittest.TestCase):
    """Tests the search for all points within a radius."""

    def test_example(self) -> None:
        """The points within 1 of (1, 1) are itself and its two direct neighbors."""
        self.assertCountEqual(KdTree(POINTS).query_radius((1, 1), 1.0), [0, 1, 2])

    def test_boundary_is_included(self) -> None:
        """A point exactly at distance radius is a match."""
        self.assertCountEqual(KdTree([(0, 0), (3, 4)]).query_radius((0, 0), 5.0), [0, 1])

    def test_just_outside_is_excluded(self) -> None:
        """A point slightly farther than the radius is not a match."""
        self.assertEqual(KdTree([(0, 0), (3, 4)]).query_radius((0, 0), 4.999), [0])

    def test_radius_zero_finds_exact_duplicates(self) -> None:
        """All copies of a point are found, each with its own index."""
        points = [(1, 1), (2, 2), (1, 1), (1, 1)]
        self.assertCountEqual(KdTree(points).query_radius((1, 1), 0.0), [0, 2, 3])

    def test_large_radius_returns_everything(self) -> None:
        """A radius covering all points returns every index once."""
        self.assertCountEqual(KdTree(POINTS).query_radius((2, 2), 100.0), range(len(POINTS)))

    def test_no_match(self) -> None:
        """A query far from all points returns an empty list."""
        self.assertEqual(KdTree(POINTS).query_radius((50, 50), 1.0), [])

    def test_empty_tree(self) -> None:
        """An empty tree has no matches."""
        self.assertEqual(KdTree([]).query_radius((1, 2), 10.0), [])

    def test_wrong_dimension_raises(self) -> None:
        """The query must have k coordinates."""
        with self.assertRaises(ValueError):
            KdTree(POINTS).query_radius((1,), 1.0)

    def test_matches_brute_force(self) -> None:
        """Random queries find exactly the brute-force matches, without duplicates."""
        rng = random.Random(2)
        for k in (1, 2, 3, 5):
            for grid in (False, True):
                points = random_points(rng, 200, k, grid)
                tree = KdTree(points)
                for radius in (0.0, 0.1, 0.5, 1.5):
                    for _ in range(20):
                        query = tuple(rng.gauss(0, 1.5) for _ in range(k))
                        found = tree.query_radius(query, radius)
                        with self.subTest(k=k, grid=grid, radius=radius, query=query):
                            self.assertEqual(len(found), len(set(found)))
                            self.assertCountEqual(found, brute_force_radius(points, query, radius))


class TestLeafSizesAndBruteForce(unittest.TestCase):
    """Tests that leaf sizes and the numpy fallback for many dimensions give the same results."""

    def test_all_modes_match_brute_force(self) -> None:
        """Every leaf size, and the fallback without a tree, finds the same points."""
        rng = random.Random(3)
        for k in (2, 3, 10):
            points = random_points(rng, 300, k)
            trees = {
                f"leaf_size={leaf_size}": KdTree(points, leaf_size=leaf_size)
                for leaf_size in LEAF_SIZES
            }
            trees["no tree"] = KdTree(points, max_tree_dims=0)
            for _ in range(30):
                query = tuple(rng.gauss(0, 1.5) for _ in range(k))
                best = min(KdTree.distance_squared(query, p) for p in points)
                expected = brute_force_radius(points, query, 1.0)
                for mode, tree in trees.items():
                    with self.subTest(k=k, mode=mode, query=query):
                        found = tree.closest_point(query)
                        assert found is not None
                        self.assertAlmostEqual(KdTree.distance_squared(query, found), best)
                        self.assertCountEqual(tree.query_radius(query, 1.0), expected)

    def test_many_dimensions_use_no_tree(self) -> None:
        """Above max_tree_dims no tree is built; queries still work."""
        tree = KdTree([(0.0,) * 9, (1.0,) * 9])
        self.assertIsNone(root_of(tree))
        self.assertEqual(tree.query_radius((0.0,) * 9, 0.5), [0])
        self.assertEqual(tree.closest_point((0.9,) * 9), (1.0,) * 9)

    def test_fallback_returns_stored_object(self) -> None:
        """Without a tree, closest_point still returns the stored point itself."""
        points = [(0.0, 0.0), (5.0, 5.0)]
        self.assertIs(KdTree(points, max_tree_dims=0).closest_point((4, 4)), points[1])


if __name__ == "__main__":
    unittest.main()
