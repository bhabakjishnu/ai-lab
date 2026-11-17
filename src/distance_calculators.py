"""Manhattan and Euclidean Distance Calculators (Informed Search & Heuristics).

This module implements foundational distance metrics used as heuristic evaluation
functions in Informed (Heuristic) Search algorithms such as A*, Greedy Best-First Search,
and IDA*. It provides implementations of Manhattan ($L_1$), Euclidean ($L_2$),
Chebyshev ($L_\\infty$), Octile, and Minkowski distance metrics for 2D coordinate spaces,
complete with input validation, batch comparison, metric space axiom verification,
ASCII grid rendering, and graphical visualization.

Theoretical Background & AI Concepts:
- Heuristic Functions in State-Space Search:
    In informed search, a heuristic function h(n) estimates the cost of the cheapest path
    from state n to a goal state. The choice of distance metric depends fundamentally on
    the topology of the state space and the allowed operator set (action transitions).
- Manhattan Distance (L1 Norm / Taxicab Metric):
    Formula: d_M(p1, p2) = |x1 - x2| + |y1 - y2|
    Named after the grid street layout of Manhattan. It measures the distance traversed
    when movement is strictly constrained to orthogonal (cardinal) directions: Up, Down,
    Left, and Right (4-directional grid).
- Euclidean Distance (L2 Norm / Straight-Line Metric):
    Formula: d_E(p1, p2) = sqrt((x1 - x2)^2 + (y1 - y2)^2)
    Represents the straight-line distance "as the crow flies". It is the natural metric
    for continuous Euclidean space or unconstrained any-angle pathfinding.
- Chebyshev Distance (L_infinity Norm / Chessboard Metric - Optional Challenge):
    Formula: d_C(p1, p2) = max(|x1 - x2|, |y1 - y2|)
    Named after Pafnuty Chebyshev. Represents the minimum number of moves a King requires
    on a chessboard to travel between two squares, allowing 8-directional movement
    (orthogonal and diagonal) where diagonal moves have unit cost (cost = 1).
- Octile Distance:
    Formula: d_O(p1, p2) = (D2 - D1) * min(dx, dy) + D1 * max(dx, dy)
    Used on 8-directional grids where diagonal movement incurs a cost of sqrt(2)
    while orthogonal movement costs 1.
- Admissibility and Dominance:
    - Admissible: A heuristic h(n) is admissible if it never overestimates the true cost
      to reach the goal: 0 <= h(n) <= h*(n) for all n.
    - On a 4-connected orthogonal grid without obstacles:
      Manhattan distance is exact: h_M(n) = h*(n).
      Euclidean distance is admissible but strictly underestimates: h_E(n) <= h_M(n).
      Chebyshev distance also underestimates: h_C(n) <= h_E(n) <= h_M(n).
      Therefore, Manhattan dominates Euclidean on 4-connected grids (h_M(n) >= h_E(n)),
      expanding fewer or equal nodes in A* search while preserving optimality.
    - If Euclidean or Manhattan is misapplied to an 8-connected grid with unit cost,
      Manhattan overestimates (inadmissible!), destroying the optimality guarantee of A*.
- Metric Space Axioms:
    A valid distance metric d(x, y) must satisfy:
    1. Non-negativity: d(x, y) >= 0
    2. Identity of Indiscernibles: d(x, y) = 0 <=> x == y
    3. Symmetry: d(x, y) = d(y, x)
    4. Triangle Inequality: d(x, z) <= d(x, y) + d(y, z)

Complexity:
- Time Complexity: O(1) arithmetic operations for all distance functions.
- Space Complexity: O(1) auxiliary memory.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# Coordinate Type Definitions
Point = Tuple[Union[int, float], Union[int, float]]
PointLike = Union[
    Point,
    Sequence[Union[int, float]],
    "Point2D",
]


class Point2D(NamedTuple):
    """Immutable representation of a 2D Cartesian coordinate point.

    Attributes:
        x: The horizontal coordinate (abscissa).
        y: The vertical coordinate (ordinate).
    """

    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        """Return point as a standard (x, y) float tuple."""
        return (float(self.x), float(self.y))

    def is_integral(self) -> bool:
        """Check if both coordinates represent exact integer values."""
        return float(self.x).is_integer() and float(self.y).is_integer()

    def __str__(self) -> str:
        if self.is_integral():
            return f"({int(round(float(self.x)))}, {int(round(float(self.y)))})"
        return f"({float(self.x):.2f}, {float(self.y):.2f})"


def _coerce_point(p: PointLike) -> Point2D:
    """Validate and convert any 2D point-like input into a Point2D object.

    Args:
        p: A 2-element sequence or Point2D of numeric values (x, y).

    Returns:
        A validated Point2D instance.

    Raises:
        TypeError: If p is not a sequence, or if coordinates cannot be converted to float.
        ValueError: If p does not contain exactly two coordinate elements.
    """
    if isinstance(p, Point2D):
        return Point2D(float(p.x), float(p.y))

    if not isinstance(p, (tuple, list, Sequence)) or isinstance(p, (str, bytes)):
        raise TypeError(f"Point must be a 2-element numeric sequence or tuple, got {type(p).__name__}: {p!r}")

    if len(p) != 2:
        raise ValueError(f"Point must contain exactly 2 coordinates (x, y), got {len(p)} elements: {p!r}")

    try:
        x = float(p[0])
        y = float(p[1])
    except (ValueError, TypeError) as exc:
        raise TypeError(f"Coordinates must be numeric (int or float), got ({p[0]!r}, {p[1]!r})") from exc

    if math.isnan(x) or math.isnan(y):
        raise ValueError(f"Coordinates cannot be NaN: ({x}, {y})")

    if math.isinf(x) or math.isinf(y):
        raise ValueError(f"Coordinates cannot be Infinite: ({x}, {y})")

    return Point2D(x, y)


# =====================================================================
# Core Distance Functions
# =====================================================================


def manhattan(p1: PointLike, p2: PointLike) -> Union[int, float]:
    """Compute the Manhattan distance (L1 norm / Taxicab metric) between two 2D points.

    Formula:
        d_M(p1, p2) = |x1 - x2| + |y1 - y2|

    AI Context:
        Admissible and consistent heuristic for 4-directional (orthogonal) grid search
        where movement is limited to Up, Down, Left, and Right.

    Args:
        p1: Coordinates of first point (x1, y1).
        p2: Coordinates of second point (x2, y2).

    Returns:
        The Manhattan distance as an integer (if both points are integral) or float.

    Complexity:
        Time: O(1)
        Space: O(1)

    Example:
        >>> manhattan((1, 1), (4, 5))
        7
    """
    pt1 = _coerce_point(p1)
    pt2 = _coerce_point(p2)
    dx = abs(pt1.x - pt2.x)
    dy = abs(pt1.y - pt2.y)
    dist = dx + dy
    if pt1.is_integral() and pt2.is_integral():
        return int(round(dist))
    return dist


def euclidean(p1: PointLike, p2: PointLike) -> float:
    """Compute the Euclidean distance (L2 norm / Straight-line metric) between two 2D points.

    Formula:
        d_E(p1, p2) = sqrt((x1 - x2)^2 + (y1 - y2)^2)

    AI Context:
        Admissible heuristic for continuous space, any-angle pathfinding, and straight-line
        lower bounds. Strictly underestimates Manhattan distance on orthogonal grids.

    Args:
        p1: Coordinates of first point (x1, y1).
        p2: Coordinates of second point (x2, y2).

    Returns:
        The Euclidean distance as a float.

    Complexity:
        Time: O(1)
        Space: O(1)

    Example:
        >>> round(euclidean((1, 1), (4, 5)), 2)
        5.0
    """
    pt1 = _coerce_point(p1)
    pt2 = _coerce_point(p2)
    dx = pt1.x - pt2.x
    dy = pt1.y - pt2.y
    return math.hypot(dx, dy)


def chebyshev(p1: PointLike, p2: PointLike) -> Union[int, float]:
    """Compute the Chebyshev distance (L_infinity norm / Chessboard metric) between two points.

    Formula:
        d_C(p1, p2) = max(|x1 - x2|, |y1 - y2|)

    AI Context:
        Admissible heuristic for 8-directional grid search when diagonal movement costs
        the same as orthogonal movement (cost = 1, e.g., King's moves in Chess).

    Args:
        p1: Coordinates of first point (x1, y1).
        p2: Coordinates of second point (x2, y2).

    Returns:
        The Chebyshev distance as an integer (if both points are integral) or float.

    Complexity:
        Time: O(1)
        Space: O(1)

    Example:
        >>> chebyshev((1, 1), (4, 5))
        4
    """
    pt1 = _coerce_point(p1)
    pt2 = _coerce_point(p2)
    dx = abs(pt1.x - pt2.x)
    dy = abs(pt1.y - pt2.y)
    dist = max(dx, dy)
    if pt1.is_integral() and pt2.is_integral():
        return int(round(dist))
    return dist


def octile(
    p1: PointLike,
    p2: PointLike,
    orthogonal_cost: float = 1.0,
    diagonal_cost: Optional[float] = None,
) -> float:
    """Compute the Octile distance for 8-directional movement with standard diagonal costs.

    Formula:
        dx = |x1 - x2|, dy = |y1 - y2|
        d_O = D1 * (max(dx, dy) - min(dx, dy)) + D2 * min(dx, dy)
            = D1 * max(dx, dy) + (D2 - D1) * min(dx, dy)

    AI Context:
        Admissible and exact (without obstacles) heuristic for 8-connected grid navigation
        where orthogonal moves cost 1 and diagonal moves cost sqrt(2) (~1.414).

    Args:
        p1: Coordinates of first point (x1, y1).
        p2: Coordinates of second point (x2, y2).
        orthogonal_cost: Step cost for orthogonal movements (default 1.0).
        diagonal_cost: Step cost for diagonal movements (default sqrt(2) * orthogonal_cost).

    Returns:
        The Octile distance as a float.
    """
    pt1 = _coerce_point(p1)
    pt2 = _coerce_point(p2)
    dx = abs(pt1.x - pt2.x)
    dy = abs(pt1.y - pt2.y)
    d1 = float(orthogonal_cost)
    d2 = math.sqrt(2.0) * d1 if diagonal_cost is None else float(diagonal_cost)
    return d1 * max(dx, dy) + (d2 - d1) * min(dx, dy)


def minkowski(p1: PointLike, p2: PointLike, p: float = 3.0) -> float:
    """Compute the Minkowski distance (L_p norm) between two points.

    Formula:
        d_p(p1, p2) = (|x1 - x2|^p + |y1 - y2|^p)^(1/p)

    AI Context:
        Generalization of Lp metric spaces:
        - p = 1: Manhattan distance
        - p = 2: Euclidean distance
        - p -> infinity: Chebyshev distance

    Args:
        p1: Coordinates of first point (x1, y1).
        p2: Coordinates of second point (x2, y2).
        p: Norm degree parameter (must be >= 1.0 to satisfy triangle inequality).

    Returns:
        The Minkowski distance as a float.

    Raises:
        ValueError: If p < 1.0.
    """
    if p < 1.0:
        raise ValueError(f"Minkowski norm parameter p must be >= 1.0 to satisfy triangle inequality, got {p}")
    pt1 = _coerce_point(p1)
    pt2 = _coerce_point(p2)
    dx = abs(pt1.x - pt2.x)
    dy = abs(pt1.y - pt2.y)
    return (dx**p + dy**p) ** (1.0 / p)


# =====================================================================
# Distance Result & Comparison Dataclass
# =====================================================================


@dataclass(frozen=True)
class DistanceResult:
    """Immutable encapsulation of multiple distance metrics between two points.

    Attributes:
        p1: First point as a Point2D.
        p2: Second point as a Point2D.
        manhattan_dist: Computed Manhattan distance.
        euclidean_dist: Computed Euclidean distance.
        chebyshev_dist: Computed Chebyshev distance.
        octile_dist: Computed Octile distance (D1=1, D2=sqrt(2)).
    """

    p1: Point2D
    p2: Point2D
    manhattan_dist: Union[int, float]
    euclidean_dist: float
    chebyshev_dist: Union[int, float]
    octile_dist: float

    @property
    def dx(self) -> float:
        """Absolute horizontal difference |x1 - x2|."""
        return abs(self.p1.x - self.p2.x)

    @property
    def dy(self) -> float:
        """Absolute vertical difference |y1 - y2|."""
        return abs(self.p1.y - self.p2.y)

    @property
    def difference(self) -> float:
        """Difference between Manhattan and Euclidean distances (Manhattan - Euclidean)."""
        return float(self.manhattan_dist) - self.euclidean_dist

    @property
    def ratio_manhattan_to_euclidean(self) -> float:
        """Ratio of Manhattan to Euclidean distance. Range [1.0, sqrt(2)]."""
        if self.euclidean_dist == 0.0:
            return 1.0
        return float(self.manhattan_dist) / self.euclidean_dist

    def formatted(self, precision: int = 2, include_chebyshev: bool = False) -> str:
        """Return formatted string adhering to canonical sample output requirements.

        Sample Output requirement:
            "Manhattan: 7, Euclidean: 5.00"

        Args:
            precision: Number of decimal places for floating-point values (default 2).
            include_chebyshev: Whether to append Chebyshev distance.

        Returns:
            Human-readable comparison string.
        """
        # Format Manhattan: if integer, display as int; otherwise formatted float
        if isinstance(self.manhattan_dist, int) or float(self.manhattan_dist).is_integer():
            m_str = str(int(self.manhattan_dist))
        else:
            m_str = f"{self.manhattan_dist:.{precision}f}"

        e_str = f"{self.euclidean_dist:.{precision}f}"

        base = f"Manhattan: {m_str}, Euclidean: {e_str}"
        if include_chebyshev:
            if isinstance(self.chebyshev_dist, int) or float(self.chebyshev_dist).is_integer():
                c_str = str(int(self.chebyshev_dist))
            else:
                c_str = f"{self.chebyshev_dist:.{precision}f}"
            base += f", Chebyshev: {c_str}"
        return base

    def formatted_fixed(self, precision: int = 2) -> str:
        """Return formatted string where both distances strictly adhere to fixed decimal places.

        Example:
            "Manhattan: 7.00, Euclidean: 5.00"
        """
        return f"Manhattan: {float(self.manhattan_dist):.{precision}f}, Euclidean: {self.euclidean_dist:.{precision}f}"

    def as_dict(self) -> Dict[str, Any]:
        """Convert result fields to a JSON-serializable dictionary."""
        return {
            "p1": [self.p1.x, self.p1.y],
            "p2": [self.p2.x, self.p2.y],
            "dx": self.dx,
            "dy": self.dy,
            "manhattan": self.manhattan_dist,
            "euclidean": round(self.euclidean_dist, 6),
            "chebyshev": self.chebyshev_dist,
            "octile": round(self.octile_dist, 6),
            "ratio_manhattan_euclidean": round(self.ratio_manhattan_to_euclidean, 4),
        }


def compute_distances(p1: PointLike, p2: PointLike) -> DistanceResult:
    """Compute all primary distance metrics between two points.

    Args:
        p1: First point coordinates (x1, y1).
        p2: Second point coordinates (x2, y2).

    Returns:
        A DistanceResult instance containing Manhattan, Euclidean, Chebyshev, and Octile distances.
    """
    pt1 = _coerce_point(p1)
    pt2 = _coerce_point(p2)
    m = manhattan(pt1, pt2)
    e = euclidean(pt1, pt2)
    c = chebyshev(pt1, pt2)
    o = octile(pt1, pt2)
    return DistanceResult(
        p1=pt1,
        p2=pt2,
        manhattan_dist=m,
        euclidean_dist=e,
        chebyshev_dist=c,
        octile_dist=o,
    )


def compare_point_pairs(
    pairs: Sequence[Tuple[PointLike, PointLike]],
) -> List[DistanceResult]:
    """Process multiple point pairs and return calculated distance metrics for each pair.

    Args:
        pairs: A sequence of (p1, p2) coordinate pairs.

    Returns:
        A list of DistanceResult instances.
    """
    results: List[DistanceResult] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Each element in pairs must be a tuple of two points (p1, p2), got: {pair!r}")
        results.append(compute_distances(pair[0], pair[1]))
    return results


def format_comparison_table(
    results: Sequence[DistanceResult],
    include_chebyshev: bool = True,
    include_octile: bool = False,
    precision: int = 2,
) -> str:
    """Format multiple DistanceResult records into an aligned ASCII comparison table.

    Args:
        results: Sequence of DistanceResult records.
        include_chebyshev: Whether to include Chebyshev column.
        include_octile: Whether to include Octile column.
        precision: Decimal formatting precision.

    Returns:
        Formatted ASCII table string.
    """
    headers = ["Pair #", "Point 1", "Point 2", "dx", "dy", "Manhattan", "Euclidean"]
    if include_chebyshev:
        headers.append("Chebyshev")
    if include_octile:
        headers.append("Octile")
    headers.append("Ratio (M/E)")

    rows: List[List[str]] = []
    for idx, res in enumerate(results, start=1):
        row = [
            str(idx),
            str(res.p1),
            str(res.p2),
            f"{res.dx:.{precision}f}".rstrip("0").rstrip(".") if not res.dx.is_integer() else str(int(res.dx)),
            f"{res.dy:.{precision}f}".rstrip("0").rstrip(".") if not res.dy.is_integer() else str(int(res.dy)),
            f"{float(res.manhattan_dist):.{precision}f}",
            f"{res.euclidean_dist:.{precision}f}",
        ]
        if include_chebyshev:
            row.append(f"{float(res.chebyshev_dist):.{precision}f}")
        if include_octile:
            row.append(f"{res.octile_dist:.{precision}f}")
        row.append(f"{res.ratio_manhattan_to_euclidean:.3f}")
        rows.append(row)

    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]

    sep_line = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_line = "| " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"

    body_lines: List[str] = []
    for row in rows:
        formatted_row = "| " + " | ".join(f"{val:<{col_widths[i]}}" for i, val in enumerate(row)) + " |"
        body_lines.append(formatted_row)

    table = [sep_line, header_line, sep_line] + body_lines + [sep_line]
    return "\n".join(table)


# =====================================================================
# Parsing Helpers for CLI and User Input
# =====================================================================


def parse_point(point_str: str) -> Point2D:
    """Parse a coordinate string into a Point2D object.

    Supports formats like:
        "(1, 1)", "1, 1", "1,1", "(1.5, 4.2)", "1 1"

    Args:
        point_str: Raw string representing coordinate pair.

    Returns:
        Point2D instance.

    Raises:
        ValueError: If the string cannot be parsed into 2 numeric coordinates.
    """
    clean = point_str.strip().strip("()[]{}")
    parts = re.split(r"[,;\s]+", clean)
    parts = [p for p in parts if p]
    if len(parts) != 2:
        raise ValueError(f"Could not parse 2D point from string: {point_str!r}. Expected format '(x, y)'.")
    try:
        x = float(parts[0])
        y = float(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid numeric values in point string: {point_str!r}") from exc
    return Point2D(x, y)


def parse_point_pairs(pairs_str: str) -> List[Tuple[Point2D, Point2D]]:
    """Parse a string representing one or more point pairs.

    Supports formats:
        "(1, 1), (4, 5)"
        "(1, 1) -> (4, 5); (0, 0) -> (3, 4)"
        "[(1, 1), (4, 5)], [(2, 2), (6, 8)]"

    Args:
        pairs_str: String containing pairs.

    Returns:
        List of (p1, p2) tuples.
    """
    # Check for arrow or semicolon delimiters
    if ";" in pairs_str:
        pair_tokens = pairs_str.split(";")
    else:
        # Match pattern of form (x1, y1), (x2, y2)
        # Find all coordinate tuples (x, y)
        coord_matches = re.findall(r"\(\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*\)", pairs_str)
        if len(coord_matches) >= 2 and len(coord_matches) % 2 == 0:
            results: List[Tuple[Point2D, Point2D]] = []
            for i in range(0, len(coord_matches), 2):
                results.append((parse_point(coord_matches[i]), parse_point(coord_matches[i + 1])))
            return results
        pair_tokens = [pairs_str]

    results = []
    for token in pair_tokens:
        token = token.strip()
        if not token:
            continue
        # Split by '->' or ',' between tuples
        if "->" in token:
            sub = token.split("->")
        else:
            # Match tuples
            sub = re.findall(r"\([^)]*\)", token)
            if len(sub) != 2:
                # Fallback: split by comma if only 4 numbers
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", token)
                if len(nums) == 4:
                    sub = [f"({nums[0]}, {nums[1]})", f"({nums[2]}, {nums[3]})"]
                else:
                    raise ValueError(f"Cannot parse point pair from segment: {token!r}")
        results.append((parse_point(sub[0]), parse_point(sub[1])))

    return results


# =====================================================================
# Metric Space Axiom Verification & Heuristic Properties
# =====================================================================


def verify_metric_space_axioms(
    p1: PointLike,
    p2: PointLike,
    p3: PointLike,
    metric_func: Callable[[PointLike, PointLike], Union[int, float]],
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """Verify that a given distance function satisfies the 4 metric space axioms.

    Axioms:
    1. Non-negativity: d(p1, p2) >= 0
    2. Identity of Indiscernibles: d(p1, p2) == 0 <=> p1 == p2
    3. Symmetry: d(p1, p2) == d(p2, p1)
    4. Triangle Inequality: d(p1, p3) <= d(p1, p2) + d(p2, p3)

    Args:
        p1: First test point.
        p2: Second test point.
        p3: Third test point (for triangle inequality).
        metric_func: Distance function under test.
        tolerance: Numerical comparison tolerance.

    Returns:
        Dictionary reporting verification status for each axiom.
    """
    pt1 = _coerce_point(p1)
    pt2 = _coerce_point(p2)
    pt3 = _coerce_point(p3)

    d12 = float(metric_func(pt1, pt2))
    d21 = float(metric_func(pt2, pt1))
    d11 = float(metric_func(pt1, pt1))
    d23 = float(metric_func(pt2, pt3))
    d13 = float(metric_func(pt1, pt3))

    # 1. Non-negativity
    non_negative = d12 >= -tolerance and d23 >= -tolerance and d13 >= -tolerance

    # 2. Identity
    identity = abs(d11) <= tolerance

    # 3. Symmetry
    symmetry = abs(d12 - d21) <= tolerance

    # 4. Triangle inequality
    triangle = d13 <= (d12 + d23 + tolerance)

    all_satisfied = non_negative and identity and symmetry and triangle

    return {
        "metric_name": metric_func.__name__,
        "all_satisfied": all_satisfied,
        "non_negativity": {
            "satisfied": non_negative,
            "d(p1, p2)": d12,
        },
        "identity_of_indiscernibles": {
            "satisfied": identity,
            "d(p1, p1)": d11,
        },
        "symmetry": {
            "satisfied": symmetry,
            "d(p1, p2)": d12,
            "d(p2, p1)": d21,
            "abs_difference": abs(d12 - d21),
        },
        "triangle_inequality": {
            "satisfied": triangle,
            "d(p1, p3)": d13,
            "d(p1, p2) + d(p2, p3)": d12 + d23,
            "slack": (d12 + d23) - d13,
        },
    }


def compare_heuristics_for_grid_topology(
    p1: PointLike,
    p2: PointLike,
    topology: str = "orthogonal",
) -> Dict[str, Any]:
    """Analyze heuristic admissibility and dominance under different movement topologies.

    Topologies supported:
    - 'orthogonal' (4-way): Only Up, Down, Left, Right moves allowed (cost 1).
    - 'octile' (8-way-weighted): Orthogonal cost 1, Diagonal cost sqrt(2).
    - 'chebyshev' (8-way-uniform): King moves, all 8 directions cost 1.
    - 'euclidean' (continuous): Straight-line movement in any continuous direction.

    Args:
        p1: Start point.
        p2: Goal point.
        topology: Movement topology name.

    Returns:
        Analysis report comparing admissibility, true cost h*(n), and dominance.
    """
    pt1 = _coerce_point(p1)
    pt2 = _coerce_point(p2)

    m = float(manhattan(pt1, pt2))
    e = euclidean(pt1, pt2)
    c = float(chebyshev(pt1, pt2))
    o = octile(pt1, pt2)

    topology_lower = topology.lower().strip()
    if topology_lower in ("orthogonal", "4-way", "cardinal", "manhattan"):
        h_star = m
        admissible_heuristics = {
            "Manhattan": {"cost": m, "is_admissible": True, "is_exact": True, "quality": "Optimal & Dominant (Exact h*)"},
            "Euclidean": {"cost": e, "is_admissible": True, "is_exact": math.isclose(m, e), "quality": "Admissible (Underestimates h*)"},
            "Chebyshev": {"cost": c, "is_admissible": True, "is_exact": math.isclose(m, c), "quality": "Admissible (Severe underestimation)"},
        }
    elif topology_lower in ("octile", "8-way-weighted", "diagonal-weighted"):
        h_star = o
        admissible_heuristics = {
            "Manhattan": {"cost": m, "is_admissible": math.isclose(m, o) or m <= o, "is_exact": False, "quality": "INADMISSIBLE (Overestimates h*!)" if m > o + 1e-6 else "Admissible"},
            "Euclidean": {"cost": e, "is_admissible": True, "is_exact": math.isclose(e, o), "quality": "Admissible (Underestimates h*)"},
            "Chebyshev": {"cost": c, "is_admissible": True, "is_exact": math.isclose(c, o), "quality": "Admissible (Underestimates h*)"},
            "Octile": {"cost": o, "is_admissible": True, "is_exact": True, "quality": "Optimal & Dominant (Exact h*)"},
        }
    elif topology_lower in ("chebyshev", "8-way-uniform", "king", "chessboard"):
        h_star = c
        admissible_heuristics = {
            "Manhattan": {"cost": m, "is_admissible": math.isclose(m, c) or m <= c, "is_exact": False, "quality": "INADMISSIBLE (Overestimates h*!)" if m > c + 1e-6 else "Admissible"},
            "Euclidean": {"cost": e, "is_admissible": math.isclose(e, c) or e <= c, "is_exact": False, "quality": "INADMISSIBLE (Overestimates h*!)" if e > c + 1e-6 else "Admissible"},
            "Chebyshev": {"cost": c, "is_admissible": True, "is_exact": True, "quality": "Optimal & Dominant (Exact h*)"},
        }
    else:  # Continuous Euclidean
        h_star = e
        admissible_heuristics = {
            "Manhattan": {"cost": m, "is_admissible": math.isclose(m, e) or m <= e, "is_exact": False, "quality": "INADMISSIBLE (Overestimates h*!)" if m > e + 1e-6 else "Admissible"},
            "Euclidean": {"cost": e, "is_admissible": True, "is_exact": True, "quality": "Optimal & Dominant (Exact h*)"},
            "Chebyshev": {"cost": c, "is_admissible": True, "is_exact": math.isclose(c, e), "quality": "Admissible (Underestimates h*)"},
        }

    return {
        "p1": [pt1.x, pt1.y],
        "p2": [pt2.x, pt2.y],
        "topology": topology,
        "true_shortest_cost_without_obstacles": h_star,
        "evaluations": admissible_heuristics,
    }


# =====================================================================
# ASCII 2D Grid Visualizer
# =====================================================================


def render_ascii_distance_grid(
    p1: PointLike,
    p2: PointLike,
    padding: int = 1,
) -> str:
    """Render an ASCII 2D grid demonstrating Manhattan and Euclidean paths between two points.

    Legend:
        'S': Start point (p1)
        'G': Goal point (p2)
        '#': Manhattan taxicab orthogonal path
        '*': Euclidean direct line trajectory
        '+': Overlap of Manhattan and Euclidean trajectories
        '.': Empty grid cell

    Args:
        p1: First point.
        p2: Second point.
        padding: Border cell padding around bounding box.

    Returns:
        Formatted ASCII grid visualization string.
    """
    pt1 = _coerce_point(p1)
    pt2 = _coerce_point(p2)

    x1, y1 = int(round(pt1.x)), int(round(pt1.y))
    x2, y2 = int(round(pt2.x)), int(round(pt2.y))

    min_x = min(x1, x2) - padding
    max_x = max(x1, x2) + padding
    min_y = min(y1, y2) - padding
    max_y = max(y1, y2) + padding

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # Grid indexed by [row][col] where row is inverted y (top row = max_y)
    grid = [["." for _ in range(width)] for _ in range(height)]

    def to_rc(x: int, y: int) -> Tuple[int, int]:
        r = max_y - y
        c = x - min_x
        return r, c

    # 1. Plot Manhattan path: horizontal first then vertical (x1, y1) -> (x2, y1) -> (x2, y2)
    step_x = 1 if x2 >= x1 else -1
    for cur_x in range(x1, x2 + step_x, step_x):
        r, c = to_rc(cur_x, y1)
        grid[r][c] = "#"

    step_y = 1 if y2 >= y1 else -1
    for cur_y in range(y1, y2 + step_y, step_y):
        r, c = to_rc(x2, cur_y)
        grid[r][c] = "#"

    # 2. Plot Euclidean line using Bresenham algorithm
    bx1, by1 = x1, y1
    bx2, by2 = x2, y2
    dx = abs(bx2 - bx1)
    dy = -abs(by2 - by1)
    sx = 1 if bx1 < bx2 else -1
    sy = 1 if by1 < by2 else -1
    err = dx + dy

    while True:
        r, c = to_rc(bx1, by1)
        if grid[r][c] == "#":
            grid[r][c] = "+"
        else:
            grid[r][c] = "*"

        if bx1 == bx2 and by1 == by2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            bx1 += sx
        if e2 <= dx:
            err += dx
            by1 += sy

    # 3. Mark Start and Goal clearly
    r1, c1 = to_rc(x1, y1)
    r2, c2 = to_rc(x2, y2)
    grid[r1][c1] = "S"
    grid[r2][c2] = "G"

    # Assemble lines with Y-axis coordinate annotations
    lines = []
    lines.append(f"2D Grid Trajectory Visualization: P1={pt1} -> P2={pt2}")
    lines.append(f"Manhattan Path (#): {manhattan(pt1, pt2)} | Euclidean Line (*): {euclidean(pt1, pt2):.2f}")
    lines.append("+" + "-" * (width * 2 + 3) + "+")

    for y in range(max_y, min_y - 1, -1):
        r = max_y - y
        row_str = " ".join(grid[r])
        lines.append(f"| {y:2d} | {row_str} |")

    lines.append("+" + "-" * (width * 2 + 3) + "+")
    x_header = " ".join(f"{x % 10}" for x in range(min_x, max_x + 1))
    lines.append(f"     X: {x_header}")
    lines.append("Legend: S=Start(P1), G=Goal(P2), #=Manhattan(L1), *=Euclidean(L2), +=Overlap")
    return "\n".join(lines)


# =====================================================================
# Graphical Visualization (Matplotlib)
# =====================================================================


def plot_distance_metrics(
    pairs: Sequence[Tuple[PointLike, PointLike]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    title: str = "Comparison of Distance Metrics (Manhattan, Euclidean, Chebyshev)",
) -> Any:
    """Generate a matplotlib figure visualizing Manhattan, Euclidean, and Chebyshev metrics.

    Includes:
    1. Grid trajectory plots for each point pair with L1 orthogonal steps, L2 line, and L_inf box.
    2. Geometric unit balls / isocontours for L1 (Diamond), L2 (Circle), and L_inf (Square).

    Args:
        pairs: Sequence of (p1, p2) coordinate pairs.
        save_path: Optional filesystem path to save figure image.
        show: Whether to call plt.show() (disabled automatically in headless environments).
        title: Main title of the visualization figure.

    Returns:
        The created matplotlib Figure object, or None if matplotlib is not available.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle
    except ImportError:
        return None

    # Handle headless environments safely
    import os
    if os.environ.get("MPLBACKEND") == "Agg" or not os.environ.get("DISPLAY", "1"):
        matplotlib.use("Agg")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Subplot 1: Coordinate Pairs Trajectories
    ax1.set_title("Coordinate Pairs & Trajectories", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")

    colors = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED"]

    all_x: List[float] = []
    all_y: List[float] = []

    for idx, (raw_p1, raw_p2) in enumerate(pairs):
        p1 = _coerce_point(raw_p1)
        p2 = _coerce_point(raw_p2)
        color = colors[idx % len(colors)]

        all_x.extend([p1.x, p2.x])
        all_y.extend([p1.y, p2.y])

        # Euclidean direct line
        ax1.plot(
            [p1.x, p2.x],
            [p1.y, p2.y],
            color=color,
            linestyle="-",
            linewidth=2.0,
            label=f"Pair {idx+1} Euclidean: {euclidean(p1, p2):.2f}",
        )

        # Manhattan orthogonal path
        corner_x, corner_y = p2.x, p1.y
        ax1.plot(
            [p1.x, corner_x, p2.x],
            [p1.y, corner_y, p2.y],
            color=color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.7,
            label=f"Pair {idx+1} Manhattan: {manhattan(p1, p2)}",
        )

        # Start and Goal markers
        ax1.scatter([p1.x], [p1.y], color=color, s=80, marker="o", edgecolors="black", zorder=5)
        ax1.scatter([p2.x], [p2.y], color=color, s=100, marker="*", edgecolors="black", zorder=5)
        ax1.text(p1.x + 0.1, p1.y + 0.1, f"P1_{idx+1}{p1}", fontsize=9, fontweight="bold")
        ax1.text(p2.x + 0.1, p2.y + 0.1, f"P2_{idx+1}{p2}", fontsize=9, fontweight="bold")

    if all_x and all_y:
        pad = 1.5
        ax1.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax1.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax1.legend(loc="upper left", fontsize=8)

    # Subplot 2: Unit Ball Isocontours (L1, L2, L_inf)
    ax2.set_title("Unit Norm Isocontours (Metric Topology)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_xlabel("Delta X")
    ax2.set_ylabel("Delta Y")
    ax2.set_aspect("equal", "box")

    # L1 Unit Ball (Manhattan): Diamond |x| + |y| = 1
    l1_diamond = Polygon([[1, 0], [0, 1], [-1, 0], [0, -1]], closed=True, fill=False, edgecolor="#DC2626", linewidth=2.5, linestyle="--", label="L1 (Manhattan) |dx|+|dy|=1")
    ax2.add_patch(l1_diamond)

    # L2 Unit Ball (Euclidean): Circle dx^2 + dy^2 = 1
    theta = [2 * math.pi * i / 100 for i in range(101)]
    circle_x = [math.cos(t) for t in theta]
    circle_y = [math.sin(t) for t in theta]
    ax2.plot(circle_x, circle_y, color="#2563EB", linewidth=2.5, label="L2 (Euclidean) sqrt(dx²+dy²)=1")

    # L_inf Unit Ball (Chebyshev): Square max(|dx|, |dy|) = 1
    l_inf_square = Rectangle((-1, -1), 2, 2, fill=False, edgecolor="#059669", linewidth=2.5, linestyle=":", label="L∞ (Chebyshev) max(|dx|,|dy|)=1")
    ax2.add_patch(l_inf_square)

    ax2.set_xlim(-1.6, 1.6)
    ax2.set_ylim(-1.6, 1.6)
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax2.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    if show:
        try:
            plt.show()
        except Exception:
            pass

    return fig


# =====================================================================
# Comprehensive Viva Voce Reference
# =====================================================================


def get_distance_viva_reference() -> Dict[str, Dict[str, str]]:
    """Return an authoritative academic question-and-answer reference for University Viva Voce exams.

    Topics covered:
    1. Overestimation vs. Underestimation in Heuristics
    2. When to use Manhattan vs. Euclidean vs. Chebyshev
    3. Definition of an Admissible Heuristic
    4. Consistency (Monotonicity) vs. Admissibility
    5. Heuristic Dominance and Pruning Efficiency
    6. Metric Space Axioms & Mathematical Proofs
    """
    return {
        "overestimation_vs_underestimation": {
            "question": "What is the consequence of heuristic overestimation vs. underestimation in A* search?",
            "answer": (
                "Underestimation (h(n) <= h*(n)) means the heuristic is optimistic: it never estimates "
                "a cost greater than the true minimal cost to the goal. Underestimation guarantees ADMISSIBILITY, "
                "which mathematically ensures that A* tree search (and graph search with consistent heuristics) "
                "always finds an optimal (lowest-cost) path.\n\n"
                "Overestimation (h(n) > h*(n)) means the heuristic is pessimistic / inadmissible. An overestimating "
                "heuristic may assign an artificially inflated f-cost (f(n) = g(n) + h(n)) to the optimal path, "
                "causing A* to explore and terminate along a suboptimal branch first. Consequently, optimality is "
                "permanently sacrificed, though search speed may increase (acting like weighted A* or greedy search)."
            ),
            "key_takeaway": "Underestimation preserves optimality; Overestimation risks suboptimal solutions.",
        },
        "when_to_use_manhattan_vs_euclidean": {
            "question": "When should an AI agent use Manhattan distance vs. Euclidean distance vs. Chebyshev distance?",
            "answer": (
                "1. Manhattan Distance (L1 norm): Use when the agent is restricted to 4-directional ORTHOGONAL "
                "movement (Up, Down, Left, Right), such as street grid navigation, 15-puzzle / 8-puzzle sliding tiles, "
                "or standard grid mazes without diagonal transitions. On 4-connected grids without obstacles, "
                "Manhattan distance is the exact cost (h(n) = h*(n)) and dominates Euclidean distance.\n\n"
                "2. Euclidean Distance (L2 norm): Use when the agent can move in ANY angle or in continuous 2D/3D "
                "space (e.g., aerial drones, autonomous vehicles, robotic arm kinematic path planning). On orthogonal "
                "grids, Euclidean distance is admissible but underestimates more than Manhattan (h_E <= h_M).\n\n"
                "3. Chebyshev Distance (L_infinity norm): Use when the agent can move in 8 directions (orthogonal + diagonal) "
                "and the cost of a diagonal move equals the cost of an orthogonal move (cost = 1, e.g., King's moves in Chess).\n\n"
                "4. Octile Distance: Use when moving in 8 directions, but diagonal moves cost sqrt(2) (~1.414) while "
                "orthogonal moves cost 1."
            ),
            "rule_of_thumb": (
                "Match the metric to allowed agent movements: "
                "4 directions -> Manhattan; "
                "8 directions (uniform) -> Chebyshev; "
                "8 directions (diagonal=sqrt(2)) -> Octile; "
                "Continuous / Any angle -> Euclidean."
            ),
        },
        "admissible_heuristic_definition": {
            "question": "What is the formal mathematical definition of an admissible heuristic?",
            "answer": (
                "A heuristic function h(n) is admissible if for every state n in the state space:\n"
                "    0 <= h(n) <= h*(n)\n"
                "where h*(n) denotes the true minimum path cost from state n to the closest goal state, "
                "and h(goal) = 0 for all goal states.\n\n"
                "Admissibility asserts that h(n) is an optimistic lower bound. It ensures that the goal node on "
                "the optimal path will never look worse than a suboptimal goal node, guaranteeing A* finds the optimal solution."
            ),
            "admissibility_condition": "0 <= h(n) <= h*(n) for all n, with h(goal) = 0.",
        },
        "consistency_monotonicity": {
            "question": "What is a consistent (monotonic) heuristic, and how does it relate to admissibility?",
            "answer": (
                "A heuristic h(n) is consistent (or monotonic) if for every node n and every successor n' generated "
                "by action a with step cost c(n, a, n'):\n"
                "    h(n) <= c(n, a, n') + h(n')\n"
                "This is the triangle inequality applied to heuristics.\n\n"
                "Important Relationships:\n"
                "- Consistency strictly implies Admissibility (every consistent heuristic is admissible).\n"
                "- If h(n) is consistent, the f-values along any path are monotonically non-decreasing (f(n') >= f(n)).\n"
                "- With a consistent heuristic, A* graph search never needs to reopen closed nodes, ensuring optimal "
                "O(V) node expansions."
            ),
            "triangle_inequality": "h(n) <= c(n, a, n') + h(n')",
        },
        "heuristic_dominance": {
            "question": "What is heuristic dominance, and why does a dominant admissible heuristic expand fewer nodes?",
            "answer": (
                "Given two admissible heuristics h1(n) and h2(n), h2 dominates h1 if:\n"
                "    h2(n) >= h1(n) for all states n.\n\n"
                "Because h2 provides a tighter lower bound closer to the true cost h*(n), it provides more accurate "
                "guidance. In A* search, every node with f(n) = g(n) + h(n) < C* must be expanded. Since h2(n) >= h1(n), "
                "f_2(n) >= f_1(n), meaning fewer nodes satisfy f(n) < C*. Thus, A* using h2 will never expand more nodes "
                "than A* using h1 (except possibly some nodes on the boundary where f(n) == C*)."
            ),
            "dominance_rule": "h2 dominates h1 if h2(n) >= h1(n) for all n. Dominant heuristics prune more search space.",
        },
        "metric_space_axioms": {
            "question": "What four mathematical axioms must a function satisfy to be a valid distance metric?",
            "answer": (
                "A distance function d: X x X -> R on a set X is a metric if and only if for all x, y, z in X:\n"
                "1. Non-negativity: d(x, y) >= 0\n"
                "2. Identity of Indiscernibles: d(x, y) = 0 <=> x == y\n"
                "3. Symmetry: d(x, y) = d(y, x)\n"
                "4. Triangle Inequality: d(x, z) <= d(x, y) + d(y, z)\n\n"
                "Manhattan, Euclidean, Chebyshev, and Minkowski (for p >= 1) all strictly satisfy all 4 axioms."
            ),
            "axioms": "Non-negativity, Identity of Indiscernibles, Symmetry, Triangle Inequality.",
        },
    }
