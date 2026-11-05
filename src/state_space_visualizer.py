"""State-Space Visualizer for Reflex Vacuum Agent.

This module models, analyzes, and visualizes the complete state space of a
vacuum cleaner agent operating in a two-room environment (Room A and Room B),
modeled after Russell & Norvig's 'Artificial Intelligence: A Modern Approach' (AIMA).

Theoretical Background & State Space Formulation:
- State Representation:
    Each state is formally represented as an immutable 3-tuple:
        State = (AgentLoc, RoomA_Status, RoomB_Status)
    Where:
        - AgentLoc in {'A', 'B'} (2 possible physical locations)
        - RoomA_Status in {'Clean', 'Dirty'} (2 possible cleanliness statuses)
        - RoomB_Status in {'Clean', 'Dirty'} (2 possible cleanliness statuses)
- State Space Size Calculation:
    By the combinatorial product rule:
        |S| = |Locations| * |Cleanliness|^|Locations|
        |S| = 2 * 2^2 = 8 states.
    For an n-room environment with k possible status levels, |S| = n * k^n.
- Actions:
    Valid actions are A = {'Left', 'Right', 'Suck'}.
- Transitions:
    The transition model Result(s, a) maps each (state, action) pair to a successor state.
    With 8 states and 3 actions, there are exactly 8 * 3 = 24 state transitions.
- Cyclic Transitions:
    A transition is cyclic (self-loop) if Result(s, a) == s.
    For example:
        - In Room A, executing 'Left' leaves location and dirt unchanged -> Self-loop.
        - In Room B, executing 'Right' leaves location and dirt unchanged -> Self-loop.
        - In Room A with Room A already Clean, 'Suck' produces no change -> Self-loop.
        - In Room B with Room B already Clean, 'Suck' produces no change -> Self-loop.
    Exactly 12 of the 24 transitions are self-loops; the other 12 are state-changing.
- Goal States:
    A goal state is any state where both rooms are clean:
        Goal = {(A, Clean, Clean), (B, Clean, Clean)} (States 7 and 8).
    These are visually highlighted in green as part of the optional challenge.

Complexity:
- Time Complexity: O(|S|^2) for state generation and transition mapping.
- Space Complexity: O(|S|^2) for graph adjacency and transition matrices.
"""

from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import matplotlib
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import networkx as nx

# Standard domain constants
LOCATION_A = "A"
LOCATION_B = "B"
VALID_LOCATIONS: Tuple[str, ...] = (LOCATION_A, LOCATION_B)

CLEAN = "Clean"
DIRTY = "Dirty"
VALID_STATUSES: Tuple[str, ...] = (CLEAN, DIRTY)

SYMBOL_CLEAN = "C"
SYMBOL_DIRTY = "D"

ACTION_LEFT = "Left"
ACTION_RIGHT = "Right"
ACTION_SUCK = "Suck"
VALID_ACTIONS: Tuple[str, ...] = (ACTION_LEFT, ACTION_RIGHT, ACTION_SUCK)

# Visual styling constants
COLOR_GOAL = "#2ecc71"         # Vibrant Emerald Green for Goal States (Optional Challenge)
COLOR_NON_GOAL = "#5dade2"     # Soft Blue for Non-Goal States
COLOR_BORDER_GOAL = "#196f3d"  # Dark green border
COLOR_BORDER_NON_GOAL = "#2874a6"

ACTION_COLORS: Dict[str, str] = {
    ACTION_LEFT: "#2980b9",    # Blue for Left
    ACTION_RIGHT: "#e67e22",   # Orange for Right
    ACTION_SUCK: "#8e44ad",    # Purple for Suck
}

ValidLocation = Literal["A", "B"]
ValidStatus = Literal["Clean", "Dirty"]
ValidAction = Literal["Left", "Right", "Suck"]


@dataclass(frozen=True, order=True)
class VacuumState:
    """Immutable representation of a discrete state in the 2-room vacuum world.

    Attributes:
        agent_loc: Current physical room where the agent resides ('A' or 'B').
        room_a_status: Cleanliness status of Room A ('Clean' or 'Dirty').
        room_b_status: Cleanliness status of Room B ('Clean' or 'Dirty').
        state_id: Optional canonical 1-based index (1 to 8) matching AIMA conventions.
    """

    agent_loc: str
    room_a_status: str
    room_b_status: str
    state_id: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate and normalize state attributes upon instantiation."""
        loc = self.agent_loc.strip().upper()
        if loc not in VALID_LOCATIONS:
            raise ValueError(
                f"Invalid agent_loc '{self.agent_loc}'. Must be one of {VALID_LOCATIONS}."
            )
        status_a = self.room_a_status.strip().capitalize()
        if status_a not in VALID_STATUSES:
            raise ValueError(
                f"Invalid room_a_status '{self.room_a_status}'. Must be one of {VALID_STATUSES}."
            )
        status_b = self.room_b_status.strip().capitalize()
        if status_b not in VALID_STATUSES:
            raise ValueError(
                f"Invalid room_b_status '{self.room_b_status}'. Must be one of {VALID_STATUSES}."
            )

        # Bypass frozen constraint to assign normalized strings if needed
        if (
            loc != self.agent_loc
            or status_a != self.room_a_status
            or status_b != self.room_b_status
        ):
            object.__setattr__(self, "agent_loc", loc)
            object.__setattr__(self, "room_a_status", status_a)
            object.__setattr__(self, "room_b_status", status_b)

    @property
    def is_goal(self) -> bool:
        """Check whether both Room A and Room B are clean."""
        return self.room_a_status == CLEAN and self.room_b_status == CLEAN

    @property
    def is_dirty_at_current_loc(self) -> bool:
        """Check whether the room the agent is currently occupying is dirty."""
        if self.agent_loc == LOCATION_A:
            return self.room_a_status == DIRTY
        return self.room_b_status == DIRTY

    @property
    def short_label(self) -> str:
        """Return a compact alphanumeric identifier, e.g., 'S1: (A, D, D)'."""
        prefix = f"S{self.state_id}: " if self.state_id is not None else ""
        a_code = SYMBOL_CLEAN if self.room_a_status == CLEAN else SYMBOL_DIRTY
        b_code = SYMBOL_CLEAN if self.room_b_status == CLEAN else SYMBOL_DIRTY
        return f"{prefix}({self.agent_loc}, {a_code}, {b_code})"

    @property
    def multiline_label(self) -> str:
        """Return a multiline formatted label suitable for node visualization."""
        header = f"State {self.state_id}" if self.state_id is not None else "State"
        a_code = "Clean" if self.room_a_status == CLEAN else "Dirty"
        b_code = "Clean" if self.room_b_status == CLEAN else "Dirty"
        goal_tag = "\n[GOAL]" if self.is_goal else ""
        return f"{header}\nLoc: {self.agent_loc}\nA:{a_code}|B:{b_code}{goal_tag}"

    @property
    def tuple(self) -> Tuple[str, str, str]:
        """Return raw 3-tuple (agent_loc, room_a_status, room_b_status)."""
        return (self.agent_loc, self.room_a_status, self.room_b_status)

    @classmethod
    def from_tuple(
        cls,
        t: Tuple[str, str, str],
        state_id: Optional[int] = None,
    ) -> "VacuumState":
        """Instantiate VacuumState from a 3-tuple."""
        if len(t) != 3:
            raise ValueError(f"State tuple must have exactly 3 elements, got {len(t)}: {t}")
        return cls(agent_loc=t[0], room_a_status=t[1], room_b_status=t[2], state_id=state_id)


@dataclass(frozen=True)
class TransitionRecord:
    """Snapshot of a state transition under a chosen action.

    Attributes:
        source: Origin state before the action.
        action: Actuator command applied ('Left', 'Right', or 'Suck').
        target: Successor state resulting from the transition model Result(s, a).
        is_cyclic: Whether source == target (self-loop).
        description: Explanatory text of the transition dynamics.
    """

    source: VacuumState
    action: str
    target: VacuumState
    is_cyclic: bool
    description: str


class StateSpace:
    """Mathematical state space model for the 2-room vacuum cleaner environment.

    Generates all 8 combinatorial states, links them deterministically through
    valid actions (Left, Right, Suck), detects cyclic transitions (self-loops),
    and computes analytical properties (shortest paths, adjacency matrices).
    """

    def __init__(
        self,
        locations: Sequence[str] = VALID_LOCATIONS,
        statuses: Sequence[str] = VALID_STATUSES,
    ) -> None:
        """Initialize and generate the state space and all transitions.

        Args:
            locations: Sequence of valid room identifiers (defaults to ('A', 'B')).
            statuses: Sequence of cleanliness statuses (defaults to ('Clean', 'Dirty')).
        """
        self.locations: Tuple[str, ...] = tuple(locations)
        self.statuses: Tuple[str, ...] = tuple(statuses)
        self.states: List[VacuumState] = self._generate_states()
        self.state_map: Dict[Tuple[str, str, str], VacuumState] = {
            s.tuple: s for s in self.states
        }
        self.transitions: List[TransitionRecord] = self._generate_transitions()
        self._graph: Optional[nx.DiGraph] = None
        self._multi_graph: Optional[nx.MultiDiGraph] = None

    def _generate_states(self) -> List[VacuumState]:
        """Generate all 8 states combinatorially using Cartesian product.

        Canonical AIMA textbook ordering:
            1. (A, Dirty, Dirty)
            2. (B, Dirty, Dirty)
            3. (A, Clean, Dirty)
            4. (B, Clean, Dirty)
            5. (A, Dirty, Clean)
            6. (B, Dirty, Clean)
            7. (A, Clean, Clean) [Goal]
            8. (B, Clean, Clean) [Goal]

        Returns:
            List of 8 uniquely indexed VacuumState instances.
        """
        # Structured canonical status combinations (Dirty-first descending to Clean)
        status_combos = [
            (DIRTY, DIRTY),
            (CLEAN, DIRTY),
            (DIRTY, CLEAN),
            (CLEAN, CLEAN),
        ]

        states: List[VacuumState] = []
        state_id = 1
        for a_status, b_status in status_combos:
            for loc in self.locations:
                state = VacuumState(
                    agent_loc=loc,
                    room_a_status=a_status,
                    room_b_status=b_status,
                    state_id=state_id,
                )
                states.append(state)
                state_id += 1
        return states

    @staticmethod
    def calculate_state_space_size(num_rooms: int = 2, num_statuses: int = 2) -> int:
        """Calculate theoretical size of the vacuum world state space.

        Formula:
            |S| = num_rooms * (num_statuses ^ num_rooms)

        Args:
            num_rooms: Number of distinct room locations (e.g. 2).
            num_statuses: Number of cleanliness levels per room (e.g. 2).

        Returns:
            Total count of possible unique environmental states.
        """
        if num_rooms < 1 or num_statuses < 1:
            raise ValueError("Number of rooms and statuses must be positive integers.")
        return num_rooms * (num_statuses ** num_rooms)

    def transition_function(self, state: VacuumState, action: str) -> VacuumState:
        """Formal deterministic transition model Result(s, a).

        Rules:
        - 'Left':
            If agent is at Location A: agent remains at A (cyclic self-loop).
            If agent is at Location B: agent moves to Location A.
            Room cleanliness remains unchanged.
        - 'Right':
            If agent is at Location A: agent moves to Location B.
            If agent is at Location B: agent remains at B (cyclic self-loop).
            Room cleanliness remains unchanged.
        - 'Suck':
            Agent location remains unchanged.
            If agent is at Location A:
                Room A becomes Clean. If already Clean, state is unchanged (cyclic self-loop).
            If agent is at Location B:
                Room B becomes Clean. If already Clean, state is unchanged (cyclic self-loop).

        Args:
            state: Current VacuumState.
            action: Action to perform ('Left', 'Right', or 'Suck').

        Returns:
            Successor VacuumState from the state space.

        Raises:
            ValueError: If action is unrecognized.
        """
        norm_action = action.strip().capitalize()
        if norm_action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{action}'. Valid actions are {VALID_ACTIONS}."
            )

        new_loc = state.agent_loc
        new_a = state.room_a_status
        new_b = state.room_b_status

        if norm_action == ACTION_LEFT:
            new_loc = LOCATION_A
        elif norm_action == ACTION_RIGHT:
            new_loc = LOCATION_B
        elif norm_action == ACTION_SUCK:
            if state.agent_loc == LOCATION_A:
                new_a = CLEAN
            else:
                new_b = CLEAN

        successor = self.state_map[(new_loc, new_a, new_b)]
        return successor

    def _generate_transitions(self) -> List[TransitionRecord]:
        """Map all valid transitions for each state under each action.

        Returns:
            List of 24 TransitionRecord instances linking states.
        """
        transitions: List[TransitionRecord] = []
        for state in self.states:
            for action in VALID_ACTIONS:
                successor = self.transition_function(state, action)
                is_cyclic = state == successor

                # Formulate human-readable description
                if is_cyclic:
                    if action in (ACTION_LEFT, ACTION_RIGHT):
                        desc = f"Action '{action}' bumps into boundary wall at Location {state.agent_loc} (Cyclic Self-Loop)"
                    else:
                        desc = f"Action 'Suck' in already clean Location {state.agent_loc} produces no effect (Cyclic Self-Loop)"
                else:
                    if action in (ACTION_LEFT, ACTION_RIGHT):
                        desc = f"Action '{action}' moves agent from {state.agent_loc} to {successor.agent_loc}"
                    else:
                        desc = f"Action 'Suck' cleans Location {state.agent_loc}"

                transitions.append(
                    TransitionRecord(
                        source=state,
                        action=action,
                        target=successor,
                        is_cyclic=is_cyclic,
                        description=desc,
                    )
                )
        return transitions

    @property
    def goal_states(self) -> List[VacuumState]:
        """Return all goal states where both rooms are clean."""
        return [s for s in self.states if s.is_goal]

    def identify_cycles(self) -> Dict[str, Any]:
        """Perform comprehensive cycle detection on the state-space graph.

        Identifies:
        - All 1-cycles (self-loops) where Result(s, a) == s.
        - Action-specific breakdown of cyclic transitions.
        - All elementary cycles of length >= 2 (e.g. Left <-> Right oscillations).

        Returns:
            Dictionary containing cycle metrics, self-loops, and elementary cycle paths.
        """
        self_loops = [t for t in self.transitions if t.is_cyclic]
        loops_by_action: Dict[str, List[TransitionRecord]] = {
            a: [t for t in self_loops if t.action == a] for a in VALID_ACTIONS
        }

        # Build MultiDiGraph for cycle analysis
        g = self.to_networkx_graph(multi=True)
        # Simple elementary cycles in directed graph
        # Note: nx.simple_cycles on a DiGraph discovers non-self-loop simple cycles
        di_g = self.to_networkx_graph(multi=False)
        simple_cycles = list(nx.simple_cycles(di_g))

        return {
            "total_transitions": len(self.transitions),
            "total_self_loops": len(self_loops),
            "state_changing_transitions": len(self.transitions) - len(self_loops),
            "self_loops_by_action": {
                action: len(records) for action, records in loops_by_action.items()
            },
            "self_loop_records": self_loops,
            "elementary_cycles_count": len(simple_cycles),
            "elementary_cycles": simple_cycles,
        }

    def to_networkx_graph(self, multi: bool = False) -> Union[nx.DiGraph, nx.MultiDiGraph]:
        """Create and populate a NetworkX directed graph representation.

        Args:
            multi: If True, returns nx.MultiDiGraph with parallel edges per action.
                   If False, returns nx.DiGraph where parallel actions are consolidated.

        Returns:
            Populated NetworkX graph with node and edge attributes.
        """
        if multi:
            g: nx.MultiDiGraph = nx.MultiDiGraph()
        else:
            g = nx.DiGraph()

        # Add nodes with metadata
        for s in self.states:
            color = COLOR_GOAL if s.is_goal else COLOR_NON_GOAL
            border_color = COLOR_BORDER_GOAL if s.is_goal else COLOR_BORDER_NON_GOAL
            g.add_node(
                s,
                id=s.state_id,
                label=s.short_label,
                multiline_label=s.multiline_label,
                agent_loc=s.agent_loc,
                room_a=s.room_a_status,
                room_b=s.room_b_status,
                is_goal=s.is_goal,
                color=color,
                border_color=border_color,
            )

        if multi:
            for t in self.transitions:
                g.add_edge(
                    t.source,
                    t.target,
                    key=t.action,
                    action=t.action,
                    is_cyclic=t.is_cyclic,
                    color=ACTION_COLORS.get(t.action, "#333333"),
                    description=t.description,
                )
        else:
            # Group actions between identical source and target pairs
            edge_actions: Dict[Tuple[VacuumState, VacuumState], List[TransitionRecord]] = {}
            for t in self.transitions:
                pair = (t.source, t.target)
                edge_actions.setdefault(pair, []).append(t)

            for (src, tgt), records in edge_actions.items():
                actions_str = ", ".join(r.action for r in records)
                has_cycle = any(r.is_cyclic for r in records)
                # Primary color from first action or dark grey if mixed
                primary_color = (
                    ACTION_COLORS.get(records[0].action, "#333333")
                    if len(records) == 1
                    else "#34495e"
                )
                g.add_edge(
                    src,
                    tgt,
                    actions=[r.action for r in records],
                    action_label=actions_str,
                    is_cyclic=has_cycle,
                    color=primary_color,
                )

        return g

    def to_adjacency_matrix(self) -> Tuple[List[str], List[List[int]]]:
        """Compute the |S| x |S| adjacency matrix of the state-space graph.

        Returns:
            Tuple of (state_labels, 2D matrix of connection counts).
        """
        labels = [s.short_label for s in self.states]
        n = len(self.states)
        matrix = [[0] * n for _ in range(n)]
        state_to_idx = {s: i for i, s in enumerate(self.states)}

        for t in self.transitions:
            u = state_to_idx[t.source]
            v = state_to_idx[t.target]
            matrix[u][v] += 1

        return labels, matrix

    def find_shortest_paths_to_goal(
        self, start_state: VacuumState
    ) -> Dict[str, Any]:
        """Find optimal sequence of actions from a start state to nearest goal state.

        Uses Breadth-First Search (BFS) to guarantee path optimality.

        Args:
            start_state: Initial VacuumState.

        Returns:
            Dictionary with shortest path, actions taken, and step count.
        """
        if start_state.is_goal:
            return {
                "start": start_state,
                "goal": start_state,
                "path": [start_state],
                "actions": [],
                "distance": 0,
            }

        queue: List[Tuple[VacuumState, List[str], List[VacuumState]]] = [
            (start_state, [], [start_state])
        ]
        visited: Set[VacuumState] = {start_state}

        while queue:
            curr, path_actions, path_nodes = queue.pop(0)
            if curr.is_goal:
                return {
                    "start": start_state,
                    "goal": curr,
                    "path": path_nodes,
                    "actions": path_actions,
                    "distance": len(path_actions),
                }

            for action in VALID_ACTIONS:
                succ = self.transition_function(curr, action)
                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, path_actions + [action], path_nodes + [succ]))

        return {
            "start": start_state,
            "goal": None,
            "path": [],
            "actions": [],
            "distance": -1,
        }


class StateSpaceVisualizer:
    """Renderer and visualizer for the reflex vacuum agent's state-space graph.

    Provides publication-quality matplotlib plotting with customized layouts,
    distinct goal highlighting, cyclic transition self-loop overlays, and ASCII rendering.
    """

    def __init__(self, state_space: Optional[StateSpace] = None) -> None:
        """Initialize visualizer with a StateSpace model.

        Args:
            state_space: StateSpace instance (creates standard 2-room space if None).
        """
        self.state_space: StateSpace = state_space if state_space is not None else StateSpace()

    def get_structured_layout(self) -> Dict[VacuumState, Tuple[float, float]]:
        """Generate structured 2-column textbook layout matching AIMA Figure 3.2.

        Organization:
            - Left column (x = -1.5): Agent at Location A (States 1, 3, 5, 7)
            - Right column (x = 1.5): Agent at Location B (States 2, 4, 6, 8)
            - Vertical levels (y):
                y = 3.0: Both rooms Dirty (States 1, 2)
                y = 2.0: Room A Clean, Room B Dirty (States 3, 4)
                y = 1.0: Room A Dirty, Room B Clean (States 5, 6)
                y = 0.0: Both rooms Clean [GOAL STATES] (States 7, 8)

        Returns:
            Dictionary mapping each VacuumState to its (x, y) coordinate.
        """
        pos: Dict[VacuumState, Tuple[float, float]] = {}
        for s in self.state_space.states:
            x = -1.6 if s.agent_loc == LOCATION_A else 1.6

            # Determine cleanliness level
            if s.room_a_status == DIRTY and s.room_b_status == DIRTY:
                y = 3.0
            elif s.room_a_status == CLEAN and s.room_b_status == DIRTY:
                y = 2.0
            elif s.room_a_status == DIRTY and s.room_b_status == CLEAN:
                y = 1.0
            else:  # Both Clean (Goal)
                y = 0.0

            pos[s] = (x, y)
        return pos

    def compute_positions(
        self,
        graph: nx.Graph,
        layout_type: str = "textbook",
    ) -> Dict[VacuumState, Tuple[float, float]]:
        """Compute 2D node layout positions based on requested layout mode.

        Args:
            graph: NetworkX graph.
            layout_type: Layout algorithm: 'textbook' (bipartite), 'circular',
                         'spring', 'shell', or 'kamada_kawai'.

        Returns:
            Dictionary mapping nodes to 2D coordinates.
        """
        mode = layout_type.strip().lower()
        if mode in ("textbook", "bipartite", "grid"):
            return self.get_structured_layout()
        elif mode == "circular":
            return nx.circular_layout(graph)
        elif mode == "spring":
            return nx.spring_layout(graph, seed=42, k=1.8)
        elif mode == "shell":
            return nx.shell_layout(graph)
        else:
            return self.get_structured_layout()

    def plot_graph(
        self,
        layout_type: str = "textbook",
        highlight_goals: bool = True,
        highlight_cycles: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        figsize: Tuple[int, int] = (15, 11),
        title: str = "State-Space Graph: 2-Room Vacuum Cleaner Agent (8 States, 24 Transitions)",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Render publication-quality state-space directed graph using matplotlib.

        Highlights goal states in emerald green (Optional Challenge) and displays
        state-changing directed transitions and self-loop cyclic actions clearly.

        Args:
            layout_type: Node placement strategy ('textbook', 'circular', 'spring').
            highlight_goals: Whether to color goal states in green.
            highlight_cycles: Whether to annotate cyclic self-loop transitions.
            save_path: Optional file path to export image (PNG, SVG, PDF).
            show: Whether to invoke plt.show() (disabled by default for test safety).
            figsize: Tuple of (width, height) in inches.
            title: Main chart title.

        Returns:
            Tuple of (matplotlib Figure, matplotlib Axes).
        """
        g = self.state_space.to_networkx_graph(multi=False)
        pos = self.compute_positions(g, layout_type=layout_type)

        fig, ax = plt.subplots(figsize=figsize, facecolor="#f8f9fa")
        ax.set_facecolor("#f8f9fa")

        # 1. Separate state-changing edges from self-loops
        inter_edges = [(u, v) for u, v in g.edges() if u != v]
        self_loop_edges = [(u, v) for u, v in g.edges() if u == v]

        # 2. Draw state-changing directed edges with curvature
        for u, v in inter_edges:
            actions_list = g[u][v].get("actions", [])
            label_text = ", ".join(actions_list)

            # Determine arrow curvature: curve slightly to prevent bidirectional overlap
            # Left transitions curve one way, Right transitions curve the other
            rad = 0.15 if (u.agent_loc == LOCATION_B and v.agent_loc == LOCATION_A) else -0.15
            if u.room_a_status != v.room_a_status or u.room_b_status != v.room_b_status:
                rad = 0.08  # Suck action going downwards

            primary_action = actions_list[0] if actions_list else ACTION_SUCK
            edge_color = ACTION_COLORS.get(primary_action, "#333333")

            arrow = FancyArrowPatch(
                pos[u],
                pos[v],
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=16,
                color=edge_color,
                linewidth=1.8,
                alpha=0.85,
                zorder=2,
            )
            ax.add_patch(arrow)

            # Position edge label near arrow midpoint with slight offset
            mid_x = (pos[u][0] + pos[v][0]) / 2.0
            mid_y = (pos[u][1] + pos[v][1]) / 2.0 + rad * 0.4
            ax.text(
                mid_x,
                mid_y,
                label_text,
                fontsize=8,
                color=edge_color,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=edge_color, alpha=0.85, lw=0.6),
                zorder=3,
            )

        # 3. Draw and annotate cyclic self-loops
        if highlight_cycles:
            for u, _ in self_loop_edges:
                actions = g[u][u].get("actions", [])
                action_text = f"↺ {', '.join(actions)}"
                x, y = pos[u]

                # Position loop bubble outward (left for Loc A, right for Loc B)
                offset_x = -0.52 if u.agent_loc == LOCATION_A else 0.52
                loop_x = x + offset_x

                ax.text(
                    loop_x,
                    y,
                    action_text,
                    fontsize=8,
                    fontweight="bold",
                    color="#c0392b",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="#fadbd8",
                        edgecolor="#e74c3c",
                        linestyle="--",
                        alpha=0.9,
                        lw=1.0,
                    ),
                    zorder=4,
                )

                # Connect self-loop indicator with small dashed arc
                loop_arc = FancyArrowPatch(
                    (x + (offset_x * 0.4), y + 0.15),
                    (x + (offset_x * 0.4), y - 0.15),
                    arrowstyle="->",
                    connectionstyle=f"arc3,rad={'0.6' if offset_x > 0 else '-0.6'}",
                    color="#e74c3c",
                    linestyle="--",
                    linewidth=1.2,
                    zorder=3,
                )
                ax.add_patch(loop_arc)

        # 4. Draw nodes (Goal vs Non-Goal)
        node_colors: List[str] = []
        border_colors: List[str] = []
        for s in g.nodes():
            if highlight_goals and s.is_goal:
                node_colors.append(COLOR_GOAL)
                border_colors.append(COLOR_BORDER_GOAL)
            else:
                node_colors.append(COLOR_NON_GOAL)
                border_colors.append(COLOR_BORDER_NON_GOAL)

        # Node drawing via NetworkX
        nx.draw_networkx_nodes(
            g,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=3600,
            edgecolors=border_colors,
            linewidths=2.5,
            node_shape="s",  # Rounded rectangular box
        )

        # 5. Draw node labels
        labels = {s: s.multiline_label for s in g.nodes()}
        for s, label_text in labels.items():
            x, y = pos[s]
            font_color = "#0b5345" if (highlight_goals and s.is_goal) else "#1b4f72"
            ax.text(
                x,
                y,
                label_text,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                color=font_color,
                zorder=5,
            )

        # 6. Build Legend and Title
        legend_patches = [
            mpatches.Patch(
                facecolor=COLOR_GOAL,
                edgecolor=COLOR_BORDER_GOAL,
                label="Goal State (Both Rooms Clean) [Optional Challenge]",
            ),
            mpatches.Patch(
                facecolor=COLOR_NON_GOAL,
                edgecolor=COLOR_BORDER_NON_GOAL,
                label="Non-Goal State (Dirt Present in A or B)",
            ),
            mpatches.Patch(
                facecolor=ACTION_COLORS[ACTION_LEFT],
                label="Action: Left (Move/Stay at A)",
            ),
            mpatches.Patch(
                facecolor=ACTION_COLORS[ACTION_RIGHT],
                label="Action: Right (Move/Stay at B)",
            ),
            mpatches.Patch(
                facecolor=ACTION_COLORS[ACTION_SUCK],
                label="Action: Suck (Clean Current Room)",
            ),
            mpatches.Patch(
                facecolor="#fadbd8",
                edgecolor="#e74c3c",
                linestyle="--",
                label="Cyclic Transition ↺ (Self-Loop: Wall bump / Redundant Suck)",
            ),
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=9,
            frameon=True,
            facecolor="white",
            framealpha=0.95,
            edgecolor="#bdc3c7",
        )

        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
            pad=18,
            color="#2c3e50",
        )

        # Set clean plot bounds
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
        ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
        ax.axis("off")
        plt.tight_layout()

        # Save to file if path specified
        if save_path:
            out_file = Path(save_path).resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_file, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax

    def render_ascii_graph(self) -> str:
        """Render a clean textual / ASCII diagram of the 8-state transition graph.

        Useful for headless CLI environments and terminal logging.

        Returns:
            Human-readable formatted ASCII representation of all 8 states.
        """
        divider = "=" * 76
        sub_divider = "-" * 76
        lines = [
            divider,
            "         STATE-SPACE TRANSITION TABLE: 2-ROOM REFLEX VACUUM AGENT",
            divider,
            f"{'State':<10}{'Tuple (Loc, A, B)':<22}{'Action Left':<15}{'Action Right':<15}{'Action Suck':<14}",
            sub_divider,
        ]

        for s in self.state_space.states:
            goal_marker = " [GOAL]" if s.is_goal else ""
            state_str = f"S{s.state_id}{goal_marker}"
            tuple_str = f"({s.agent_loc}, {s.room_a_status[:1]}, {s.room_b_status[:1]})"

            res_left = self.state_space.transition_function(s, ACTION_LEFT)
            res_right = self.state_space.transition_function(s, ACTION_RIGHT)
            res_suck = self.state_space.transition_function(s, ACTION_SUCK)

            l_str = f"-> S{res_left.state_id}" + (" (↺)" if s == res_left else "")
            r_str = f"-> S{res_right.state_id}" + (" (↺)" if s == res_right else "")
            s_str = f"-> S{res_suck.state_id}" + (" (↺)" if s == res_suck else "")

            lines.append(
                f"{state_str:<10}{tuple_str:<22}{l_str:<15}{r_str:<15}{s_str:<14}"
            )

        lines.append(divider)
        cycle_info = self.state_space.identify_cycles()
        lines.append(
            f"  Total States: {len(self.state_space.states)} | Total Transitions: {cycle_info['total_transitions']}"
        )
        lines.append(
            f"  Self-Loop Cycles (↺): {cycle_info['total_self_loops']} (Left: {cycle_info['self_loops_by_action'][ACTION_LEFT]}, "
            f"Right: {cycle_info['self_loops_by_action'][ACTION_RIGHT]}, Suck: {cycle_info['self_loops_by_action'][ACTION_SUCK]})"
        )
        lines.append(
            f"  State-Changing Transitions: {cycle_info['state_changing_transitions']}"
        )
        lines.append(
            f"  Goal States (Both Clean): S7 (A, Clean, Clean) and S8 (B, Clean, Clean)"
        )
        lines.append(divider)
        return "\n".join(lines)


def get_viva_reference() -> str:
    """Return comprehensive Viva Voce theory & reference study guide.

    Topics covered:
    1. State space size calculation (|S| = n * 2^n)
    2. Formulation of search problems (Initial state, Actions, Transition model, Goal test, Path cost)
    3. Graph visualization techniques (Layouts, curved multi-edges, self-loops)
    4. Reflex vs Goal-Based Agents in state-space search
    5. Cyclic transitions and infinite loops in memoryless reflex agents

    Returns:
        Formatted multi-section educational string.
    """
    return """================================================================================
          VIVA VOCE REFERENCE GUIDE: STATE-SPACE FORMULATION & VISUALIZATION
================================================================================

1. STATE SPACE SIZE CALCULATION:
   -----------------------------
   * General Formula:
     For a vacuum cleaner world consisting of 'n' distinct locations, where each
     location can independently hold one of 'k' cleanliness statuses:
         |S| = n * (k ^ n)

   * 2-Room Environment Calculation (Russell & Norvig Benchmark):
     - Locations: {A, B} -> n = 2
     - Statuses: {Clean, Dirty} -> k = 2
     - Total States: |S| = 2 * (2 ^ 2) = 2 * 4 = 8 states.

   * Scaling Behavior with Number of Rooms:
     +-------+-------------------------+----------------------+
     | Rooms | Formula: n * 2^n        | Total State Space    |
     +-------+-------------------------+----------------------+
     |   1   | 1 * 2^1                 | 2 states             |
     |   2   | 2 * 2^2                 | 8 states             |
     |   3   | 3 * 2^3                 | 24 states            |
     |   4   | 4 * 2^4                 | 64 states            |
     |   5   | 5 * 2^5                 | 160 states           |
     |  10   | 10 * 2^10               | 10,240 states        |
     +-------+-------------------------+----------------------+
     Notice exponential growth with respect to room count! This demonstrates the
     classic "Combinatorial State Space Explosion".

2. FORMULATION OF SEARCH PROBLEMS (AIMA 5-TUPLE):
   ----------------------------------------------
   A classical search problem is mathematically formulated using 5 components:
   1. Initial State:
      The starting state the agent occupies, e.g., State 1: (A, Dirty, Dirty).
   2. Actions Set Actions(s):
      The legal actions available in state s. Here, Actions(s) = {Left, Right, Suck}
      for every state s.
   3. Transition Model Result(s, a):
      A formal function returning the successor state achieved by executing
      action 'a' in state 's'.
   4. Goal Test Is-Goal(s):
      A Boolean predicate determining whether state s is a goal state.
      Here: Is-Goal(s) <=> (RoomA == Clean AND RoomB == Clean).
      Satisfied by State 7: (A, Clean, Clean) and State 8: (B, Clean, Clean).
   5. Path Cost Function c(s, a, s'):
      Assigns a numeric cost to taking step 'a' from s to s' (typically step cost = 1).

3. CYCLIC TRANSITIONS & REVISION DYNAMICS:
   ---------------------------------------
   * What is a Cyclic Transition?
     A transition Result(s, a) -> s' where s == s' (a directed self-loop or 1-cycle).
   * Exact Count in 2-Room Environment:
     Across all 8 states * 3 actions = 24 transitions:
     - 4 Left self-loops: Occur whenever agent is already in Room A.
     - 4 Right self-loops: Occur whenever agent is already in Room B.
     - 4 Suck self-loops: Occur when agent sucks in an already clean room:
       * S3: (A, Clean, Dirty) + Suck -> S3
       * S6: (B, Dirty, Clean) + Suck -> S6
       * S7: (A, Clean, Clean) + Suck -> S7
       * S8: (B, Clean, Clean) + Suck -> S8
     - Total: Exactly 12 self-loops and 12 state-changing transitions.
   * Why do cyclic transitions trap Simple Reflex Agents?
     Because a simple reflex agent has NO internal memory. If its condition-action
     rule leads to an unchanged percept (e.g. Left at Room A when Clean), it
     enters an infinite execution loop unless randomized or goal-directed.

4. GRAPH VISUALIZATION TECHNIQUES:
   -------------------------------
   * Layout Selection:
     - Spring Layout: Uses force-directed physics (Fruchterman-Reingold). Good for
       unstructured graphs, but can produce erratic node placement.
     - Bipartite / Structured Grid Layout (Textbook):
       Places Location A in column 1 and Location B in column 2, ordered vertically
       by cleanliness level. Perfectly mirrors AIMA Figure 3.2.
   * Handling Parallel & Bidirectional Edges:
     - In NetworkX and Matplotlib, reciprocal edges (u -> v and v -> u) overlap
       if drawn as straight lines.
     - Solution: Use curved spline arrows via FancyArrowPatch with opposite
       radii (connectionstyle="arc3,rad=0.15" vs "arc3,rad=-0.15").
   * Self-Loop Annotation:
     - Straight edge renderers fail to display self-loops. Drawing circular loop
       markers or explicit label badges guarantees that reflexive actions are clear.
   * Color Coding for Goal States:
     - Goal states are visually distinguished with Emerald Green (#2ecc71),
       drawing immediate attention to target problem objectives.

5. COMPUTATIONAL COMPLEXITY:
   -------------------------
   - State Generation: O(|S|) = O(n * 2^n) time and space.
   - Transition Mapping: O(|S| * |A|) = O(8 * 3) = 24 operations <= O(|S|^2).
   - Adjacency Matrix Representation: O(|S|^2) time and space (64 cells for 8 states).
   - Path Search (BFS): O(|S| + |E|) where |E| = 24 directed edges.
================================================================================"""
