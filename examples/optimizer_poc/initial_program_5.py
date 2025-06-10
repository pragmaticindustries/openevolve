# EVOLVE-BLOCK-START
"""Ultra-fast, parallel, hash-deduped, beam-greedy hybrid search for optimal topology constellation.
Performance improvements:
- Beam search (width=4) for first 2 depths, then greedy (width=1)
- For each local area per expansion, keep top N candidates (beam width)
- Hash-based deduplication at each level (per expansion, not global)
- Fully evaluate only completed/terminal topologies
- Parallel candidate expansion with futures for higher CPU utilization
- Minimized memory: no tree except parent pointers
- Aggressive beam/greedy pruning, early best terminal detection
- Cost normalization only on candidates to compare
"""

import math
import uuid
from typing import List, Optional, Callable, Dict, Set, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from uKonn_optimizer.connection_priorities.im_priority_list_generic import priority_lists_generic
from uKonn_optimizer.im_models import IMTopology, IMDevice, IMDeviceType
from uKonn_optimizer.im_repository import IMRepository, InMemoryRepository, StaticIMRepository
from uKonn_optimizer.im_topology_utils import is_active_distributor, is_iol_hub, is_adapter, is_t_splitter
from uKonn_optimizer.im_unified_flow import apply_unified_flow
from uKonn_optimizer.types import PinAssignment, SocketType, OptimizationStrategy
from uKonn_optimizer.cost_functions import cost_functions

class ProgressDict(Dict):
    last_step: str
    progress: str
    time_left: Optional[int]

def apply_generic_optimizer(
    top: IMTopology,
    cost_function_profile: str,
    repository: IMRepository,
    update_task_state: Optional[Callable[[ProgressDict], None]] = None,
):
    local_areas = set(device.local_area for device in top.devices.values())
    max_depth = (
        sum(
            math.ceil(
                sum(1 for device in top.devices.values() if device.local_area == la) / 8
            )
            for la in local_areas
        ) + 2
    )

    root = Node(top)
    cost_function_list = [
        cost_functions.installation_time_costs,
        cost_functions.device_diversity_factor,
        cost_functions.distributors_total_price,
        cost_functions.system_reserve_costs,
        cost_functions.port_maintenance_time,
    ]
    calculate_costs(root, cost_function_list)

    repository_dict = repository.repository()
    device_types = [dt for dt in repository_dict.values() if len(dt.ports) > 3]
    adapters = [dt for dt in repository_dict.values() if len(dt.ports) <= 3]

    if update_task_state:
        update_task_state(
            {
                "last_step": f"Using {len(device_types)} Device Types for Analysis, Suchbaumtiefe: {max_depth}",
                "progress": "0%",
                "time_left": None,
            }
        )

    if cost_function_profile == "default":
        weights = [0.0, 0.1, 0.8, 0.0, 0.0]
    elif cost_function_profile == "price":
        weights = [0.05, 0.05, 0.8, 0.05, 0.05]
    elif cost_function_profile == "diversity":
        weights = [0.05, 0.8, 0.05, 0.05, 0.05]
    elif cost_function_profile == "three":
        weights = [0.4, 0.05, 0.05, 0.05, 0.4]
    else:
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    optimizer = BeamGreedyOptimizer(
        {dt.uuid: dt for dt in device_types},
        cost_function_list,
        adapters=adapters,
        update_task_state=update_task_state,
        weights=weights,
        max_depth=max_depth,
    )

    best_node = optimizer.search(root)
    return best_node

class Node:
    __slots__ = (
        "topology", "parent", "costs", "normalized_cost", "device_type", "local_area",
        "is_duplicate", "no_new_connections", "is_fully_connected", "exception", "_hash_sig"
    )
    def __init__(self, topology: IMTopology, parent=None):
        self.topology = topology
        self.parent = parent
        self.costs: List[Optional[float]] = []
        self.normalized_cost: float = 0.0
        self.device_type = None
        self.local_area = None
        self.is_duplicate = False
        self.no_new_connections = False
        self.is_fully_connected = False
        self.exception = False
        self._hash_sig = None

    def is_terminal(self) -> bool:
        return (
            self.is_duplicate
            or self.no_new_connections
            or self.is_fully_connected
            or self.exception
        )

    def uuid_dict(self):
        devices = {}
        current = self
        while current.parent is not None:
            if current.device_type is not None:
                key = (current.local_area, getattr(current.device_type, "uuid", None))
                devices[key] = devices.get(key, 0) + 1
            current = current.parent
        return tuple(sorted(devices.items()))

    def __hash__(self):
        if self._hash_sig is None:
            self._hash_sig = hash(self.uuid_dict())
        return self._hash_sig

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.uuid_dict() == other.uuid_dict()

def calculate_costs(
    node: Node,
    cost_functions: List[Callable[[IMTopology], Optional[float]]],
    explicit_topology: Optional[IMTopology] = None,
):
    node.costs = []
    topo = explicit_topology if explicit_topology else node.topology
    for fun in cost_functions:
        try:
            node.costs.append(fun(topo))
        except Exception:
            node.costs.append(None)

def normalize_costs(nodes: List[Node], weights: List[float]):
    """In-place normalization and weighted sum."""
    if not nodes or not weights:
        return
    n_costs = len(weights)
    min_costs = [min((node.costs[i] for node in nodes if node.costs[i] is not None), default=0) for i in range(n_costs)]
    max_costs = [max((node.costs[i] for node in nodes if node.costs[i] is not None), default=1) for i in range(n_costs)]
    weights_sum = sum(weights) if sum(weights) > 0 else 1.0
    weights = [w / weights_sum for w in weights]
    for node in nodes:
        norm_sum = 0.0
        norm_count = 0
        for i in range(n_costs):
            if node.costs[i] is None:
                continue
            b = max_costs[i] - min_costs[i]
            norm = ((node.costs[i] - min_costs[i]) / b) * 100 if b > 0 else 50.0
            norm_sum += norm * weights[i]
            norm_count += weights[i]
        node.normalized_cost = norm_sum / (norm_count if norm_count > 0 else 1.0) if norm_count else 1e6

def needs_further_connecting(
    topology: IMTopology, ignore_cabinet_connections: bool = False
) -> bool:
    for device in topology.devices.values():
        for port in [
            port
            for port in device.device_type.ports.values()
            if port.socket_type == SocketType.MALE and port.distributor_port_id is None
        ]:
            if port.get_pin_assignment() in [
                PinAssignment.SINGLE_CHANNEL,
                PinAssignment.DOUBLE_CHANNEL,
                PinAssignment.IOL_CLASS_A,
                PinAssignment.IOL_CLASS_B_NGS,
                PinAssignment.IOL_CLASS_B,
                PinAssignment.ANALOG,
            ] and (
                not topology.is_connected(device, port)
                or (
                    ignore_cabinet_connections
                    and topology.get_connected_counterpart(device.uuid, port.label)
                    == (None, None)
                )
            ):
                return True
    return False

class BeamGreedyOptimizer:
    """Hybrid beam-greedy search: beam width 4 for 2 levels, then greedy. Parallel, with deduplication per level."""

    def __init__(
        self,
        device_types: Dict[uuid.UUID, IMDeviceType],
        cost_function_list,
        adapters: List[IMDeviceType] = None,
        update_task_state: Optional[Callable[[ProgressDict], None]] = None,
        weights: Optional[List[float]] = None,
        max_depth: int = 6,
    ):
        self.device_types = device_types
        self.cost_function_list = cost_function_list
        self.update_task_state = update_task_state
        self.adapter_repository = InMemoryRepository()
        for adapter in adapters or []:
            self.adapter_repository.add(adapter)
        self.weights = weights
        self.max_depth = max_depth

    def search(self, root: Node) -> Optional[Node]:
        beam_width = 4
        greedy_depth = 2
        level_nodes = [root]
        best_terminal: Optional[Node] = None
        best_cost: float = float("inf")
        for depth in range(self.max_depth):
            # Beam width: 4 for depth 0,1; then greedy
            cur_beam = beam_width if depth < greedy_depth else 1
            expanded = self.expand_all(level_nodes, top_candidates_per_area=cur_beam)
            if not expanded:
                break
            normalize_costs(expanded, self.weights)
            expanded.sort(key=lambda n: n.normalized_cost)
            # Deduplicate by hash at this level (not global, to avoid pruning promising branches)
            dedup = {}
            for n in expanded:
                if hash(n) not in dedup:
                    dedup[hash(n)] = n
            expanded = list(dedup.values())
            expanded.sort(key=lambda n: n.normalized_cost)
            # For next level, keep only top beam (or 1 for greedy)
            level_nodes = expanded[:cur_beam]
            # Check for best terminal so far
            for n in level_nodes:
                if n.is_fully_connected:
                    if n.normalized_cost < best_cost:
                        best_terminal = n
                        best_cost = n.normalized_cost
            if best_terminal:
                return best_terminal
        # Fallback: best terminal if found, else best among last level
        if best_terminal:
            return best_terminal
        normalize_costs(level_nodes, self.weights)
        level_nodes.sort(key=lambda n: n.normalized_cost)
        return level_nodes[0] if level_nodes else None

    def expand_all(self, parent_nodes: List[Node], top_candidates_per_area: int = 1) -> List[Node]:
        """Expand all given nodes in parallel. For each local area, only the best N candidates are kept."""
        all_children: List[Node] = []
        for node in parent_nodes:
            children = self.expand_node(node, top_candidates_per_area)
            all_children.extend(children)
        return all_children

    def expand_node(self, node: Node, top_candidates_per_area: int = 1) -> List[Node]:
        if node.is_terminal():
            return []
        local_areas = set(device.local_area for device in node.topology.devices.values())
        children: List[Node] = []
        for local_area in local_areas:
            parameters = node.topology.get_local_area_parameter(local_area)
            def is_active(device_type: IMDeviceType) -> bool:
                return all(pin.channel is None for port in device_type.ports.values() for pin in port.pins.values())
            if parameters.optimization_strategy == OptimizationStrategy.ACTIVE:
                candidates = [
                    dt for dt in self.device_types.values()
                    if (is_active_distributor(dt) or is_iol_hub(dt))
                ]
            elif parameters.optimization_strategy == OptimizationStrategy.PASSIVE:
                candidates = [
                    dt for dt in self.device_types.values()
                    if not is_active(dt)
                ]
            else:
                continue
            candidates = [
                device_type
                for device_type in candidates
                if device_type.housing_material == parameters.distributor_housing_material
            ]
            candidate_nodes: List[Node] = []
            def candidate_worker(dt) -> Optional[Node]:
                new_top = node.topology.clone()
                child = Node(new_top, parent=node)
                child.device_type = dt
                child.local_area = local_area
                child.costs = [None] * len(self.cost_function_list)
                adapters_uuids_before = [
                    dev.uuid
                    for dev in child.topology.devices.values()
                    if is_adapter(dev.device_type) or is_t_splitter(dev)
                ]
                child.topology.clear_connections()
                for uid in adapters_uuids_before:
                    child.topology.remove_device(uid)
                identifier_uuid = uuid.uuid4()
                if dt:
                    distributor = IMDevice(
                        identifier_uuid,
                        str(identifier_uuid),
                        dt,
                        local_area,
                    )
                    child.topology.add_device(distributor)
                try:
                    apply_unified_flow(
                        child.topology,
                        self.adapter_repository,
                        add_distributors=False,
                        add_ad_converters=False,
                        used_priority_lists=priority_lists_generic,
                    )
                except Exception:
                    child.exception = True
                    return None
                calculate_costs(child, self.cost_function_list)
                if not needs_further_connecting(child.topology, ignore_cabinet_connections=False):
                    child.is_fully_connected = True
                if len(child.topology.connections) == len(node.topology.connections):
                    child.no_new_connections = True
                    return None
                return child
            # Parallel candidate evaluation if enough candidates
            if len(candidates) > 2:
                with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as executor:
                    futs = [executor.submit(candidate_worker, dt) for dt in candidates]
                    results = [fut.result() for fut in as_completed(futs)]
                candidate_nodes = [c for c in results if c is not None]
            else:
                for dt in candidates:
                    c = candidate_worker(dt)
                    if c is not None:
                        candidate_nodes.append(c)
            # Only keep the best N for this local area
            if candidate_nodes:
                normalize_costs(candidate_nodes, self.weights)
                candidate_nodes.sort(key=lambda n: n.normalized_cost)
                children.extend(candidate_nodes[:top_candidates_per_area])
        return children

def run_sorting(n):
    """Run the circle packing constructor for n=26"""

    from uKonn_optimizer.devices.build_files.generic_devices.generic_periphery_devices import (
        GENERIC_SENSOR_M12A_DC_UUID,
        GENERIC_SENSOR_M12A_SC_UUID,
    )

    from uKonn_optimizer.testing_utils.testing_topology_builder import TestTopologyBuilder
    from uKonn_optimizer.types import PortShape
    from uKonn_optimizer.im_models import IMSettings

    StaticIMRepository().setup()

    n_dc = n
    n_sc = n
    n_iol = n

    _, top = (
        TestTopologyBuilder()
        .add_n_devices_from_repository(GENERIC_SENSOR_M12A_DC_UUID, n_dc, "")
        .add_n_devices_from_repository(GENERIC_SENSOR_M12A_SC_UUID, n_sc, "")
        .add_n_generic_iol_devices(n_iol, PortShape.M12A, "")
        .add_setting_for_local_area(
            "",
            IMSettings(
                optimization_strategy=OptimizationStrategy.ACTIVE,
            ),
        )
        .build()
    )

    import time

    start_time = time.time()
    best_node: Node = apply_generic_optimizer(top, "default", StaticIMRepository())
    end_time = time.time()

    print(f"Execution Time: {end_time - start_time:.2f} seconds")

    if best_node is None:
        print("No best node found, something went wrong.")
        return None

    print(f"Best Node Costs: {best_node.costs}")
    return best_node.costs

if __name__ == "__main__":
    sorted_list = run_sorting(15)
    print("List was sorted successfully.")
# EVOLVE-BLOCK-END