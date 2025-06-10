# EVOLVE-BLOCK-START
"""Highly-optimized Tree search for topology optimization using fast deduplication, beam search, and aggressive pruning."""

import math
import uuid
from typing import List, Optional, Callable, Dict, TypedDict

from uKonn_optimizer.connection_priorities.im_priority_list_generic import (
    priority_lists_generic,
)
from uKonn_optimizer.im_models import (
    IMTopology,
    IMDevice,
    IMDeviceType,
)
from uKonn_optimizer.im_repository import IMRepository, InMemoryRepository, StaticIMRepository
from uKonn_optimizer.im_topology_utils import (
    is_active_distributor,
    is_iol_hub,
    is_adapter,
    is_t_splitter,
)
from uKonn_optimizer.im_unified_flow import (
    apply_unified_flow,
)

from uKonn_optimizer.types import PinAssignment, SocketType, OptimizationStrategy
from uKonn_optimizer.cost_functions import cost_functions

node_counter = 0

class ProgressDict(TypedDict):
    last_step: str
    progress: str
    time_left: Optional[int]

def apply_generic_optimizer(
    top: IMTopology,
    cost_function_profile: str,
    repository: IMRepository,
    update_task_state: Optional[Callable[[ProgressDict], None]] = None,
):
    # Estimate max depth (unchanged)
    local_areas = set([device.local_area for device in top.devices.values()])
    max_depth = (
        sum([
            math.ceil(
                len([
                    1
                    for device in top.devices.values()
                    if device.local_area == la
                ]) / 8
            )
            for la in local_areas
        ]) + 2
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

    optimization_mode = False

    traveller = NodeEvaluator(
        {dt.uuid: dt for dt in device_types},
        cost_function_list,
        include_not_fully_connected=optimization_mode,
        adapters=adapters,
        update_task_state=update_task_state,
        beam_width=4,  # Narrower beam for speed
    )

    # Run the optimized search
    best_node = traveller.beam_search(root, max_depth, weights)
    return best_node

class Node:
    __slots__ = (
        "node_id", "dead_end", "topology", "parent", "children", "costs",
        "normalized_costs", "normalized_cost", "device_type", "local_area",
        "is_duplicate", "no_new_connections", "is_fully_connected", "exception",
        "_hash_sig"
    )
    def __init__(self, topology: IMTopology, parent=None):
        global node_counter
        self.node_id = node_counter
        node_counter += 1
        self.dead_end = False
        self.topology = topology
        self.parent = parent
        self.children: List["Node"] = []
        self.costs: List[Optional[float]] = []
        self.normalized_costs: List[Optional[float]] = []
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
        # Efficient signature: devices by (local_area, device_type), sorted, for deduplication
        devices = {}
        current_node = self
        while current_node.parent is not None:
            if current_node.device_type is not None:
                key = (current_node.local_area, getattr(current_node.device_type, "uuid", None))
                devices[key] = devices.get(key, 0) + 1
            current_node = current_node.parent
        return tuple(sorted(devices.items()))

    def __hash__(self):
        if self._hash_sig is None:
            self._hash_sig = hash(self.uuid_dict())
        return self._hash_sig

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.uuid_dict() == other.uuid_dict()

    def add_child(self, child: "Node"):
        self.children.append(child)
        child.parent = self

def calculate_costs(
    node: Node,
    cost_functions: List[Callable[[IMTopology], Optional[float]]],
    explicit_topology: Optional[IMTopology] = None,
):
    node.costs = []
    for fun in cost_functions:
        node.costs.append(
            fun(explicit_topology if explicit_topology else node.topology)
        )

class NodeEvaluator:
    """
    NodeEvaluator with beam search, deduplication, and aggressive pruning.
    """
    def __init__(
        self,
        device_types: Dict[uuid.UUID, IMDeviceType],
        cost_function_list,
        include_not_fully_connected: bool = False,
        adapters: List[IMDeviceType] = None,
        update_task_state: Optional[Callable[[ProgressDict], None]] = None,
        beam_width: int = 4,
    ) -> None:
        self.n_analysis = 0
        self.seen_hashes = set()
        self.device_types = device_types
        self.cost_function_list = cost_function_list
        self.include_not_fully_connected = include_not_fully_connected
        self.update_task_state = update_task_state
        self.adapter_repository = InMemoryRepository()
        for adapter in adapters or []:
            self.adapter_repository.add(adapter)
        self.beam_width = beam_width

    def beam_search(
        self,
        root: Node,
        max_depth: int,
        weights: Optional[List[float]],
    ) -> Optional[Node]:
        # Fastest: maintain only beam_width best nodes at each level
        current_level = [root]
        best_terminal = None
        best_cost = float("inf")
        for d in range(max_depth):
            next_level = []
            for node in current_level:
                children = self.expand_node(node)
                for child in children:
                    # Immediate pruning: only keep non-duplicate/non-dead
                    if not child.is_terminal():
                        next_level.append(child)
                    elif child.is_fully_connected:
                        # Evaluate costs for best terminal
                        normalize_costs(child, weights)
                        if child.normalized_cost < best_cost:
                            best_terminal = child
                            best_cost = child.normalized_cost
            if not next_level:
                break
            # Prune to beam width, using normalized cost
            normalize_costs(root, weights)
            next_level.sort(key=lambda n: n.normalized_cost)
            if len(next_level) > self.beam_width:
                next_level = next_level[:self.beam_width]
            current_level = next_level
        # After search, find best terminal node reachable
        if best_terminal:
            return best_terminal
        # Otherwise, fallback: try best among all fully connected
        return get_best_node(root, weights=weights, include_not_terminal_nodes=False)

    def expand_node(self, node: Node) -> List[Node]:
        """Return all valid children, avoiding duplicates. Only expand if not terminal."""
        if node.is_terminal():
            return []
        children = []
        # All local areas
        local_areas = set(
            [device.local_area for device in node.topology.devices.values()]
        )
        for local_area in local_areas:
            parameters = node.topology.get_local_area_parameter(local_area)
            def is_active(device_type: IMDeviceType) -> bool:
                for port in device_type.ports.values():
                    for pin in port.pins.values():
                        if pin.channel is not None:
                            return False
                return True
            # Filter device types for this local area and strategy
            if parameters.optimization_strategy == OptimizationStrategy.ACTIVE:
                candidates = [
                    dt
                    for dt in self.device_types.values()
                    if (is_active_distributor(dt) or is_iol_hub(dt))
                ]
            elif parameters.optimization_strategy == OptimizationStrategy.PASSIVE:
                candidates = [
                    dt
                    for dt in self.device_types.values()
                    if not is_active(dt)
                ]
            else:
                continue
            candidates = [
                device_type
                for device_type in candidates
                if device_type.housing_material == parameters.distributor_housing_material
            ]
            for dt in candidates:
                new_top = node.topology.clone()
                child = Node(new_top)
                child.device_type = dt
                child.local_area = local_area
                node.add_child(child)
                child.costs = [None for _ in self.cost_function_list]
                sig_hash = hash(child)
                if sig_hash in self.seen_hashes:
                    child.is_duplicate = True
                    continue
                self.seen_hashes.add(sig_hash)
                self.n_analysis += 1
                connections_before = len(child.topology.connections)
                adapters_uuids_before = [
                    dev.uuid
                    for dev in child.topology.devices.values()
                    if is_adapter(dev.device_type) or is_t_splitter(dev)
                ]
                adapter_count_before = len(adapters_uuids_before)
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
                    continue
                connections_after = len(child.topology.connections)
                adapter_count_after = len([
                    dev.uuid
                    for dev in child.topology.devices.values()
                    if is_adapter(dev.device_type) or is_t_splitter(dev)
                ])
                calculate_costs(child, self.cost_function_list)
                if (
                    connections_before == connections_after
                    and adapter_count_before == adapter_count_after
                ):
                    child.no_new_connections = True
                    continue
                if needs_further_connecting(new_top, ignore_cabinet_connections=False):
                    pass
                else:
                    child.is_fully_connected = True
                children.append(child)
        return children

def normalize_costs(root: Node, weights: Optional[List[float]] = None):
    n_costs = len(weights) if weights else len(root.costs)
    if weights is None:
        weights = [1.0 / n_costs for _ in range(n_costs)]
    else:
        weights_sum = sum(weights)
        weights = [weights[i] / weights_sum for i in range(n_costs)]
    min_costs = [float("inf")] * n_costs
    max_costs = [float("-inf")] * n_costs
    def visit_node(node: Node):
        for i in range(n_costs):
            if node.costs[i] is not None:
                min_costs[i] = min(node.costs[i], min_costs[i])
                max_costs[i] = max(node.costs[i], max_costs[i])
        for child in node.children:
            visit_node(child)
    visit_node(root)
    def normalize_node(node: Node):
        node.normalized_costs = []
        for i in range(n_costs):
            if node.costs[i] is None:
                node.normalized_costs.append(None)
            else:
                b = max_costs[i] - min_costs[i]
                if b > 0:
                    normalized_value = ((node.costs[i] - min_costs[i]) / b) * 100
                    node.normalized_costs.append(normalized_value)
                else:
                    node.normalized_costs.append(50.0)
        for child in node.children:
            normalize_node(child)
    normalize_node(root)
    def weight_costs(node: Node):
        factor = sum(
            [weights[i] for i in range(n_costs) if node.normalized_costs[i] is not None]
        )
        if factor > 0:
            node.normalized_cost = (
                sum(
                    [
                        node.normalized_costs[i] * weights[i]
                        for i in range(n_costs)
                        if node.normalized_costs[i] is not None
                    ]
                )
                / factor
            )
        else:
            node.normalized_cost = 1e6
        for child in node.children:
            weight_costs(child)
    weight_costs(root)

def get_best_node(
    root: Node,
    weights: Optional[List[float]] = None,
    include_not_terminal_nodes: bool = False,
) -> Optional[Node]:
    normalize_costs(root, weights)
    node_candidates = (
        find_terminal_nodes(root)
        if not include_not_terminal_nodes
        else get_all_nodes_in_tree_except_root(root)
    )
    if len(node_candidates) == 0:
        return None
    sorted_nodes = sorted(
        node_candidates, key=lambda node: node.normalized_cost, reverse=False
    )
    return sorted_nodes[0]

def find_terminal_nodes(
    node: Node,
) -> List[Node]:
    if node.is_fully_connected:
        return [node]
    else:
        current_nodes = []
        for child in node.children:
            current_nodes.extend(find_terminal_nodes(child))
        return current_nodes

def get_all_nodes_in_tree_except_root(root: Node) -> List[Node]:
    nodes = []
    def append_all_nodes(node: Node, append_root: bool = True):
        append_root and nodes.append(node)
        for child in node.children:
            append_all_nodes(child)
    append_all_nodes(root, False)
    return nodes

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

# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
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