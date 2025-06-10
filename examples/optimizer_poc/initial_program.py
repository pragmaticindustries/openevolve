# EVOLVE-BLOCK-START
"""Tree search to optimize a topology"""
import logging
import math
import os
import uuid
import time
from datetime import datetime
import tempfile
import subprocess


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

logger = logging.getLogger(__name__)

node_counter = 0


class ProgressDict(TypedDict):
    last_step: str
    progress: str
    time_left: Optional[int]


# Helper Util to call it easily
def apply_generic_optimizer(
    top: IMTopology,
    cost_function_profile: str,
    repository: IMRepository,
    update_task_state: Optional[Callable[[ProgressDict], None]] = None,
):
    # We need the max amount of devices in any local area
    local_areas = set([device.local_area for device in top.devices.values()])

    # For each local areas we calculate the number of distributors we expect and sum them up
    max_depth = (
        sum(
            [
                math.ceil(
                    len(
                        [
                            1
                            for device in top.devices.values()
                            if device.local_area == la
                        ]
                    )
                    / 8
                )
                for la in local_areas
            ]
        )
        + 2
    )

    print("Max search Depth:", max_depth)

    root = Node(top)

    cost_function_list = [
        cost_functions.installation_time_costs,
        cost_functions.device_diversity_factor,
        cost_functions.distributors_total_price,
        cost_functions.system_reserve_costs,
        cost_functions.port_maintenance_time,
    ]
    # initiate the list of cost functions
    calculate_costs(root, cost_function_list)

    device_types = [dt for dt in repository.repository().values() if len(dt.ports) > 3]
    adapters = [dt for dt in repository.repository().values() if len(dt.ports) <= 3]

    # update task state
    if update_task_state:
        update_task_state(
            {
                "last_step": f"Using {len(device_types)} Device Types for Analysis, Suchbaumtiefe: {max_depth}",
                "progress": "0%",
                "time_left": None,
            }
        )

    print(f"Using {len(device_types)} Device Types for Analysis")

    #
    # cost_function_profile == "one"

    print(f"cost_function_profile {cost_function_profile}")

    # Here we decide how to weight the cost functions
    # 1 [installation_time_costs,
    # 2 device_diversity_factor,
    # 3 distributors_total_price,
    # 4 system_reserve_costs,
    # 5 port_maintenance_time]
    if cost_function_profile == "default":  # equal
        weights = [0.0, 0.1, 0.8, 0.0, 0.0]
    elif cost_function_profile == "price":  # distributor total price
        weights = [0.05, 0.05, 0.8, 0.05, 0.05]
    elif cost_function_profile == "diversity":  # diversity
        weights = [0.05, 0.8, 0.05, 0.05, 0.05]
    elif (
        cost_function_profile == "three"
    ):  # installation time/maintenance time plays role
        weights = [0.4, 0.05, 0.05, 0.05, 0.4]
    else:
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # all equal #TODO recheck

    optimization_mode = False

    traveller = NodeEvaluator(
        {dt.uuid: dt for dt in device_types},
        cost_function_list,
        include_not_fully_connected=optimization_mode,
        draw_debug_tree=False,
        adapters=adapters,
        update_task_state=update_task_state,
    )

    traveller.traverse_breadth_first(
        root, max_depth, print_steps=True, debug_weights=weights
    )
    best_node = get_best_node(
        root, weights=weights, include_not_terminal_nodes=optimization_mode
    )

    return best_node


class Node:
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
        # Device that was added here
        self.device_type = None
        # Local area where the device was placed
        self.local_area = None
        self.is_duplicate = False
        self.no_new_connections = False
        self.is_fully_connected = False
        self.exception = False

    def is_terminal(self) -> bool:
        """
        Returns true if it is for whatever reason a terminal node
        """
        return (
            self.is_duplicate
            or self.no_new_connections
            or self.is_fully_connected
            or self.exception
        )

    def uuid_dict(self):
        devices = {}
        current_node = self
        while current_node.parent is not None:
            # root node has no device type
            if current_node.device_type is not None:
                if (
                    devices.get(
                        (current_node.local_area, current_node.device_type.uuid)
                    )
                    is not None
                ):
                    devices[
                        (current_node.local_area, current_node.device_type.uuid)
                    ] += 1
                else:
                    devices[
                        (current_node.local_area, current_node.device_type.uuid)
                    ] = 1
            else:
                devices[(current_node.local_area, None)] = 1
            current_node = current_node.parent

        return frozenset([(k, v) for k, v in devices.items()])

    def __hash__(self):
        return self.uuid_dict().__hash__()

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
    Class that provides ways to traverse the node
    """

    def __init__(
        self,
        device_types: Dict[uuid.UUID, IMDeviceType],
        cost_function_list,
        include_not_fully_connected: bool = False,
        draw_debug_tree: bool = False,
        adapters: List[IMDeviceType] = None,
        update_task_state: Optional[Callable[[ProgressDict], None]] = None,
    ) -> None:
        self.n_analysis = 0
        self.visited_nodes = set()
        self.device_types = device_types
        self.cost_function_list = cost_function_list
        self.include_not_fully_connected = include_not_fully_connected
        self.draw_debug_tree = draw_debug_tree
        self.update_task_state = update_task_state
        self.adapter_repository = InMemoryRepository()
        for adapter in adapters or {}:
            self.adapter_repository.add(adapter)

    def traverse_breadth_first(
        self,
        root: Node,
        max_depth: int = 5,
        print_steps: bool = False,
        debug_weights: Optional[List[float]] = None,
    ):
        cumulative_duration = 0
        update_time_ns = int(1.5e9)  # 3 seconds

        if print_steps:
            folder = datetime.now().strftime("search-%Y%m%d%H%M%S")
            os.makedirs(f"/tmp/{folder}", exist_ok=True)
            os.makedirs(f"/tmp/{folder}/neato", exist_ok=True)
        for d in range(max_depth):
            # First, we find all edge nodes that are NOT terminal
            def find_active_edges(
                node: Node,
            ) -> List[Node]:
                if len(node.children) == 0:
                    # Edge!
                    if node.is_terminal():
                        # Inactive
                        return []
                    else:
                        # Active
                        return [node]
                else:
                    current_nodes = []
                    for child in node.children:
                        current_nodes.extend(find_active_edges(child))
                    return current_nodes

            edges = find_active_edges(root)
            print(
                f"Expanding at depth {d + 1}/{max_depth} ({100.0 * d / max_depth:.1f}%), inspecting {len(edges)} active edges"
            )
            if print_steps:
                # calculate_costs(root, self.cost_function_list)
                best_node = get_best_node(
                    root,
                    debug_weights,
                    include_not_terminal_nodes=self.include_not_fully_connected,
                )
                if best_node is not None:
                    print(f"We have a best Node: {best_node.node_id}")
                    print(f"Weights of cost functions: {debug_weights} ")
                else:
                    print("Sadly no best Node")
                # experimental_tree.tree_to_graphviz(root, best_node.node_id)
                if self.draw_debug_tree:
                    with tempfile.NamedTemporaryFile() as tmp:
                        # Open the file for writing.
                        with open(tmp.name, "w") as f:
                            graphviz_code = tree_to_graphviz(
                                root, best_node.node_id if best_node else Node
                            )
                            f.write(
                                graphviz_code
                            )  # where `stuff` is, y'know... stuff to write (a string)
                        print(f"I have written my stuff to {tmp.name}")
                        print("====")
                        print(graphviz_code)
                        print("====")

                        subprocess.run(
                            [
                                "dot",
                                "-Tpdf",
                                f"{tmp.name}",
                                "-o",
                                f"/tmp/{folder}/step_{d}.pdf",
                            ],
                        )
                        subprocess.run(
                            [
                                "dot",
                                "-Tpdf",
                                f"{tmp.name}",
                                "-o",
                                f"/tmp/{folder}/neato/step_neato_{d}.pdf",
                                "-Kneato",
                                "-Goverlap=false",
                            ],
                        )

            start = time.time_ns()
            last_update_time = time.time_ns()
            # Second, we expand all of them once
            for num, edge in enumerate(edges):
                # update task state
                current_time = time.time_ns()
                if self.update_task_state and (
                    num == 0 or current_time - last_update_time > update_time_ns
                ):
                    last_update_time = current_time
                    progress = 100 * num / len(edges)
                    time_left = (
                        (current_time - start) / progress * (100 - progress)
                        if num != 0
                        else None
                    )
                    self.update_task_state(
                        {
                            "last_step": f"Depth: {d + 1}/{max_depth}",
                            "progress": f"{progress:.2f}%",
                            "time_left": (
                                int(time_left / 1e9) if time_left is not None else None
                            ),
                        }
                    )
                # do depth first with depth 1
                self.traverse_depth_first(edge, 1)
                print(".", end="")
            print("")

            end = time.time_ns()

            cumulative_duration += end - start

            duration = (end - start) / 10**9

            print(f"Step Took {duration:.1f}s")
            total_duration_ms = cumulative_duration / 10**6
            print(
                f"Total Search took {total_duration_ms / 10 ** 3:.1f}s for {self.n_analysis} Analysis"
            )
            if self.n_analysis > 0:
                print(
                    f"Duration per analysis {total_duration_ms / self.n_analysis:.1f}ms"
                )

        # total_duration_ms = cumulative_duration / 10**6
        #
        # print(
        #     f"Total Search took {total_duration_ms/10**3:.1f}s for {self.n_analysis} Analysis"
        # )
        # print(f"Duration per analysis {total_duration_ms/self.n_analysis:.1f}ms")

    def traverse_depth_first(
        self,
        current_node: Node,
        max_depth: int = 5,
        current_depth: int = 0,
    ):
        if current_depth >= max_depth:
            return
        self.visited_nodes.add(current_node)

        # First, add children

        # Know all local areas
        local_areas = set(
            [device.local_area for device in current_node.topology.devices.values()]
        )
        for local_area in local_areas:
            # Filter for only device types that apply in this local area
            parameters = current_node.topology.get_local_area_parameter(local_area)

            # print(f"Local Area {local_area} has the following parameters:")
            # print("Strategy:", parameters.optimization_strategy)
            # print("Mateiral:", parameters.distributor_housing_material)

            def is_active(device_type: IMDeviceType) -> bool:
                """
                Returns true if the device is active
                The current check if it has NO channels in any port / pin
                """
                for port in device_type.ports.values():
                    for pin in port.pins.values():
                        if pin.channel is not None:
                            return False
                return True

            if parameters.optimization_strategy == OptimizationStrategy.ACTIVE:
                device_types = [
                    dt
                    for dt in self.device_types.values()
                    # Check if this is an active device
                    if (is_active_distributor(dt) or is_iol_hub(dt))
                ]
            elif parameters.optimization_strategy == OptimizationStrategy.PASSIVE:
                device_types = [
                    dt
                    for dt in self.device_types.values()
                    # Check if this is an active device
                    if not is_active(dt)
                ]

                # Also check Main Cable Interface
                # TODO Activate as soon as DB is up to date
                # device_types = [
                #     dtnormalized_cost
                #     for dt in device_types
                #     if dt.distributor_main_cable_interface
                #     == parameters.distributor_main_cable_interface
                # ]

            else:
                raise RuntimeError("Unknown Optimization Strategy")

            # Filter for Housing Material
            device_types = [
                device_type
                for device_type in device_types
                if device_type.housing_material
                == parameters.distributor_housing_material
            ]

            # FIXME, Remove later
            # if current_depth == 0:
            #    device_types.append(None)

            # print(f"Using the following device types in local area {local_area}:")
            # for dt in device_types:
            #     print(f" - {dt.identifier}")
            for dt in device_types:
                # Now we apply the main flow
                new_run = current_node.topology.clone()

                child = Node(new_run)
                child.device_type = dt
                child.local_area = local_area
                current_node.add_child(child)
                child.costs = [None for _ in self.cost_function_list]

                # Check if child is unique
                if child in self.visited_nodes:
                    # Do not expand duplicates, we also save time for the evaluation here
                    logger.debug("This setup is duplicate, no calculation is done here")
                    child.is_duplicate = True
                    continue

                self.visited_nodes.add(child)

                self.n_analysis += 1

                # Check if any connections where made
                connections_before = len(child.topology.connections)
                adapters_uuids_before = [
                    dev.uuid
                    for dev in child.topology.devices.values()
                    if is_adapter(dev.device_type) or is_t_splitter(dev)
                ]
                adapter_count_before = len(adapters_uuids_before)

                # child.topology.connections.clear()
                child.topology.clear_connections()
                # remove placed adapters
                for uid in adapters_uuids_before:
                    child.topology.remove_device(uid)

                # Device.objects.create(
                #     identifier=f"Distributor {uuid.uuid4()}",
                #     topology=child.run.topology,
                #     local_area=child.run.local_areas.first(),
                #     device_type=DeviceType.objects.get(uuid=dt),
                # )

                identifier_uuid = uuid.uuid4()
                # print(
                #     "Adding distributor", dt.identifier, " in local area ", local_area
                # )
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

                    # Connect all so far unconnected devices to cabinet (EXPERIMENTAL!!!!)
                    # So we can compare this installation  traveller.traverse_breadth_first(root, max_depth, print_steps=True)
                    # --------------------------
                    # if self.include_not_fully_connected:
                    #     connect_all_unconnected_periphery_devices_to_cabinet(
                    #         child.topology
                    #     )
                    # --------------------------
                except Exception:
                    child.exception = True
                    continue

                connections_after = len(child.topology.connections)
                adapter_count_after = len(
                    [
                        dev.uuid
                        for dev in child.topology.devices.values()
                        if is_adapter(dev.device_type) or is_t_splitter(dev)
                    ]
                )

                calculate_costs(child, self.cost_function_list)

                if (
                    connections_before == connections_after
                    and adapter_count_before == adapter_count_after
                ):
                    child.no_new_connections = True
                    continue

                # Descend further if not done yet
                if needs_further_connecting(new_run, ignore_cabinet_connections=False):
                    self.traverse_depth_first(child, max_depth, current_depth + 1)
                else:
                    child.is_fully_connected = True


def normalize_costs(root: Node, weights: Optional[List[float]] = None):
    """
    Normalizes the costs
    """
    # First, get the min / max value for each cost component
    n_costs = len(weights) if weights else len(root.costs)
    # Initialize the weights if none are given
    if weights is None:
        weights = [1.0 / n_costs for _ in range(n_costs)]
    else:
        weights_sum = sum(weights)
        weights = [weights[i] / weights_sum for i in range(n_costs)]

    min_costs = [1e6 for _ in range(n_costs)]
    max_costs = [-1e6 for _ in range(n_costs)]

    def visit_node(node: Node):
        for i in range(n_costs):
            if node.costs[i] is not None:
                min_costs[i] = min(node.costs[i], min_costs[i])
                max_costs[i] = max(node.costs[i], max_costs[i])
        for child in node.children:
            visit_node(child)

    visit_node(root)

    # We have the min / max values

    # print("Min Costs", min_costs)
    # print("Max Costs", max_costs)

    # Second, normalize them all between 0 / 1
    def normalize_node(node: Node):
        node.normalized_costs = []
        for i in range(n_costs):
            if node.costs[i] is None:
                node.normalized_costs.append(None)
            else:
                b = max_costs[i] - min_costs[i]
                if b > 0:
                    #
                    # min-max normalization to [0..100] range
                    normalized_value = ((node.costs[i] - min_costs[i]) / b) * 100
                    node.normalized_costs.append(normalized_value)
                else:
                    # set mid-point of [0..100] range
                    node.normalized_costs.append(50.0)
        for child in node.children:
            normalize_node(child)

    normalize_node(root)

    # Calculate the final cost using the weights
    def weight_costs(node: Node):
        # We need to normalize it with the weights of the non-zero values (usually 1)
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
            # costs are missing, assign a very high cost value
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

    # for positive values in range [0..100]
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


def tree_to_graphviz(current_node: Node, best_node_id: Optional[int] = None) -> str:
    s = "digraph G {\n"
    s += _tree_to_graphviz(current_node, best_node_id)
    s += "}\n"
    return s


def _tree_to_graphviz(current_node: Node, best_node_id: Optional[int] = None) -> str:
    s = ""
    if current_node.is_fully_connected:
        thickness = ",penwidth=4.0"
    elif current_node.no_new_connections:
        thickness = ",penwidth=4.0,color=red"
    else:
        thickness = ""
    if best_node_id is not None and current_node.node_id == best_node_id:
        color = ",style=filled,fillcolor=yellow"
    elif current_node.is_duplicate:
        color = ",style=filled,fillcolor=gray"
    elif current_node.exception:
        color = ",style=filled,fillcolor=red"
    else:
        color = ""
    label = f"<{current_node.node_id}>\n{current_node.normalized_cost:.2f}\n"
    label += ",".join([(f"{c:.2f}" if c else "-") for c in current_node.costs]) + "\n"
    label += ",".join(
        [(f"{c:.2f}" if c else "-") for c in current_node.normalized_costs]
    )
    s += f'{current_node.node_id} [shape="circle",label="{label}"{thickness}{color}]\n'
    for child in current_node.children:
        # We add an invisible node for the device
        tmp_node_id = uuid.uuid4().__str__()
        identifier = (
            child.device_type.identifier.replace('"', "")
            if child.device_type
            else "No Device"
        )
        s += f'"{tmp_node_id}" [shape="plaintext", label="{identifier}"]\n'
        s += f'{current_node.node_id} -> "{tmp_node_id}"\n'
        s += f'"{tmp_node_id}" -> {child.node_id}\n'
        s += _tree_to_graphviz(child, best_node_id)
    return s


def needs_further_connecting(
    topology: IMTopology, ignore_cabinet_connections: bool = False
) -> bool:
    """
    Returns true if not all IO Ports are connected yet or are connected to Terminal
    """
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
                # or is connected to terminal
                or (
                    ignore_cabinet_connections
                    and topology.get_connected_counterpart(device.uuid, port.label)
                    == (None, None)
                )
            ):
                return True
    return False


def print_tree(node: Node, prefix: str = ""):
    print(
        prefix
        + f" - {node.normalized_cost}"
        + ("" if needs_further_connecting(node.topology) else "*")
        + " "
        + node.device_type.identifier
        if node.device_type
        else "" + f"({node.costs} / {node.normalized_costs})"
    )
    for child in node.children:
        print_tree(child, prefix + "  ")

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

    # Import Prices
    StaticIMRepository().setup()
    # prices_import.import_prices(repository=StaticIMRepository(), use_dict=True)

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
                # distributor_housing_material=DistributorHousingMaterial.PLASTIC,
                optimization_strategy=OptimizationStrategy.ACTIVE,
            ),
        )
        .build()
    )

    # Measure execution time (here as well)

    import time

    start_time = time.time()
    best_node = apply_generic_optimizer(top, "default", StaticIMRepository())
    end_time = time.time()

    print(f"Execution Time: {end_time - start_time:.2f} seconds")

    if best_node is None:
        print("No best node found, something went wrong.")
        return None

    # best_node.topology.debug_print()

    print(f"Best Node Costs: {best_node.costs}")
    return best_node.costs


if __name__ == "__main__":
    sorted_list = run_sorting(15)
    print("List was sorted successfully.")
