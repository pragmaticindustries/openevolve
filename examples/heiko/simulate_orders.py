import random
from dataclasses import dataclass
from typing import List, Callable

import simpy

# Initialize all machine "names"

KOMAX_AUTO = "Komax Auto"
KOMAX_KLEIN = "Komax Klein"
CRIMPEN_3_4_AUTO = "Crimpen 3/4 Auto"
CRIMPEN_5_AUTO = "Crimpen 5 Auto"
CRIMPEN_3_4_KLEIN = "Crimpen 3/4 Klein"
UMSPRITZEN_AUTO = "Umspritzen Auto"
UMSPRITZEN_KLEIN = "Umspritzen Klein"
CRIMPEN_LED = "Crimpen LED"
UMSPRITZEN_LED = "Umspritzen LED"


@dataclass
class Auftrag:
    """Repräsentiert einen Produktionsauftrag"""
    id: int
    process_steps: List[str]  # List of Machines that need to be passed
    ankunftszeit: float = 0
    menge: int = 1  # Anzahl der produzierten Einheiten, Standard ist 1
    led: bool = True
    pins: int = 4  # Anzahl der Pins, Standard ist 4
    rohmaterial: str = ""
    expected_end_time: float = 0  # Erwartete Endzeit des Auftrags, wird später gesetzt
    actual_end_time: float = 0  # Tatsächliche Endzeit des Auftrags, wird später gesetzt


class BaseMachine:

    def __init__(self, env, name, setup_condition: Callable[[Auftrag, Auftrag], bool], setup_time_s: int,
                 processing_time_s: Callable[[Auftrag], int]):
        self.env = env
        self.ressource = simpy.Resource(env)
        self.name = name
        self.setup_condition = setup_condition  # if it returns true, setup is needed
        self.setup_time = setup_time_s
        self.processing_time = processing_time_s  # Funktion zur Berechnung der Bearbeitungszeit
        self.recent_order = None  # Zuletzt bearbeiteter Auftrag
        self.activity_log = []  # Liste zur Auslastung der Maschine
        self.waiting_queue = 0  # Anzahl der Aufträge in der Warteschlange vor dieser Maschine
        self.waiting_queue_log = []  # Anzahl der Aufträge in der Warteschlange vor dieser Maschine

    def new_order(self, order: Auftrag):
        """Bearbeitet einen neuen Auftrag"""

        # print(f"{self.name} nimmt Auftrag {order.id} an mit Menge {order.menge} und Pins {order.pins}")
        with self.ressource.request() as request:
            self.waiting_queue += 1
            self.waiting_queue_log.append((self.env.now, self.waiting_queue))
            yield request
            self.waiting_queue -= 1
            self.waiting_queue_log.append((self.env.now, self.waiting_queue))

            maschine_start = self.env.now

            # Prüfen ob eine Umrüstung nötig ist
            if self.recent_order is None or self.recent_order.rohmaterial != order.rohmaterial:
                # print(
                #     f"{self.name} rüstet um von {self.recent_order.rohmaterial if self.recent_order else 'keinem'} auf {order.rohmaterial} um")
                yield self.env.timeout(self.setup_time)

            self.recent_order = order

            # print(f"{self.name} startet Auftrag {order.id} um {self.env.now:.1f}")
            processing_time = self.processing_time(order)
            yield self.env.timeout(processing_time)

            # print(f"{self.name} beendet Auftrag {order.id} um {self.env.now:.1f}")

            machine_end = self.env.now

            # Update auslastung
            self.activity_log.append(machine_end - maschine_start)


class Produktionslinie2:

    def __init__(self, env):
        self.env = env
        never_setup = lambda o, n: False
        self.maschinen = {
            KOMAX_AUTO: BaseMachine(env, KOMAX_AUTO,
                                    setup_condition=lambda old, new: old is None or old.material != new.material,
                                    setup_time_s=5 * 60, processing_time_s=lambda order: 20 * order.menge),
            KOMAX_KLEIN: BaseMachine(env, KOMAX_KLEIN,
                                     setup_condition=lambda old, new: old is None or old.material != new.material,
                                     setup_time_s=5 * 60, processing_time_s=lambda order: 20 * order.menge),
            CRIMPEN_3_4_AUTO: BaseMachine(env, CRIMPEN_3_4_AUTO, setup_condition=never_setup, setup_time_s=1 * 60,
                                          processing_time_s=lambda order: 60 * order.menge),
            CRIMPEN_5_AUTO: BaseMachine(env, CRIMPEN_5_AUTO, setup_condition=never_setup, setup_time_s=1 * 60,
                                        processing_time_s=lambda order: 90 * order.menge),
            CRIMPEN_3_4_KLEIN: BaseMachine(env, CRIMPEN_3_4_KLEIN, setup_condition=never_setup, setup_time_s=1 * 60,
                                           processing_time_s=lambda order: 80 * order.menge),
            UMSPRITZEN_AUTO: BaseMachine(env, UMSPRITZEN_AUTO, setup_condition=never_setup, setup_time_s=2 * 60,
                                         processing_time_s=lambda order: 60 * order.menge),
            UMSPRITZEN_KLEIN: BaseMachine(env, UMSPRITZEN_KLEIN, setup_condition=never_setup, setup_time_s=2 * 60,
                                          processing_time_s=lambda order: 60 * order.menge),
            CRIMPEN_LED: BaseMachine(env, CRIMPEN_LED, setup_condition=never_setup, setup_time_s=1 * 60,
                                     processing_time_s=lambda order: 110 * order.menge),
            UMSPRITZEN_LED: BaseMachine(env, UMSPRITZEN_LED, setup_condition=never_setup, setup_time_s=1 * 60,
                                        processing_time_s=lambda order: 120 * order.menge),
        }
        self.end_time = 0  # Ende der Simulation in Minuten

    def process_order(self, order: Auftrag):
        """Bearbeitet einen Auftrag durch alle Prozessschritte"""
        for step in order.process_steps:
            if step in self.maschinen:
                yield self.env.process(self.maschinen[step].new_order(order))
            # If the order is done, set the end time there
        order.actual_end_time = self.env.now
        self.end_time = self.env.now


# EVOLVE-BLOCK-START

def process_planning(all_orders: List[Auftrag]) -> List[Auftrag]:
    for order in all_orders:
        if order.menge > 20:
            order.process_steps = [KOMAX_AUTO]
            if order.pins == 3 or order.pins == 4:
                order.process_steps += [CRIMPEN_3_4_AUTO, UMSPRITZEN_AUTO]
            else:
                order.process_steps += [CRIMPEN_5_AUTO, UMSPRITZEN_AUTO]
        else:
            order.process_steps = [KOMAX_KLEIN]
            if order.pins == 3 or order.pins == 4:
                order.process_steps += [CRIMPEN_3_4_KLEIN, UMSPRITZEN_KLEIN]
            else:
                order.process_steps += [CRIMPEN_5_AUTO, UMSPRITZEN_KLEIN]

        # Add LED processing if needed
        if order.led:
            order.process_steps += [CRIMPEN_LED, UMSPRITZEN_LED]
    return all_orders


# EVOLVE-BLOCK-END


def simulate_process(n_orders):
    """Simuliert die Produktion von n Aufträgen"""
    # From Scratch
    all_orders = [
        Auftrag(id=i, process_steps=[], menge=random.randint(1, 50), led=random.random() < 0.75,
                pins=random.choice([3, 4, 5]),
                rohmaterial=f"Kupfer {i % 3}", expected_end_time=60 * random.randint(30, 90)) for i in range(n_orders)
    ]

    # Process planning for the orders
    try:
        all_orders = process_planning(all_orders)

        # Assign process steps based on the orders
        env = simpy.Environment()

        produktionslinie = Produktionslinie2(env)

        # Now process the aufträge in their order
        for auftrag in all_orders:
            # print(f"Processing Auftrag {auftrag.id} with Menge {auftrag.menge} and Pins {auftrag.pins}")
            env.process(produktionslinie.process_order(auftrag))

        env.run(until=60 * 60 * 24 * 31)

        print("\n=== Zusammenfassung der Aufträge ===")
        # Display the number of processed orders
        print(f"Anzahl bearbeiteter Aufträge: {len(all_orders)}")
        print(
            f"Durchschnittliche Anzahl an Kabeln pro Auftrag: {sum(auftrag.menge for auftrag in all_orders) / len(all_orders):.1f}")
        print(f"Anzahl an Aufträgen <= 20 Kabel: {sum(1 for auftrag in all_orders if auftrag.menge <= 20)}")
        print(f"Anzahl an Aufträgen > 20 Kabel: {sum(1 for auftrag in all_orders if auftrag.menge > 20)}")
        print(f"Anzahl an Aufträgen mit 3/4 Pins: {sum(1 for auftrag in all_orders if auftrag.pins in [3, 4])}")
        print(f"Anzahl an Aufträgen mit 5 Pins: {sum(1 for auftrag in all_orders if auftrag.pins == 5)}")
        print(f"Anzahl an Aufträgen mit LED: {sum(1 for auftrag in all_orders if auftrag.led)}")

        print("\n=== Produktionslinie ===")
        # Display the Simulation Time human readable (Day, Hour, Minurtes, Seconds, ..)
        days = int(produktionslinie.end_time // (60 * 60 * 24))
        hours = int((produktionslinie.end_time % (60 * 60 * 24)) // (60 * 60))
        minutes = int((produktionslinie.end_time % (60 * 60)) // 60)
        seconds = int(produktionslinie.end_time % 60)
        print(f"Simulation Time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")
        # print(f"Ende der Simulation: {produktionslinie.end_time:.1f} Sekunden")
        print(f"Maschinen Auslastung:")
        for name, machine in produktionslinie.maschinen.items():
            gesamt_zeit = sum(machine.activity_log)
            print(f"{name}: {gesamt_zeit:.1f} Minuten Auslastung")
            # In Percentages
            auslastung_prozent = (gesamt_zeit / produktionslinie.end_time) * 100
            print(f"{name}: {auslastung_prozent:.1f}% Auslastung")
            # Max Queue Size
            max_queue_size = max(machine.waiting_queue_log, key=lambda x: x[1])[1] if machine.waiting_queue_log else 0
            print(f"{name}: Max Queue Size: {max_queue_size}")


        # Calculate a set of metrics for the production line
        def calculate_metrics():
            total_processing_time = sum(sum(machine.activity_log) for machine in produktionslinie.maschinen.values())
            total_orders = len(all_orders)
            average_efficiency = total_processing_time / produktionslinie.end_time
            cables_per_minute = sum(auftrag.menge for auftrag in all_orders) / (produktionslinie.end_time / 60)
            order_in_time_ratio = sum(1 for order in all_orders if
                        order.actual_end_time <= order.expected_end_time) / total_orders if total_orders > 0 else 0
            return {
                "total_processing_time": total_processing_time,
                "total_orders": total_orders,
                "average_processing_time": total_processing_time / total_orders if total_orders > 0 else 0,
                "longest_queue": max(
                    max(machine.waiting_queue_log, key=lambda x: x[1])[1] if machine.waiting_queue_log else 0
                    for machine in produktionslinie.maschinen.values()
                ),
                "overall_duration": produktionslinie.end_time,
                "average_machine_efficiency": average_efficiency / len(produktionslinie.maschinen),
                "cable_per_minute": cables_per_minute,
                "order_in_time_ratio": order_in_time_ratio,
                "overall_score": cables_per_minute * order_in_time_ratio
            }

        return calculate_metrics()
    except Exception as e:
        print(f"Simulation failed: {str(e)}")
        return {
            "total_processing_time": 0,
            "total_orders": 0,
            "average_processing_time": 0,
            "longest_queue": 0,
            "overall_duration": 0,
            "overall_score": 0,
            "overall_efficiency": 0,
            "average_machine_efficiency": 0,
            "cable_per_minute": 0,
            "order_in_time_ratio": 0,
        }
