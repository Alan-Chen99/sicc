"""
Equivalent of `https://stationeers-wiki.com/IC10#Harvie_automation`.

Wiring:
- `d0`: Harvie
- `d1`: Hydroponics tray
"""

from sicc import *
from sicc.devices import *

harvie_pin = d0
tray_pin = d1

# refers to all Harvie
harvie = Harvie()


@program
@transpile
def main():
    while True:
        yield_()

        # Read plant data from the tray slot (slot 0).
        # Harvestable plants return 1, young plants return 0, and nothing planted returns -1.
        mature = tray_pin.slots[0]["Mature"]

        if mature == -1:
            # Stop planting if no seeds are available, otherwise it will plant nothing repeatedly.
            occupied = harvie_pin.slots[0]["Occupied"]
            if not occupied:
                continue

            harvie.Plant = 1
            continue

        if mature == 1:
            harvie.Harvest = 1
            continue

        # Seeds available returns 1, all seeds picked returns 0,
        # and plants too young/old for seeds returns -1.
        seeding = tray_pin.slots[0]["Seeding"]
        if seeding == 1:
            harvie.Harvest = 1


if __name__ == "__main__":
    main.cli()
