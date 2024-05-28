"""
Copyright 2023-2024 Mikael Lund & Ladislav Bartos

Licensed under the Apache license, version 2.0 (the "license");
you may not use this file except in compliance with the license.
You may obtain a copy of the license at

    http://www.apache.org/licenses/license-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the license is distributed on an "as is" basis,
without warranties or conditions of any kind, either express or implied.
See the license for the specific language governing permissions and
limitations under the license.
"""

"""
Script for converting from Martini topology to Faunus topology.
"""

# ruff: noqa: E402
import sys
import martini_openmm  # type: ignore
from topology import FaunusTopology  # type: ignore

try:
    input_file = sys.argv[1]
except IndexError:
    print(
        f"Usage: python {sys.argv[0]} INPUT_MARTINI_TOPOLOGY_FILE > OUTPUT_FAUNUS_TOPOLOGY_FILE"
    )
    sys.exit()

martini_topology = martini_openmm.MartiniTopFile(
    input_file,
    periodicBoxVectors=[[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]],
)
print(FaunusTopology.from_martini(martini_topology).to_yaml())
