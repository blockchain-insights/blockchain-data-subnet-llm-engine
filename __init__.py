"""
Copyright © 2023 Blockchain Insights

Blockchain Insights rights reserved.
Source code produced by Blockchain Insights may not be reproduced, modified, or distributed
without the express permission of Blockchain Insights.

"""

__version__ = "0.1.1"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
