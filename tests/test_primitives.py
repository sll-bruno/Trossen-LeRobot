import unittest

from trossen_experiment.models import Observation, RobotState
from trossen_experiment.primitives import WaypointPrimitive


class PrimitiveTests(unittest.TestCase):
    def test_waypoint_primitive_interpolates_and_holds_last_target(self):
        primitive = WaypointPrimitive(
            [
                {"target": [0.2] * 14, "steps": 2},
                {"target": [0.4] * 14, "steps": 2},
            ]
        )
        observation = Observation(RobotState.from_values([0.0] * 14, 0.0), {}, None, 0.0)
        for expected in (0.1, 0.2, 0.3, 0.4, 0.4):
            action = primitive(observation)
            self.assertTrue(all(abs(value - expected) < 1e-12 for value in action))


if __name__ == "__main__":
    unittest.main()
