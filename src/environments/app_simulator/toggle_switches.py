import argparse
import numpy as np
from ui import AppObservation, ToggleSwitch
from run_interactive import run_interactive


class ToggleSwitchesApp:
    def __init__(self, num_controllable_switches=1, num_random_switches=99, reset_behavior="keep_state", screenshot_size=512):
        """
        This app consists of a grid of toggle switches, some of which can be controlled
        by the agent and some of which will randomly toggle upon each action taken.

        num_controllable_switches: number of switches that can be controlled by the agent
        num_random_switches: number of switches that are randomly toggled
        reset_behavior: "zero", "random", "keep_state"
            - "keep_state": do not reset toggle switches, reset action does nothing
            - "zero": reset all agent controllable toggle switches to off
            - "random": reset all toggle switches to random values
        """
        self.num_controllable_switches = num_controllable_switches
        self.num_random_switches = num_random_switches
        self.screenshot_size = screenshot_size

        assert reset_behavior in ["zero", "random", "keep_state"]
        self.reset_behavior = reset_behavior

        self.toggle_switches = np.zeros(num_controllable_switches, dtype=np.int64)
        self.cached_observation = None

    def reset(self):
        # invalidate cached observation
        self.cached_observation = None

        if self.reset_behavior == "zero":
            self.toggle_switches = np.zeros(self.num_controllable_switches, dtype=np.int64)
        if self.reset_behavior == "random":
            self.toggle_switches = np.random.randint(2, size=self.num_controllable_switches, dtype=np.int64)
        if self.reset_behavior == "keep_state":
            # do nothing
            pass

        obs = self.get_observation()
        info = self.get_info()
        return obs, info

    def step(self, action):
        # invalidate cached observation
        self.cached_observation = None

        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            raise ValueError(f"Received invalid action: {action}. Valid actions are {valid_actions}")

        if "toggle" in action:
            idx = int(action.split("_")[-1])
            self.toggle_switches[idx] = not self.toggle_switches[idx]

        obs = self.get_observation()
        reward = None
        done = False
        info = self.get_info()
        return obs, reward, done, info

    def get_ground_truth_state(self):
        return self.toggle_switches

    def get_valid_actions(self):
        valid_actions = ["do_nothing"]
        for i in range(self.num_controllable_switches):
            valid_actions.append(self._button_to_action(i))
        return valid_actions

    def get_observation(self):
        if self.cached_observation is not None:
            return self.cached_observation

        self.cached_observation = AppObservation(size=self.screenshot_size)
        total_switches = self.num_controllable_switches + self.num_random_switches
        n_col = int(np.ceil(np.sqrt(total_switches)))
        spacing = self.screenshot_size // n_col
        offset = spacing // 2
        height = int(offset * 0.8)

        for i in range(total_switches):
            x = (i % n_col) * spacing + offset
            y = (i // n_col) * spacing + offset
            state = self.toggle_switches[i] if i < self.num_controllable_switches else np.random.randint(2)
            self.cached_observation.add_ui_element(
                ToggleSwitch(x, y, height, action=self._button_to_action(i), display_state=state)
            )

        return self.cached_observation

    def get_info(self):
        return {
            "valid_actions": self.get_valid_actions(),
            "ground_truth_state": self.get_ground_truth_state(),
        }

    def _button_to_action(self, index):
        if index >= 0 and index < self.num_controllable_switches:
            return f"toggle_{index}"
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllable_switches", type=int, default=10)
    parser.add_argument("--random_switches", type=int, default=90)
    parser.add_argument("--reset_behavior", choices=["zero", "random", "keep_state"], default="zero")
    args = parser.parse_args()

    app = ToggleSwitchesApp(
        num_controllable_switches=args.controllable_switches,
        num_random_switches=args.random_switches,
        reset_behavior=args.reset_behavior,
        screenshot_size=512,
    )
    run_interactive(app)
