import argparse
import numpy as np
from environments.app_simulator.ui import AppObservation, ToggleSwitch, Button, Text
from environments.app_simulator.run_interactive import run_interactive
from environments.intrepid_env_meta.action_type import ActionType
from environments.intrepid_env_meta.intrepid_env_interface import IntrepidEnvInterface


class TreeNavApp(IntrepidEnvInterface):
    def __init__(
        self,
        depth=4,
        num_node_toggle_switches=4,
        num_shared_toggle_switches=4,
        goto_root_action=True,
        unique_action_names=True,
        reset_behavior="keep_state",
        screenshot_size=512,
    ):
        """
        This app consists of a binary tree of nodes, which are numbered like this:
            1
            2 3
            4 5 6 7
            8 9 10 11 12 13 14 15
        Each node has its own toggle switches. Additional switches can be shared across all nodes.

        depth: depth of the tree
        num_node_toggle_switches: number of toggle switches per node of the tree
        num_shared_toggle_switches: number of toggle switches shared across all nodes
        goto_root_action: whether to include a goto root action
        unique_action_names: whether action names are unique per node or identical across all nodes
        reset_behavior: how reset affects toggle switches: "zero" or "random" or "keep_state"
            - "keep_state": do not reset toggle switches, reset becomes equivalent to goto_root
            - "zero": reset all toggle switches to off
            - "random": reset all toggle switches to random values
        """
        self.depth = depth
        self.max_node = 2**depth - 1
        self.num_node_toggle_switches = num_node_toggle_switches
        self.num_shared_toggle_switches = num_shared_toggle_switches
        self.goto_root_action = goto_root_action
        self.unique_action_names = unique_action_names
        self.screenshot_size = screenshot_size

        assert reset_behavior in ["zero", "random", "keep_state"]
        self.reset_behavior = reset_behavior

        self.current_node = 1
        self.node_toggle_switches = np.zeros((self.max_node, self.num_node_toggle_switches), dtype=np.int64)
        self.shared_toggle_switches = np.zeros(self.num_shared_toggle_switches, dtype=np.int64)
        self.cached_observation = None

    def reset(self):
        # always reset page navigation to root node
        self.current_node = 1

        if self.reset_behavior == "zero":
            self.node_toggle_switches = np.zeros((self.max_node, self.num_node_toggle_switches), dtype=np.int64)
            self.shared_toggle_switches = np.zeros(self.num_shared_toggle_switches, dtype=np.int64)
        if self.reset_behavior == "random":
            self.node_toggle_switches = np.random.randint(
                2, size=(self.max_node, self.num_node_toggle_switches), dtype=np.int64
            )
            self.shared_toggle_switches = np.random.randint(2, size=self.num_shared_toggle_switches, dtype=np.int64)
        if self.reset_behavior == "keep_state":
            # do nothing
            pass

        # invalidate cached observation
        self.cached_observation = None

        obs = self.get_observation()
        info = self.get_info()
        return obs, info

    def step(self, action):
        rewards = []

        if "goto_parent" in action:
            self.current_node = max(self.current_node // 2, 1)
        elif "goto_left_child" in action:
            self.current_node = min(self.current_node * 2, self.max_node)
        elif "goto_right_child" in action:
            self.current_node = min(self.current_node * 2 + 1, self.max_node)
        elif "goto_root" in action:
            self.current_node = 1
        elif "node_toggle" in action:
            row_idx = self.current_node - 1  # node 1 is at row 0
            switch_idx = int(action.split("_")[-1])
            self.node_toggle_switches[row_idx, switch_idx] = not self.node_toggle_switches[row_idx, switch_idx]
            rewards.append(
                f"reward_node_{self.current_node}_toggle_{switch_idx}_state_{self.node_toggle_switches[row_idx, switch_idx]}"
            )
        elif "shared_toggle" in action:
            switch_idx = int(action.split("_")[-1])
            self.shared_toggle_switches[switch_idx] = not self.shared_toggle_switches[switch_idx]
            rewards.append(f"reward_shared_toggle_{switch_idx}_state_{self.shared_toggle_switches[switch_idx]}")
        else:
            raise ValueError(f"Invalid action: {action}")

        # invalidate cached observation
        self.cached_observation = None

        obs = self.get_observation()
        done = False
        info = self.get_info()
        return obs, rewards, done, info

    def get_action_type(self):
        return ActionType.Variable

    def get_ground_truth_state(self):
        return {
            "current_node": self.current_node,
            "node_toggle_switches": self.node_toggle_switches,
            "shared_toggle_switches": self.shared_toggle_switches,
        }

    def get_valid_actions(self):
        # for this app, all actions are clickable
        obs = self.get_observation()
        return obs.get_all_clickable_actions()

    def get_observation(self):
        if self.cached_observation is not None:
            return self.cached_observation

        self.cached_observation = AppObservation(size=self.screenshot_size)

        # to make action names unique per node, we prefix them with the node number
        action_prefix = f"node_{self.current_node}_" if self.unique_action_names else ""

        # show the current node
        self.cached_observation.add_ui_element(
            Text(self.screenshot_size // 2, self.screenshot_size // 10, f"Current node: {self.current_node}", font_size=36)
        )

        # add page navigation buttons
        font_size = 24
        button_width = self.screenshot_size // 3
        button_height = self.screenshot_size // 8
        self.cached_observation.add_ui_element(
            Button(
                x_center=self.screenshot_size // 4,
                y_center=self.screenshot_size // 3,
                width=button_width,
                height=button_height,
                action=action_prefix + "goto_parent",
                text="Parent",
                font_size=font_size,
                enabled=self.current_node > 1,
            )
        )
        self.cached_observation.add_ui_element(
            Button(
                x_center=3 * self.screenshot_size // 4,
                y_center=self.screenshot_size // 3,
                width=button_width,
                height=button_height,
                action=action_prefix + "goto_root",
                text="Root",
                font_size=font_size,
                enabled=self.goto_root_action,
            )
        )
        self.cached_observation.add_ui_element(
            Button(
                x_center=self.screenshot_size // 4,
                y_center=self.screenshot_size // 2,
                width=button_width,
                height=button_height,
                action=action_prefix + "goto_left_child",
                text="Left child",
                font_size=font_size,
                enabled=self.current_node * 2 <= self.max_node,
            )
        )
        self.cached_observation.add_ui_element(
            Button(
                x_center=3 * self.screenshot_size // 4,
                y_center=self.screenshot_size // 2,
                width=button_width,
                height=button_height,
                action=action_prefix + "goto_right_child",
                text="Right child",
                font_size=font_size,
                enabled=self.current_node * 2 <= self.max_node,
            )
        )

        # add toggle switches
        total_switches = self.num_node_toggle_switches + self.num_shared_toggle_switches
        n_col = int(np.ceil(np.sqrt(2 * total_switches)))
        spacing = self.screenshot_size // (2 * n_col)
        height = int(spacing * 0.8)

        for i in range(total_switches):
            x = (i % n_col * 2 + 1) * spacing
            y = (i // n_col + 1) * spacing + (2 * self.screenshot_size // 3)

            if i < self.num_node_toggle_switches:
                state = self.node_toggle_switches[self.current_node - 1, i]
                action = action_prefix + f"node_toggle_{i}"
            else:
                state = self.shared_toggle_switches[i - self.num_node_toggle_switches]
                action = action_prefix + f"shared_toggle_{i - self.num_node_toggle_switches}"

            self.cached_observation.add_ui_element(ToggleSwitch(x, y, height, action=action, display_state=state))

        return self.cached_observation

    def get_info(self):
        return {
            "valid_actions": self.get_valid_actions(),
            "ground_truth_state": self.get_ground_truth_state(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--node_switches", type=int, default=5)
    parser.add_argument("--shared_switches", type=int, default=5)
    parser.add_argument("--no_goto_root", action="store_false", dest="goto_root")
    parser.add_argument("--no_unique_action_names", action="store_false", dest="unique_action_names")
    parser.add_argument("--reset_behavior", choices=["zero", "random", "keep_state"], default="zero")
    args = parser.parse_args()

    app = TreeNavApp(
        depth=args.depth,
        num_node_toggle_switches=args.node_switches,
        num_shared_toggle_switches=args.shared_switches,
        goto_root_action=args.goto_root,
        unique_action_names=args.unique_action_names,
        reset_behavior=args.reset_behavior,
        screenshot_size=512,
    )
    run_interactive(app)
