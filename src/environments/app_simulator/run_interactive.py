import sys
import matplotlib.pyplot as plt


def run_interactive(app):
    """
    Run app in human interactive mode
    This will open a window that allows the user to click on UI elements
    Actions can also be taken by providing strings in the command prompt
    """
    def on_click(event):
        if event.xdata is None or event.ydata is None:
            # clicked outside of plot
            return

        obs = app.get_observation()
        action = obs.get_action_at_click(event.xdata, event.ydata)
        if action:
            print(f"\nClicked on: {action}")
            obs, reward, done, info = app.step(action)
            plt.imshow(obs.get_screenshot())
            plt.show()

    plt.ion()
    plt.axis("off")
    plt.connect("button_press_event", on_click)
    plt.connect("close_event", lambda _: sys.exit(0))

    obs, info = app.reset()
    while True:
        plt.imshow(obs.get_screenshot())
        plt.show()

        available_actions = info["valid_actions"]
        print(f"\nAvailable actions: {available_actions}")
        action = input("Enter action: ")

        if not action or action == "quit":
            print("Quitting app")
            break
        elif action == "reset":
            print("Resetting app")
            obs, info = app.reset()
        elif action not in available_actions:
            print(f"Invalid action: {action}")
        else:
            obs, reward, done, info = app.step(action)
