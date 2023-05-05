import pickle
import numpy as np


def terminate(performance, exp_setup, seeds):
    setting = dict()

    for k, v in exp_setup.config.items():
        setting["config/%s" % k] = v

    for k, v in exp_setup.constants.items():
        setting["constants/%s" % k] = v

    for k, v in exp_setup.args.__dict__.items():
        setting["args/%s" % k] = v

    results = {"setting": setting, "performance": performance, "seeds": seeds}

    # Save performance
    with open("%s/results.pickle" % exp_setup.experiment, "wb") as f:
        pickle.dump(results, f)

    if len(performance) > 0:
        for key in performance[0]:  # Assumes the keys are same across all runes
            if not isinstance(performance[0][key], int) and not isinstance(
                performance[0][key], float
            ):
                continue

            metrics = [performance_[key] for performance_ in performance]

            exp_setup.logger.log(
                "%r: Mean %f, Std %f, Median %f, Min %f, Max %f, Num runs %d, All performance %r"
                % (
                    key,
                    np.mean(metrics),
                    np.std(metrics),
                    np.median(metrics),
                    np.min(metrics),
                    np.max(metrics),
                    len(metrics),
                    metrics,
                )
            )

    exp_setup.logger.log("Experiment Completed.")

    # Cleanup
    exp_setup.logger_manager.cleanup()
