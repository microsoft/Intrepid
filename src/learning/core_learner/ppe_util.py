import os
import imageio
import numpy as np

from environments.intrepid_env_meta.environment_keys import EnvKeys
from learning.learning_utils.count_probability import CountProbability
from learning.learning_utils.evaluate_state_decoder import EvaluateStateDecoder


class PPEDebugger:
    """
        Debugger for PPE algorithm (Efroni et al., 2021)
    """

    def __init__(self, exp_setup):

        self.config = exp_setup.config
        self.constants = exp_setup.constants
        self.logger = exp_setup.logger
        self.experiment = exp_setup.experiment

        # Evaluate state decoder
        self.evaluate_state_decoder = EvaluateStateDecoder(exp_setup)

        self.eval_sample_size = exp_setup.constants["eval_homing_policy_sample_size"]

    # TODO the next function is a duplicate
    @staticmethod
    def _follow_path(env, path):

        obs, info = env.reset(generate_obs=False)
        reward = None
        path_len = path.num_timesteps()

        for h in range(0, path_len):
            action = path.sample_action(obs, h)
            obs, reward, done, info = env.step(action, generate_obs=(h == path_len - 1))

        return obs, info[EnvKeys.ENDO_STATE], reward

    @staticmethod
    def _calc_gap(i, j, dataset, state_stats, path_map):

        gap = 0
        for dp in dataset:
            _, path_id, endo_state, _ = dp
            gap += abs(state_stats[endo_state].get_prob_entry(path_map[i].path_id) -
                       state_stats[endo_state].get_prob_entry(path_map[j].path_id))
        gap /= float(len(dataset))

        return gap

    def _save_fig(self, h, dataset, path_common, path_map):

        base_folder = "%s/obs_figures/step_%d" % (self.experiment, h)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        backward_map = dict()
        for path_id, other_path_ids in path_common.items():
            for other_path_id in other_path_ids:
                backward_map[other_path_id] = path_id

        for path_id in range(0, len(path_map)):
            parent_path_id, action = path_map[path_id].parent_path_id, path_map[path_id].action
            for dp in dataset:
                if dp[1] == path_id:

                    if not os.path.exists("%s/parent_%d" % (base_folder, parent_path_id)):
                        os.makedirs("%s/parent_%d" % (base_folder, parent_path_id))
                    img = (dp[0] * 255).astype(np.uint8)

                    if path_id in path_common:
                        imageio.imwrite("%s/selected_path_number_%d.png" % (base_folder, path_id), img)

                        imageio.imwrite("%s/parent_%d/action_%d_path_number_%d_selected.png" %
                                        (base_folder, parent_path_id, action, path_id), img)
                    else:
                        merged_path_id = backward_map[path_id]     # The path with which it is merged
                        merged_path = path_map[merged_path_id]
                        imageio.imwrite("%s/parent_%d/action_%d_path_number_%d_merged_"
                                        "with_id_%d_parent_%d_action_%d.png" %
                                        (base_folder, parent_path_id, action, path_id, merged_path_id,
                                         merged_path.parent_path_id, merged_path.action), img)
                    break

    def _save_sim_mat(self, path_map, prob, step):

        sim_mat = np.zeros((len(path_map), len(path_map)))

        for i in range(0, len(path_map)):
            for j in range(0, len(path_map)):
                sim = abs(prob[:, i] - prob[:, j]).sum().item()
                sim_mat[i, j] = sim

        base_folder = "%s/sim_matrices/" % self.experiment
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        np.savetxt("%s/sim_matrix_%d.out" % (base_folder, step), sim_mat)

    def _evaluate_homing_policy(self, env, paths):

        count = CountProbability()
        for path in paths:
            for _ in range(self.eval_sample_size // len(paths)):
                count.add(self._follow_path(env, path)[1])

        return count.get_probability_dict()

    def _compute_dynamics_error(self, path_map, prob, path_common, env, abstract_to_state_map, step, error_util,
                                dataset, state_stats, elim_param):

        _attr = getattr(env, "calc_step", None)
        if not callable(_attr):
            return

        mappings = dict()
        for i, values in path_common.items():
            for j in values:
                mappings[j] = i

        for i in range(0, len(path_map)):

            old_path_id_1, old_action_1 = path_map[i].parent_path_id, path_map[i].action
            new_state1 = env.calc_step(abstract_to_state_map[old_path_id_1], old_action_1)
            path1 = "%r -> %s -> %r" % (abstract_to_state_map[old_path_id_1],
                                        env.act_to_str(old_action_1),
                                        new_state1)

            for j in range(i + 1, len(path_map)):

                old_path_id_2, old_action_2 = path_map[j].parent_path_id, path_map[j].action
                new_state2 = env.calc_step(abstract_to_state_map[old_path_id_2], old_action_2)
                path2 = "%r -> %s -> %r" % (abstract_to_state_map[old_path_id_2],
                                            env.act_to_str(old_action_2),
                                            new_state2)

                result = error_util.record(new_state1, new_state2, mappings[i], mappings[j])

                sim = abs(prob[:, i] - prob[:, j]).sum().item()

                if result == ErrorUtil.ERROR1:
                    # Error of type 1: two paths are merged which should not have been merged
                    gap = self._calc_gap(i, j, dataset, state_stats, path_map)
                    self.logger.log("[Error 1] Path %s (%s) merged with path %s (%s). "
                                    "Empirical Similarity %f < threshold %f. Expected gap %s." %
                                    (path_map[j], path1, path_map[i], path2, sim, elim_param, gap))

                elif result == ErrorUtil.ERROR2:
                    # Error of type 2: two paths are not merged which should have been merged
                    self.logger.log("[Error 2] Path %s (%s) was not merged with path %s (%s). "
                                    "Empirical Similarity %f >= threshold %f. Expected gap 0." %
                                    (path_map[j], path1, path_map[i], path2, sim, elim_param))

        self.logger.log("Number of states reachable via augmented paths: %d" % len(state_stats))
        for ctr, (state, state_prob) in enumerate(state_stats.items()):
            paths = state_prob.get_entries()
            self.logger.log("%d. State: %r has %d many paths reaching it in new policy cover." %
                            (ctr, state, len([path_id for path_id in paths if path_id in path_common])))

    def _policy_cover_validation(self, env, step, homing_policies):

        _attr_valid_fn = getattr(env, "generate_homing_policy_validation_fn", None)
        policy_cover_validator = None if not callable(_attr_valid_fn) else env.generate_homing_policy_validation_fn()

        if policy_cover_validator is not None:

            state_dist = self._evaluate_homing_policy(env=env, paths=homing_policies[step])
            self.logger.log("Policy Cover state distribution %r" % state_dist)

            if not policy_cover_validator(state_dist, step):
                self.logger.log("Didn't find a useful policy cover for step %r" % step)
                return False
            else:
                self.logger.log("Found useful policy cover for step %r " % step)
                return True

        else:
            return None

    def _state_decoding_acc(self, env, step, state_decoder):

        _attr_cover = getattr(env, "get_perfect_homing_policy", None)
        perfect_cover = None if not callable(_attr_cover) else env.get_perfect_homing_policy(step)

        if perfect_cover is not None:
            perfect_homing_policies = {step: env.get_perfect_homing_policy(step)}
            state_decoder_acc = self.evaluate_state_decoder.evaluate(env, step, perfect_homing_policies, state_decoder)
            self.logger.log("State Decoder Accuracy is %f" % state_decoder_acc)
            return state_decoder_acc
        else:
            return None

    def debug(self, path_map, prob, path_common, env, abstract_to_state_map, step, error_util,
              dataset, state_stats, elim_param, state_decoder, homing_policies):

        metrics = dict()

        # 1. Save path similarity matrix
        self._save_sim_mat(path_map, prob, step)

        # 2. Save figures if possible
        if self.config["feature_type"] == "image":
            self._save_fig(step, dataset, path_common, path_map)

        # 3. Compute dynamics error Error1 and Error2 (only makes sense for deterministic problems)
        self._compute_dynamics_error(path_map, prob, path_common, env, abstract_to_state_map, step, error_util,
                                     dataset, state_stats, elim_param)

        # 4. Compute policy cover accuracy
        metrics["policy_cover_validation"] = self._policy_cover_validation(env, step, homing_policies)

        # 5. Compute state decoding accuracy
        metrics["state_decoder_acc"] = self._state_decoding_acc(env, step, state_decoder)

        return metrics


class ErrorUtil:

    ERROR1, ERROR2, SUCCESS = range(3)

    def __init__(self):
        self.error1 = 0
        self.error2 = 0
        self.num_entries = 0

    def record(self, state1, state2, abstract_state1, abstract_state2):

        self.num_entries += 1

        if abstract_state1 == abstract_state2 and state1 != state2:
            # Error 1: Merging states which should not be merged
            self.error1 += 1
            return ErrorUtil.ERROR1

        elif abstract_state1 != abstract_state2 and state1 == state2:
            # Error 2: Did not merge states which should be merged
            self.error2 += 1
            return ErrorUtil.ERROR2

        else:
            return ErrorUtil.SUCCESS

    def __str__(self):
        return "Error-1: %d (%f %%), Error-2: %d (%f %%), Total entries %d" % \
               (self.error1,
                (self.error1 * 100.0) / float(max(1, self.num_entries)),
                self.error2,
                (self.error2 * 100.0) / float(max(1, self.num_entries)),
                self.num_entries)