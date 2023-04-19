REQUIRED_CONFIG_KEYS = ["num_actions", "actions", "horizon", "obs_dim", "feature_type", "gamma"]

REQUIRED_CONSTANT_KEYS = ["learning_rate",
                          "num_homing_policy",
                          "encoder_training_num_samples",
                          "encoder_training_epoch",
                          "encoder_training_lr",
                          "encoder_training_batch_size",
                          "validation_data_percent",
                          "psdp_training_num_samples",
                          "cb_oracle_epoch",
                          "cb_oracle_lr",
                          "cb_oracle_batch_size",
                          "eval_homing_policy_sample_size",
                          "reward_free_planner",
                          "reward_sensitive_planner"]


def validate(config, constants):
    # TODO validate based on the algorithm that is being run

    for key in REQUIRED_CONFIG_KEYS:
        assert key in config, "Did not find the key %r in config in dictionary" % key

    for key in REQUIRED_CONSTANT_KEYS:
        assert key in constants, "Did not find the key %r in constants dictionary" % key

    return True
