class CountBasedEstimation:
    def __init__(self, stationary=False):
        self.stationary = stationary

    def estimate_all(self, replay_memory, decoder):
        raise NotImplementedError()

    def estimate_step(self, mdp, replay_memory, step, decoders):
        if not isinstance(replay_memory, list):
            raise AssertionError("Replay memory must be a list")

        transitions = [episode.get_transitions_at_step(step - 1) for episode in replay_memory]

        latent_transitions = [
            (
                decoders[step - 1].encode_observations(x),
                a,
                r,
                decoders[step].encode_observations(next_x),
            )
            for (x, a, r, next_x) in transitions
        ]

        abs_states = set([lt[3] for lt in latent_transitions])

        for abs_state in abs_states:
            mdp.add_state(abs_state, step)

        for abs_state, action, reward, next_abs_state in latent_transitions:
            mdp.add_transition(abs_state, action, next_abs_state)
            mdp.add_reward(abs_state, action, next_abs_state, reward)
