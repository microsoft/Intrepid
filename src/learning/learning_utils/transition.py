class TransitionDatapoint:

    def __init__(self, curr_obs, action, next_obs, y, curr_state, next_state, action_prob, policy_index, step, reward):
        """
        :param curr_obs: Current observation on which action is taken
        :param action: Action that was taken on current observation
        :param next_obs: The observation that is "alleged" to be observed after taking the action
        :param y: If y=1 then the observation was observed otherwise it is a candidate imposter
        :param curr_state: State for the curr_obs
        :param next_state: State for the next_obs
        :param action_prob: Probability with which action was taken
        :param policy_index: If the curr_obs was generated using a set of policies then policy_index is the index of
                             the policy that is used to observe curr_obs
        :param step: The timestep at which curr_obs was observed. We start with timestep of 1.
        :param reward: reward for taking this transition. If the transition is imposter (y=1) then this reward
                        value maybe incorrect.
        """
        self.curr_obs = curr_obs
        self.action = action
        self.next_obs = next_obs
        self.y = y
        self.curr_state = curr_state
        self.next_state = next_state
        self.action_prob = action_prob
        self.policy_index = policy_index
        self.step = step
        self.reward = reward

        # Meta dictionary contains temporary solution associated with this datapoint.
        # These are not copied on making a copy.
        self.meta_dict = dict()

    def is_valid(self):
        return self.y

    def get_curr_obs(self):
        return self.curr_obs

    def get_curr_state(self):
        return self.curr_state

    def get_action(self):
        return self.action

    def get_action_prob(self):
        return self.action_prob

    def get_next_obs(self):
        return self.next_obs

    def get_next_state(self):
        return self.next_state

    def get_policy_index(self):
        return self.policy_index

    def get_timestep(self):
        return self.step

    def get_reward(self):
        return self.reward

    def make_copy(self):

        return TransitionDatapoint(
            curr_obs=self.curr_obs,
            action=self.action,
            next_obs=self.next_obs,
            y=self.y,
            curr_state=self.curr_state,
            next_state=self.next_state,
            action_prob=self.action_prob,
            policy_index=self.policy_index,
            step=self.step,
            reward=self.reward
        )
