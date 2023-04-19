class CerebralEnvInterface:
    """ Any environment using Cerebral Env Interface must support the following API """

    def reset(self):
        """
            :return:
                obs:        Agent observation. No assumption made on the structure of observation.
                info:       Dictionary containing relevant information such as latent state, etc.
        """

        raise NotImplementedError()

    def step(self, action):
        """
            :param action:
            :return:
                obs:        Agent observation. No assumption made on the structure of observation.
                reward:     Reward received by the agent. No Markov assumption is made.
                done:       True if the episode has terminated and False otherwise.
                info:       Dictionary containing relevant information such as latent state, etc.
        """
        raise NotImplementedError()

    def get_action_type(self):
        """
            :return:
                action_type:     Return type of action space the agent is using
        """
        raise NotImplementedError()

    def save(self, save_path, fname=None):
        """
            Save the environment
            :param save_path:   Save directory
            :param fname:       Additionally, a file name can be provided. If save is a single file, then this will be
                                used else it can be ignored.
            :return: None
        """
        raise NotImplementedError()

    def load(self, load_path, fname=None):
        """
            Save the environment
            :param load_path:   Load directory
            :param fname:       Additionally, a file name can be provided. If load is a single file, then only file
                                with the given fname will be used.
            :return: Environment
        """
        raise NotImplementedError()

    def is_episodic(self):
        """
            :return:                Return True or False, True if the environment is episodic and False otherwise.
        """
        raise NotImplementedError()

    def act_to_str(self, action):
        """
            :param: given an action
            :return: action in string representation
        """

        return "%r" % action
