class ExperimentalSetup:
    def __init__(
        self,
        config,
        constants,
        experiment,
        exp_name,
        env_name,
        args,
        debug,
        logger,
        logger_manager,
    ):
        """
        :param config: Dictionary containing values for the environment
        :param constants: Dictionary containing hyperparameters for the algorithm
        :param experiment: the full experiment folder where all contents should be saved
        :param exp_name: name of the main experiment log file
        :param env_name: name of the environment
        :param args: command line arguments
        :param debug: if set to true, then run the code in debug mode
        :param logger: Logger for logging data
        :param logger_manager: Logger Manager
        """

        self.config = config
        self.constants = constants
        self.experiment = experiment
        self.exp_name = exp_name
        self.env_name = env_name
        self.base_env_name = env_name.split("/")[-1]
        self.args = args
        self.logger = logger
        self.debug = debug
        self.logger_manager = logger_manager
