class EnvKeys:
    """
        meta information returned can use the following keys to make the code more
        generalizable across different environments
    """

    # Counter from which the time step in an episode
    # Designed to deal with starting with 0 vs 1 issue
    INITIAL_TIME_STEP = 0

    # Overall state
    STATE = "state"

    # Endogenous state
    ENDO_STATE = "endogenous_state"

    # Time step
    TIME_STEP = "timestep"
