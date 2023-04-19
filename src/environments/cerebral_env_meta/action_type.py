class ActionType:

    Discrete = "discrete"
    Continuous = "continuous"
    Structured = "structured"

    @staticmethod
    def get_action_type_from_name(act_type_name):

        if act_type_name == "discrete":
            return ActionType.Discrete

        elif act_type_name == "continuous":
            return ActionType.Continuous

        elif act_type_name == "structured":
            return ActionType.Structured

        else:
            raise AssertionError("No action type found for %r" % act_type_name)
