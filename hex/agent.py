import logging as log

from hex.board import Hex


class AgentError(Exception):
    pass


class HexAgent:
    agent_dict = {}
    setup_status = {}

    def __init__(self, name, best_move_func,
                 description='',
                 setup_func=lambda: None):
        self.name = name
        self.description = description
        self.best_move_func = best_move_func
        self.setup_func = setup_func
        if self.name in HexAgent.agent_dict:
            raise AgentError(f'Agent {self.name} already exists')
        HexAgent.agent_dict[self.name] = self
        HexAgent.setup_status[self.name] = False
        log.info(f'Initialised agent {self.name}')

    def best_move(self, state: Hex):
        return self.best_move_func(state)

    def __repr__(self) -> str:
        return f'Agent {self.name} ({self.description})'

    @staticmethod
    def get_agent(name) -> 'HexAgent':
        if name in HexAgent.agent_dict:
            return HexAgent.agent_dict[name]
        else:
            raise AgentError(f'Agent {name} does not exist')

    @staticmethod
    def setup(name, all=False):
        if all:
            for agent in HexAgent.agent_dict:
                if not HexAgent.setup_status[agent]:
                    HexAgent.agent_dict[agent].setup_func()
                    HexAgent.setup_status[agent] = True
                    log.info(f'Agent {agent} successfully set up')
        else:
            agent = HexAgent.agent_dict[name]
            agent.setup_func()
            HexAgent.setup_status[name] = True
            log.info(f'Agent {agent} successfully set up')

    @staticmethod
    def show_agents():
        for agent in HexAgent.agent_dict.values():
            print(agent)
