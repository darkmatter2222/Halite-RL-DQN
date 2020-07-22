from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from helpers import susman_logic

def human_action(observation, configuration):
    try:
        board = Board(observation, configuration)
        current_player = board.current_player

        for ship in current_player.ships:
            res = susman_logic.origional(board)
            print(res)

            field_matrix = susman_logic.render_field_matrix(board)
            print(field_matrix)

            field_matrix = susman_logic.center_field_matrix(board, field_matrix, ship.id)
            print(field_matrix)

            field_matrix_ASCII = susman_logic.render_field_ASCII(board, field_matrix, True)
            print(field_matrix_ASCII)

        return current_player.next_actions
    except Exception as e:
        print(e)
        return current_player.next_actions

board_size = 10
environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
agent_count = 2
environment.reset(agent_count)
state = environment.state[0]
environment.reset(agent_count)
environment.configuration.actTimeout += 10000
environment.run([human_action, "random"])