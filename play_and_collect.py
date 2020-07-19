from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from helpers import render
from helpers import data
from helpers import image


def human_action(observation, configuration):
    try:
        board = Board(observation, configuration)
        current_player = board.current_player

        for ship in current_player.ships:
            render.renderer(board, ship.position)
            ship_directive = input(f"What Direction to Move Ship w/ cargo "
                                   f"{ship.halite} for this step:{observation.step}?")
            if ship_directive != '':
                ship.next_action = SHIP_DIRECTIVES[ship_directive]

            img = data.get_training_data(board,
                                         ship,
                                         board.ships,
                                         board.shipyards,
                                         board_size,
                                         current_player.id,
                                         board.observation["players"][0][0],
                                         ship.halite,
                                         'ship',
                                         board.observation['halite'])
            image.save_image(img, SHIP_DIRECTIVES[ship_directive], root_image_directory)
        return current_player.next_actions
    except Exception as e:
        return current_player.next_actions


root_image_directory = 'N:\\Halite'
board_size = 10
environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
agent_count = 2
environment.reset(agent_count)
state = environment.state[0]
SHIP_DIRECTIVES = {'w': ShipAction.NORTH,
                   'd': ShipAction.EAST,
                   's': ShipAction.SOUTH,
                   'a': ShipAction.WEST,
                   '7': ShipAction.CONVERT,
                   '': 'NOTHING'}

environment.reset(agent_count)
environment.configuration.actTimeout += 10000
environment.run([human_action, "random"])
#environment.render(mode="ipython", width=800, height=600)
