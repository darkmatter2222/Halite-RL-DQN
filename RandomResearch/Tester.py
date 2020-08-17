from SusmanGameResearch.SusmanGame import SusmanGameEnv

env = SusmanGameEnv()

env.reset()

env.reset_board()

env.set_goal()

env.get_state()

env.get_observations()

env.render()

print(env.step(1))

env.render()