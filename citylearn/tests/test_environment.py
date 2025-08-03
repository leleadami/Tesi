import pickle
import time
import sys
sys.path.insert(0, '..')
from citylearn.citylearn import CityLearnEnv

RESULT_FILEPATH = 'test_environment.pkl'
schema = 'baeda_3dem'

def main():
    # simulation
    env = CityLearnEnv(schema, simulation_end_time_step=500)
    model = env.load_agent()
    model.learn(episodes=1, deterministic_finish=False, logging_level=1)

    print(env.evaluate())

    # # save simulator to file
    # with open(RESULT_FILEPATH, 'wb') as f:
    #     pickle.dump(model, f)

if __name__ == '__main__':
    start_time = time.time()
    main()
    runtime = time.time() - start_time
    print('runtime:',runtime)