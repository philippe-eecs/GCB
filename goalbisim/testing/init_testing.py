



def init_testing(details):

	tests = details['tests']

	functions = []

	for test in tests:
		if test == 'trajectory_heat_map':
			from goalbisim.testing.heatmap_test_phi import create_trajectory_color_map
			functions.append(create_trajectory_color_map)

		elif test == 'psi_analogy_test':
			pass #Need to think of how to do

		elif test == 'state_regression':
			from goalbisim.testing.state_regression_test import state_regression_test

			functions.append(state_regression_test)

		elif test == 'analogy_test':
			from goalbisim.testing.analogy_test import analogy_test

			functions.append(analogy_test)

		elif test == 'implicit_analogy_test':
			from goalbisim.testing.implicit_policy import implicit_analogy_test

			functions.append(implicit_analogy_test)

		elif test == 'nn_analogy':
			from goalbisim.testing.analogy_nn_test import nearest_neighbor_analogy

			functions.append(nearest_neighbor_analogy)

		elif test == 'nn_analogy_static':
			from goalbisim.testing.analogy_nn_test_2 import nearest_neighbor_analogy2

			functions.append(nearest_neighbor_analogy2)

		elif test == 'nn_goal_analogy':
			from goalbisim.testing.analogy_goals_test import nearest_neighbor_analogy3

			functions.append(nearest_neighbor_analogy3)

		elif test == 'nn_goal_analogy_complex':
			from goalbisim.testing.analogy_goals_test2 import nearest_neighbor_analogy4

			functions.append(nearest_neighbor_analogy4)

		else:
			raise NotImplementedError


	return functions