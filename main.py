#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
import click

import rewardpredictive as rp

EXPERIMENT_SET_NAME_LIST = [
    'ExperimentSetRepresentationEvaluation',
    'ExperimentSetMazeQLearning',
    'ExperimentSetMazeQTransfer',
    'ExperimentSetMazeSFLearning',
    'ExperimentSetMazeSFTransfer',
    'ExperimentSetMazeMaximizingQLearning',
    'ExperimentSetMazePredictiveQLearning',
    'ExperimentSetMazePredictiveSFLearning',
    'ExperimentSetGuitarSFLearning',
    'ExperimentSetGuitarSFTransfer',
    'ExperimentSetGuitarRewardPredictive',
    'ExperimentSetGuitarSFLearningZeroNegativeReward',
    'ExperimentSetGuitarSFTransferZeroNegativeReward',
    'ExperimentSetGuitarRewardPredictiveZeroNegativeReward',
    'ExperimentSetCycleMDPDatasetPredictive',
    'ExperimentSetCycleMDPDatasetMaximizing',
    'ExperimentSetTaskSequenceRewardChangeQLearning',
    'ExperimentSetTaskSequenceRewardChangeQTransfer',
    'ExperimentSetTaskSequenceRewardChangeSFLearning',
    'ExperimentSetTaskSequenceRewardChangeSFTransfer',
    'ExperimentSetTaskSequenceRewardChangeSFTransferAll',
    'ExperimentSetTaskSequenceRandomRewardChangeQLearning',
    'ExperimentSetTaskSequenceRandomRewardChangeQTransfer',
    'ExperimentSetTaskSequenceRandomRewardChangeSFLearning',
    'ExperimentSetTaskSequenceRandomRewardChangeSFTransfer',
    'ExperimentSetTaskSequenceRandomRewardChangeSFTransferAll'
]
EXPERIMENT_SET_ALPHA_BETA_NAME_LIST = [
    'ExperimentSetMazeMaximizingQLearning',
    'ExperimentSetMazePredictiveQLearning',
    'ExperimentSetMazePredictiveSFLearning',
    'ExperimentSetCycleMDPDatasetPredictive',
    'ExperimentSetCycleMDPDatasetMaximizing'
]


@click.command()
@click.option('-e', '--experiment-set', default=None, type=str, help='Experiment set to run.')
@click.option('-b', '--best', is_flag=True, help='Run only best experiment.')
@click.option('--alpha', default=None, type=float, help='Alpha parameter.')
@click.option('--beta', default=None, type=float, help='Beta parameter.')
def main(experiment_set, best, alpha, beta):
    if alpha is not None and best:
        raise Exception('Cannot specify --alpha and run -b/--best flag at the same time.')
    if beta is not None and best:
        raise Exception('Cannot specify --beta and run -b/--best flag at the same time.')

    if experiment_set is None:
        experiment_set_name_list = EXPERIMENT_SET_NAME_LIST
    else:
        experiment_set_name_list = [experiment_set]

    for experiment_set_name in experiment_set_name_list:
        if experiment_set_name in EXPERIMENT_SET_ALPHA_BETA_NAME_LIST:
            es = rp.construct_experiment_set_by_name(experiment_set_name, alpha=alpha, beta=beta)
        else:
            es = rp.construct_experiment_set_by_name(experiment_set_name)
        if best:
            es.run_best(12345)
        else:
            es.run(12345)

    print('Done.')


if __name__ == "__main__":
    main()
