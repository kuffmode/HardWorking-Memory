import numpy as np
import pandas as pd
import copy
from typing import Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


# the most fundamental functions, make a sinusoidal time series and stack them to have the reference stimulus
def make_sin(*,
             frequency: int = 10,
             event_duration: int = 500) -> np.ndarray:
    """
    Create a sin wave with the given parameters such that there will be "f" peaks during "t" timepoints.
    So if f is weirdly large (like more than 1 peak per "t") then the signal won't be sinusoidal anymore.
    Same for weirdly small f.
    Args:
        frequency (int):
            Frequency of the sine wave.
            For now stick to a range of [1, t/10]

        event_duration (int):
            Number of time points.
    Returns (np.ndarray):
        A sinusoidal time series with length t.

    TODO: Error handling and exceptions.
    TODO: Check for the f/t ratio.

    """
    return np.sin(2 * np.pi * frequency * np.arange(event_duration) / event_duration)


def make_sin_stimulus(*,
                      frequencies: List,
                      event_duration: int = 500) -> np.ndarray:
    """
    Generates a vector of sinusoidal events given the frequencies by looping through them.
    So the order of events will be the same as the elements in the list.
    Note: it uses the "make_sin" function.

    Args:
        frequencies (List):
            Just as "make_sin", keep the duration in mind. If the ratio is off (like more than one peak per sample)
            then the signal will be trash instead of sinusoidal! So max(frequency) < event_duration/10

        event_duration (int):
            Number of samples per frequency.
            So the time series will have (len(frequencies) * event_duration) samples.

    Returns (np.ndarray):
        The first stimulus (reference stimulus) that can be then transformed
        or directly recalled as the target stimulus.

    TODO: enforce the frequency/duration ratio
    """
    stimulus = []
    for frequency in frequencies:
        stimulus.extend(make_sin(frequency=frequency, event_duration=event_duration))
    return np.array(stimulus)


# functions to alter the reference stimulus and produce the target stimulus for the working memory condition.
def make_retrograde(*,
                    stimulus: np.ndarray) -> np.ndarray:
    """
    Reverse the temporal order of the reference stimulus and makes a "match" condition to be used as the target.

    Args:
        stimulus (List): the reference stimulus.

    Returns (np.ndarray):
        Mirrored stimulus to be placed as the target.

    """
    return np.array(stimulus[::-1])


def make_one_event_off(*,
                       stimulus: np.ndarray,
                       n_events: int,
                       new_f: int,
                       is_retrograde: bool = False) -> np.ndarray:
    """
    Randomly chooses an event and changes its frequency to the given new frequency using the "make_sin" function.
    Makes a "no_match" condition of the reference stimulus,
    either in a recall condition or a retrograde working memory condition.

    Args:

        stimulus (List):
            The reference stimulus.

        n_events (int):
            Assumes the stimulus has events with equal lengths.

        new_f (int):
            A random event will be replaced with a new sinusoidal event with this frequency.

        is_retrograde (bool):
           If True, it mirrors the reference signal first (using make_retrograde) and then makes an event off.
    Returns (np.ndarray):
        The second stimulus (target stimulus) in the no_match conditions.

    TODO: The new frequency shouldn't be the same as the one it's replacing. Handle that.
    TODO: What about N events_off?
    """
    target_stimulus = copy.deepcopy(stimulus)
    if is_retrograde:
        target_stimulus = make_retrograde(stimulus=target_stimulus)

    event_length = int(len(target_stimulus) / n_events)
    event_index = np.random.choice(n_events)
    target_stimulus[(event_index * event_length):(event_index * event_length) + event_length] = make_sin(
        frequency=new_f, event_duration=event_length)
    return np.array(target_stimulus)


# function to copy the reference stimulus to generate target stimulus for the recall condition.
def make_identical(*,
                   stimulus: np.ndarray) -> np.ndarray:
    """
        This might look unnecessary but it can be fed into the trial generator as the "transformation" function.
        It's just to make things more explicit.
        Naturally, makes a "Match" condition of the reference.
    Args:
        stimulus:
            The reference stimulus, of course.

    Returns:
        Copy of the reference for the recall condition.

    """
    return stimulus


# functions to make an experiment.
def make_cue_signal(*,
                    signal_duration: int,
                    cue_onset: int,
                    cue_offset: int):
    """
    Generates the cue signal in which the ANN should replicate during the task.
    It is a zero vector a step function showing the "go" signal.

    Args:
        signal_duration (int): Duration of the whole signal. Should match the trial duration.
        cue_onset (int): When to start the cue. Should match the end of the target stimulus.
        cue_offset (int): When to stop the cue. Should match the end of the trial.

    Returns (np.ndarray): cue signal for one trial
    """
    no_go = np.zeros(signal_duration - (signal_duration - cue_onset))
    go = np.ones((cue_offset - cue_onset))
    return np.hstack((no_go, go))


def make_trial(*,
               stimulus: np.ndarray,
               silence_duration: int = 1_000,
               global_noise: float = .0,
               response_duration: int = 500,
               transformation: Callable = None,
               transformation_params: Optional[Dict] = None,
               cue_generator: Callable = make_cue_signal,
               ) -> np.ndarray:
    """
    Create stimuli of one trial for the working memory/recall task by stacking
    reference stimulus, silence, target stimulus (transformed or not), response duration.

    Args:
        stimulus (np.ndarray):
            The reference stimulus that is presented in the first part of the trial (pre delay).
            It will be passed to "transformation" if needed (for the working memory condition).

        silence_duration (int):
            self explanatory! the longer the silence the harder, I guess!

        global_noise (float):
            adds noise to the whole signal (not the cue) so should make the task harder.
            For now stick to a range of [0, 1].
            Drawn from a Gaussian distribution with mean 0.

        response_duration (int):
            Duration of the response that the ANN should hold onto the correct answer.

        transformation (Callable):
            Transformation to apply to the reference stimulus to make the target stimulus.
            Based on its own parameters (see below).

        cue_generator (Callable):
            Function to generate the cue signal.
            Should get a signal and return a signal.

        transformation_params (Optional[Dict]) :
            Parameters to pass to the transformation function (if it needs anything more than the signal itself).

    Returns (np.ndarray):
        One trial of the working memory/recall task (signal, cue).
        If cue is expected then it returns both signals in a 2D np array. so [0] is the signal [1] is the cue.

    TODO: Error handling and exceptions.
    TODO: Enforce the internal I/O formatting.
    TODO: a parameter called stimulus is passed to the transformation. Another flexible solution?
    """
    transformation_params = transformation_params if transformation_params else {}

    # Generate the target stimulus (transform if needed).
    if transformation is not None:
        stimulus_transformed = transformation(stimulus=stimulus,
                                              **transformation_params)
    else:
        stimulus_transformed = stimulus

    # Generate the silence.
    silence = np.zeros(silence_duration)

    # adding everything up (stimulus, silence, transformed (or not) stimulus, response time)
    signal = np.hstack((stimulus,
                        silence,
                        stimulus_transformed,
                        np.zeros(response_duration)))
    trial_duration = len(signal)

    if global_noise > .0:
        signal = signal + np.random.normal(0, global_noise, trial_duration)

    # Generate the cue signal (if needed).
    cue_on = trial_duration - response_duration
    cue_signal = cue_generator(signal_duration=trial_duration, cue_onset=cue_on, cue_offset=trial_duration)
    return np.array((signal, cue_signal))


def make_placeholder(*,
                     n_trials: int = 10,
                     n_events: int = 3,
                     t_events: int = 200,
                     t_silence: int = 1000,
                     t_response: int = 200
                     ) -> np.ndarray:
    """
    makes a placeholder with shape (t_trials, 4, n_trials). t_trials is the total duration of a trial.
    number 4 there is all the input/outputs:
        0: signal
        1: cue
        2: match
        3: nomatch
        4: response to cue

    following arguments should match 'make_trial' arguments so it's better if defined globally for both.
    Args:
        n_trials (int):
            number of desired trials.

        n_events (int):
            number of events per trial (in the reference stimulus).

        t_events (int):
            duration of each event. probably the longer the better.

        t_silence (int):
            duration of silence between the reference and target stimuli.

        t_response (int):
            duration in which the ANN should fill with a response.

    Returns (np.ndarray):
        an empty (zeros) 3D array to be fed into the 'make_experiment' function.
    """
    t_trials = (t_events * n_events) + t_silence + (t_events * n_events) + t_response
    trials = np.zeros((t_trials, 5, n_trials))
    return trials


def make_experiment(*,
                    trials: np.ndarray,
                    n_events: int,
                    t_events: int,
                    range_frequencies: Tuple[int, int],
                    stimulus_generator: Callable = make_sin_stimulus,
                    trial_generator: Callable = make_trial,
                    trial_generator_params: Dict,
                    is_match: bool = True,
                    transformation_params: Optional[Dict] = None) -> np.ndarray:
    """

    Args:
        trials (np.ndarray):
            a 3D array (t_trials, 4, n_trials) to be filled here.

        n_events (int):
            number of events per trial.

        t_events (int):
            duration of each event.

        range_frequencies (Tuple[int, int]):
            minimum and maximum frequency to be fed into a random generator.

        stimulus_generator (Callable):
            a function to make reference stimuli with. Default is make_sin_stimulus.

        trial_generator (Callable):
            a function to make one trial. Default is make_trial, naturally.

        trial_generator_params (Dict):
            params to be fed into the trial generator.

        is_match (bool):
            is the "correct label" of this batch of trials 'match' or 'nomatch'?

        transformation_params (Optional[Dict]):
            for when there's a transform in trial_generator_params and that transformation needs its own parameters.

    Returns (np.ndarray):
        A 3D array of stimuli and labels.

    # TODO: it shouldn't be allowed to have a match condition with an altering transformation.
    # TODO: error handling and stuff.
    """
    transformation_params = transformation_params if transformation_params else {}
    reference_stimulus = []

    # make all frequencies
    all_frequencies = np.random.randint(low=range_frequencies[0],
                                        high=range_frequencies[1],
                                        size=(trials.shape[2], n_events))

    # make all replacement frequencies (for make_one_off function)

    # looping through each list of frequencies (rows of all_frequencies) to make individual trials
    for index, list_of_frequencies in enumerate(all_frequencies):

        # making one reference stimulus from the list of frequencies.
        reference_stimulus.append(stimulus_generator(frequencies=list_of_frequencies,
                                                     event_duration=t_events))

        # TODO: below should be modular enough for any other transformation function.
        if is_match:
            # generates matching targets so:
            # - for 'recall' the transformation should be 'make_identical'.
            # - for 'wm' the transformation should be 'make_retrograde'.

            # making the match signal (signal, cue)
            # TODO: I'm hard coding things here, there should be a more flexible way.
            trials[:, :2, index] = trial_generator(stimulus=np.array(reference_stimulus[index]),
                                                   transformation_params=transformation_params,
                                                   **trial_generator_params).T

            # correct label of the match trial,
            # the 2nd row (count from 0) is reserved for match
            # and the 3rd row is for nomatch.
            # the 'response signal' is basically another the cue signal since cue is just a step function so why not.
            trials[:, 2, index] = trials[:, 1, index]

        else:
            # generates nomatch targets so:
            # - for 'recall' the transformation should be 'make_one_event_off' with 'retrograde = False'.
            # - for 'wm' the transformation should be 'make_one_event_off' with 'retrograde = True'.

            # prepares the make_one_off function
            if trial_generator_params["transformation"] is make_one_event_off:
                transformation_params["n_events"] = int(len(list_of_frequencies))  # expects the type 'Sized'???
                transformation_params["new_f"] = np.random.randint(low=range_frequencies[0],
                                                                   high=range_frequencies[1])

            # making the nomatch signal (signal, cue)
            trials[:, :2, index] = trial_generator(stimulus=np.array(reference_stimulus[index]),
                                                   transformation_params=transformation_params,
                                                   **trial_generator_params).T
            trials[:, 3, index] = trials[:, 1, index]
        # response to the cue signal.
        trials[:, 4, index] = trials[:, 1, index]
    return trials


# TODO: Refactor to make it more readable.
# TODO: Take care of repeating events?

n_t = 10
n_e = 3
t_e = 500
s_d = 1000
t_r = 500

placeholder = make_placeholder(n_trials=n_t,
                               n_events=n_e,
                               t_events=t_e,
                               t_silence=s_d,
                               t_response=t_r)

trial_generator_params = {"silence_duration": s_d,
                          "global_noise": .0,
                          "response_duration": t_r,
                          "transformation": make_one_event_off,
                          "cue_generator": make_cue_signal}

transformation_params = {"n_events": n_e,
                         "is_retrograde": True}

trials = make_experiment(trials=placeholder,
                         n_events=n_e,
                         t_events=t_e,
                         range_frequencies=(5, 20),
                         is_match=False,
                         trial_generator_params=trial_generator_params,
                         transformation_params=transformation_params)

plt.imshow(trials[:, :, 1].T, aspect='auto', interpolation='none')
plt.show()
