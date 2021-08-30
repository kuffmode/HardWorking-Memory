import numpy as np
import pandas as pd
import copy
from typing import Callable, Dict, List, Optional
import matplotlib.pyplot as plt


def make_sin(*,
             f: int = 10,
             t: int = 500) -> np.ndarray:
    """
    Create a sin wave with the given parameters such that there will be "f" peaks during "t" timepoints.
    So if f is weirdly large (like more than 1 peak per "t") then the signal won't be sinusoidal anymore.
    Same for weirdly small f.
    Args:
        f (int):
            Frequency of the sine wave.
            For now stick to a range of [1, t/10]

        t (int):
            Number of time points.
    Returns (np.ndarray):
        A sinusoidal time series with length t.

    TODO: Error handling and exceptions.
    TODO: Check for the f/t ratio.

    """
    return np.sin(2 * np.pi * f * np.arange(t) / t)


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
        stimulus.extend(make_sin(f=frequency,
                                 t=event_duration))
    return np.array(stimulus)


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
    target_stimulus[(event_index * event_length):(event_index * event_length) + event_length] = make_sin(f=new_f,
                                                                                                         t=event_length)
    return np.array(target_stimulus)


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


def make_cue_signal(*,
                    t: int,
                    cue_onset: int,
                    cue_offset: int):
    """
    Generates the cue signal in which the ANN should replicate during the task.
    It is a zero vector a step function showing the "go" signal.

    Args:
        t (int): Duration of the whole signal. Should match the trial duration.
        cue_onset (int): When to start the cue. Should match the end of the target stimulus.
        cue_offset (int): When to stop the cue. Should match the end of the trial.

    Returns (np.ndarray): cue signal for one trial
    """
    no_go = np.zeros(t - (t - cue_onset))
    go = np.ones((cue_offset - cue_onset))
    return np.hstack((no_go, go))


def make_stimuli_trial(*,
                       stimulus: np.ndarray,
                       silence_duration: int = 1_000,
                       global_noise: float = .0,
                       response_duration: int = 500,
                       transformation: Callable = None,
                       cue_generator: Callable = make_cue_signal,
                       transformation_params: Optional[Dict]) -> np.ndarray:
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

        transformation_params Optional[Dict] :
            Parameters to pass to the transformation function (if it needs anything more than the signal itself).

    Returns (np.ndarray):
        One trial of the working memory/recall task (signal, cue).
        It returns both signals in a 2D np array. so [0] is the signal [1] is the cue.
    
    TODO: Error handling and exceptions.
    TODO: Enforce the internal I/O formatting.
    TODO: a parameter called stimulus is passed to the transformation. Another flexible solution?
    """
    transformation_params = transformation_params if transformation_params else {}
    # Generate the second part of the stimulus (transform if needed).
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
    # Generate the cue signal.
    cue_on = trial_duration - response_duration
    cue_signal = cue_generator(t=trial_duration, cue_onset=cue_on, cue_offset=trial_duration)

    return np.array((signal, cue_signal))


# TODO: What is a good output format? numpy or pandas?
# TODO: Refactor to make it more readable.

# frequency_matrix = np.random.randint(low=5, high=30, size=(3, 4))
# modulated_frequencies = np.random.randint(low=1, high=3, size=(3, 1))
# stim = []
# trial = []
# for indx, frequencies in enumerate(frequency_matrix):
#     make_one_event_off_params = {"n_events": int(len(frequencies)),
#                                  "new_f": int(modulated_frequencies[indx]),
#                                  "is_retrograde": True}
#     stim.append(make_sin_stimulus(frequencies=frequencies, event_duration=500))
#     trial.append(
#         make_stimuli_trial(stimulus=np.array(stim[indx]),
#                            silence_duration=1000,
#                            global_noise=.0,
#                            response_duration=500,
#                            transformation=make_one_event_off,
#                            cue_generator=make_cue_signal,
#                            transformation_params=make_one_event_off_params))
# plt.plot(trial[0][0]);
# plt.show()
