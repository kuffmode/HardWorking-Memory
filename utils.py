import numpy as np
import pandas as pd
import copy
from typing import Callable, Dict, List


def make_sin(*,
             f: int = 10,
             t: int = 500) -> np.ndarray:
    """
    Create a sin wave with the given parameters such that there will be "f" peaks during "t" timepoints.
    So if f is weirdly large (like more than 1 peak per "t") then the signal won't be sinusoidal anymore.
    It will be trash. Same for weirdly small f.

    Args:
        f (int): 
            Frequency of the sine wave. 
            For now stick to a range of [1, t/10]
            
        t (int): 
            Number of time points.

    Returns (np.ndarray):
        A sin wave.
    
    TODO: Error handling and exceptions. 
    TODO: Check for the f/t ratio.
     
    """
    return np.sin(2 * np.pi * f * np.arange(t) / t)


def reverse_in_time(signal: List) -> List:
    """
    Reverse the time axis of a signal.
    
    """
    return signal[-1::-1]


def make_one_event_off(signal: List,
                       n_events: int,
                       new_f: int) -> np.ndarray:
    """
    Randomly chooses an event and changes its frequency to a new one using the make_sin function.
    Args:
        signal (List):
            The original signal, just the first stimulus part.

        n_events (int):
            Self explanatory.

        new_f (int):
            This too!

    Returns (np.ndarray):
        The second stimulus in the no_match conditions.

    TODO: What about N events_off?
    """
    altered_signal = copy.deepcopy(signal)
    event_length = int(len(altered_signal) / n_events)
    event_index = np.random.choice(n_events)
    altered_signal[(event_index * event_length):(event_index * event_length) + event_length] = make_sin(f=new_f,
                                                                                                        t=event_length)
    return np.array(altered_signal)


def make_cue_signal(*,
                    t: int,
                    cue_onset: int,
                    cue_offset: int):
    """
    Generates the cue signal in which the agent should replicate during the task.
    It is a zero vector a step function at one point showing the "go" signal.

    Args:
        t (int): Duration of the whole signal. Should match the trial duration.
        cue_onset (int): When to start the cue. Should match the end of transformed stimulus.
        cue_offset (int): When to stop the cue. Should match the end of the trial.

    Returns (np.ndarray): cue signal for one trial
    """
    no_go = np.zeros(t - (t - cue_onset))
    go = np.ones((cue_offset - cue_onset))
    return np.hstack((no_go, go))


def make_stimulus(*,
                  function: Callable = make_sin,
                  frequencies: List = None,
                  silence_duration: int = 1_000,
                  global_noise: float = .0,
                  response_duration: int = 500,
                  transformation: Callable = None,
                  cue_generator: Callable = make_cue_signal,
                  function_params: Dict,
                  transformation_params: Dict) -> np.ndarray:
    """
    Create one trial of the working memory task stimulus starting from 
    the stimulus generated by the function, followed by the silence,
    followed by the transformed (or not) stimulus, and ends with the
    response period.

    Args:
        function (Callable): 
            Function to generate the stimulus.
            
        frequencies (List): 
            List of frequencies to generate events with so len(frequencies) == n_events.
            
        silence_duration (int): 
            self explanatory! the longer the silence the harder.
        
        global_noise (float):
            adds noise to the whole signal (not the cue) so should make the task harder.
            For now stick to a range of [0, 1].
            Drawn from a Gaussian distribution with mean 0.
            
        response_duration (int): 
            Duration of the response.
            
        transformation (Callable): 
            Transformation to apply to the stimulus.
            Based on its own parameters (see below)

        cue_generator (Callable): 
            Function to generate the cue signal. 
            Should get a signal and return a signal.
            
        function_params (Dict) : 
            Parameters to pass to the event generator. 
            There should be a pair for the frequency called "f"

        transformation_params (Dict) :
            Parameters to pass to the transformation function (if it needs anything more than the signal itself).

    Returns (np.ndarray):
        One trial of the working memory task (signal, cue).
    
    TODO: Error handling and exceptions.
        
    """
    # Generate the stimulus.
    stimulus = []

    for frequency in frequencies:
        function_params["f"] = frequency
        stimulus.extend(function(**function_params))

    # Generate the second part of the stimulus (transform if needed).
    if transformation is not None:
        stimulus_transformed = transformation(stimulus,**transformation_params)
    else:
        stimulus_transformed = stimulus

    # Generate the silence.
    silence = np.zeros(silence_duration)

    signal = np.hstack((np.array(stimulus),
                        silence,
                        np.array(stimulus_transformed),
                        np.zeros(response_duration)))
    trial_duration = len(signal)

    if global_noise > .0:
        signal = signal + np.random.normal(0, global_noise, trial_duration)
    # Generate the cue signal.
    cue_on = trial_duration - response_duration
    cue_signal = cue_generator(t=trial_duration, cue_onset=cue_on, cue_offset=trial_duration)

    # adding everything up (stimulus, silence, transformed stimulus, response time)

    return np.array((signal, cue_signal))

# TODO: Frequency modulation
# TODO: What is a good output format? numpy or pandas?
# TODO: Refactor to make it more readable.


##============================================================================================##
