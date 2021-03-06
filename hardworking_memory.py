import numpy as np
from typeguard import typechecked
from typing import Callable, Dict, Optional, Tuple, Union, Any, List
from utils import normalize_step_signal


@typechecked
def fix_random_seed(seed: int = 2021):
    np.random.seed(seed)


@typechecked
def make_sin(*,
             frequency: int,
             event_duration: int) -> np.ndarray:
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


    """
    if frequency > (event_duration/10):
        raise ValueError("The frequency is too high. should be something like f <= t/10")
    elif frequency < 0:
        raise ValueError("Did you just pass a negative frequency?")
    return np.sin(2 * np.pi * frequency * np.arange(event_duration) / event_duration)


@typechecked
def make_step(*,
              frequency: int,
              event_duration: int) -> np.ndarray:
    """
    Create a flat line with length of event_duration. By itself it's not impressive but give it to the make_stimulus
    and you'll get a step function.

    Args:
        frequency (int):
            Well, amplitude is a more accurate term here, I know.

        event_duration (int):
            Number of time points.

    Returns (np.ndarray):
        A flat line with amplitude "frequency" and length "event_duration".
    """
    if event_duration < 0:
        raise ValueError("Obviously, duration can't be negative.")
    return np.full(event_duration, frequency)


@typechecked
def make_stimulus(*,
                  function: Callable,
                  function_kw: Optional[Dict],
                  frequencies: List) -> np.ndarray:
    """
    Generates a vector of events given the frequencies by looping through them.
    So the order of events will be the same as the elements in the list.
    Note: it makes one stimulus, can be used for making source or target or both.

    Args:
        function (Callable):
            The function to make events with, default is make_sin so the stimulus is sinusoidal.

        function_kw (Optional[Dict]):
            Keyword Args for the function, if it needs any.

        frequencies (List):
            Just as "make_sin", keep the duration in mind. If the ratio is off (like more than one peak per sample)
            then the signal will be trash instead of sinusoidal! So max(frequency) < event_duration/10.
            Unless you're using the "make_step" though, or something independent of "frequencies".

    Returns (np.ndarray):
        One stimulus that can be stacked with others to form a trial.

    """
    stimulus = []
    function_kw = function_kw if function_kw else {}
    for frequency in frequencies:
        stimulus.extend(function(frequency=int(frequency), **function_kw))
    return np.array(stimulus)


@typechecked
def template_generator(*,
                       n_trials: int = 10,
                       n_events: int,
                       frequency_range: Tuple[int, int, int],
                       transformation: Optional[Callable] = None,
                       transformation_kw: Optional[Dict] = None,
                       retrograde: bool) -> np.ndarray:
    """
    Makes templates to be fed into the trial_generator later. Each row of the template matrix contains information
    of one trial as [frequencies of the source stimulus, 0, frequencies of the target stimulus]. The zero in between
    indicates the delay period. if a transformation (and its keyword arguments) is provided then it'll be applied to
    the target stimulus, which regardless of transformation, can be the retrograde of the source (mirrored in time).

    Args:
        n_trials (int):
            desired number of trials, will be also the number of rows.

        n_events (int):
            desired number of events per stimulus. the more events, the harder the task.

        frequency_range (Tuple[int, int, int]):
            desired range of frequencies: (smallest, largest, step). Make sure:
                1. the largest value is smaller than the duration of each event (see make_stimulus).
                2. the step size makes sense with respect to the number of events. if you need 5 events within the
                    range of 1 to 10 Hz then a step size of 5 doesn't make sense.

        transformation (Optional[Callable]):
            A transformation function that works with a list of integers.
            Can also be a lambda function, why not, go wild.

        transformation_kw (Optional[Dict]):
            Arguments for the transformation function, if needed.

        retrograde (bool):
            If True, the target signal will be reversed in time, this by itself is a transformation but can be
            applied to already transformed signals so this step comes after transformation.
            Use for the working memory condition and not recall, unless you want the target to be nomatch,
            still I'd say don't use this transformation if your network needs to learn a wm task with retrograde too.

    Returns:
        The frequency matrix

    """
    if frequency_range[2] > frequency_range[1]:
        raise ValueError("step size can't be larger than the maximum value.")

    transformation_kw = transformation_kw if transformation_kw else {}

    template = []
    for trial in range(n_trials):
        reference = np.random.choice(np.arange(*frequency_range), n_events)

        if transformation:
            target = transformation(reference, **transformation_kw)
        else:
            target = reference.copy()

        if retrograde:
            target = target[::-1]

        one_trial = np.array([*reference, 0, *target])
        template.extend(one_trial)
    return np.array(template).astype('int').reshape(n_trials, (n_events * 2 + 1))


@typechecked
def scale_one_off(stimulus: np.ndarray, *,
                  frequency_range: Tuple[int, int, int]) -> np.ndarray:
    """
    A transformation function that randomly selects an event and changes its frequency to make a target signal
    for both working memory and recall task. Makes no_match conditions.

    Args:
        stimulus (np.ndarray):
            the template for the source signal. Should be a vector of integers with length n_events.

        frequency_range (Tuple[int, int, int]):
            frequency range of the replacing event (smallest, largest, step). Make sure:
                1. the largest value is smaller than the duration of each event (see make_sin_stimulus).
                2. the step size makes sense with respect to the number of events. if you need 5 events within the
                    range of 1 to 10 Hz then a step size of 5 doesn't make sense.
            I guess it's better to just use one frequency range through out the whole experiment.

    Returns:
        a vector of frequencies to generate the target signal with.

    """
    if frequency_range[2] > frequency_range[1]:
        raise ValueError("step size can't be larger than the maximum value.")

    alter_indx = np.random.randint(len(stimulus), size=1)
    target = stimulus.copy()

    while True:  # makes sure the new frequency != old frequency
        target[alter_indx] = np.random.choice(np.arange(*frequency_range), 1)
        if not target[alter_indx] == stimulus[alter_indx]:
            return target


@typechecked
def trial_generator(*,
                    source_generator: Callable,
                    target_generator: Callable,
                    t_silence: int,
                    t_response: int,
                    global_noise: float = .0,
                    source_generator_kw: Optional[Dict],
                    target_generator_kw: Optional[Dict]) -> Tuple[Union[np.ndarray, Any], np.ndarray]:
    """
    generates one trial of the experiment using a signal generator function and a target generator function, which
    can be different, like one produces sinusoidal signal the other sawtooth or something, or the same function
    but with different parameters.

    Args:
        source_generator (Callable):
            A function to generate the source stimulus with.

        target_generator (Callable):
            A function to generate the target stimulus with.
            Can be the same as source_generator with different parameters.

        t_silence (int):
            Duration of silence between source and target (the delay period).

        t_response (int):
            Duration of response (response period). Comes at the end of the trial.
            If 1 then the problem will be many-to-one.

        global_noise (float):
            Adds noise to the signal, drawn from a Gaussian distribution with mean zero.

        source_generator_kw (Optional[Dict]):
            Parameters for the source generator, if needed.

        target_generator_kw (Optional[Dict]):
            Parameters for the target generator, if needed.

    Returns:
        two vectors, one the signal and the other the cue.
    """
    if t_silence < 0 or t_response < 0:
        raise ValueError("duration of silence or response can't be negative.")

    source_generator_kw = source_generator_kw if source_generator_kw else {}
    target_generator_kw = target_generator_kw if target_generator_kw else {}

    silence = np.zeros(t_silence)
    response = np.zeros(t_response)

    source = source_generator(**source_generator_kw)
    target = target_generator(**target_generator_kw)

    signal = np.hstack((source, silence, target, response))
    trial_duration = len(signal)

    if global_noise > .0:
        signal = signal + np.random.normal(0, global_noise, trial_duration)

    cue_off = np.zeros(len(signal[:-t_response]))
    cue_on = np.ones(len(signal[-t_response:]))
    cue_signal = np.hstack((cue_off, cue_on))
    return signal, cue_signal


@typechecked
def experiment_generator(*,
                         frequency_mat: np.ndarray,
                         trial_generator: Callable,
                         trial_generator_kw: Optional[Dict],
                         is_wm: bool,
                         is_match: bool,
                         is_3d: bool = True):
    """
    Generates one block of trials and their correct labels using a trial_generator function and a frequency template.
    The label is -1 for a no_match condition and 1 for a match condition. 0 is no action, the correct answer of the
    ANN for when the cue is zero.

    Args:
        frequency_mat (np.ndarray):
            The template with which the trial_generator should produce the trials.

        trial_generator (Callable):
            A function to make trials with.

        trial_generator_kw (Optional[Dict]):
            Parameters for the trial generator function and the functions it's using.

        is_wm (bool):
            Informs the network with task condition. Can be dismissed but good if the network is required to
            perform both tasks. Make sure your condition is indeed the one you specify here!
            The cue is a step function with the same length as the response duration (literally the same signal
            but mirrored in time so it appears during the source signal) with 1 indicating wm and -1 for recall.

        is_match (bool):
            Creates a match response. Make sure it is indeed a match experiment!
            # TODO: not sure how but enforce matching/no_matching issue.

        is_3d (bool):
            Reshapes the output to (n_signals, n_trials, trial_duration) for easier inspection/plotting.
            by number of signals I mean: signal, cue, responses, ...
            If False, signals will be concatenated to form a 2D matrix (n_signals, trial_duration).

    Returns:
        One block of experiment.

    """

    trial_generator_kw = trial_generator_kw if trial_generator_kw else {}
    trials = []
    signals = []
    cues = []
    responses = []
    wm_cue = []
    for f_list in frequency_mat:

        # adjusting the arguments for each trial, if there's a "frequencies" argument in the source and target kw of
        # the trial generator kws.

        if "frequencies" in trial_generator_kw["source_generator_kw"]:
            trial_generator_kw["source_generator_kw"]["frequencies"] = list(f_list[:int(np.where(f_list == 0)[0])])

        if "frequencies" in trial_generator_kw["target_generator_kw"]:
            trial_generator_kw["target_generator_kw"]["frequencies"] = list(f_list[int(np.where(f_list == 0)[0]) + 1:])

        trials.append(trial_generator(**trial_generator_kw))

    # from a list of lists to a numpy array.
    for trial in trials:
        signals.extend(trial[0])
        cues.extend(trial[1])
        responses.extend(trial[1]) if is_match else responses.extend(trial[1] * -1)
        wm_cue.extend(trial[1][::-1]) if is_wm else wm_cue.extend(trial[1][::-1] * -1)
    experiment = np.array((signals, cues, responses, wm_cue))

    if is_3d:
        trial_duration = len(trials[0][0])
        n_io = len(experiment)
        n_trials = len(frequency_mat)
        experiment = experiment.reshape((n_io, n_trials, trial_duration))
    return experiment


@typechecked
def hwm_interface(*,
                  n_events: int = 3,
                  frequency_range: Tuple[int, int, int] = (2, 20, 1),
                  event_function: Callable = make_step,
                  transformation: Callable = None,
                  n_trials: int = 100,
                  t_silence: int = 100,
                  t_response: int = 100,
                  global_noise: float = .0,
                  is_retrograde: bool = False,
                  is_wm: bool = False,
                  is_match: bool = True,
                  is_3d: bool = True,
                  transformation_kw: Optional[Dict],
                  function_kw: Optional[Dict],
                  random_seed: Optional[int]) -> np.ndarray:
    """
    Calls other functions internally to produce a block of trials. Should make the conventional uses easier.

    Args:
        n_events (int):
            number of events per stimulus. The more events the harder the task will be.

        frequency_range (Tuple[int, int, int]):
            range of frequencies (amplitudes in case of make_step). should be(min, max, step).
            if you're not using make_step then make sure the values
            make sense with respect to event's duration and number of events per trial. (see template_generator)

        event_function (Callable):
             a function to make individual events with. make_sin and make_step are two built-in functions to use.

        transformation (Callable):
            a function to transform the target signal, if needed. scale_one_off is the built-in function.

        n_trials (int):
            number of trials in the block.

        t_silence (int):
            duration of silence between source and target stimuli.

        t_response (int):
            duration in which the agent should provide a respond. if 1 then the task has a many-to-one structure.

        global_noise (float):
            a random number drawn from a Gaussian distribution with mean zero
            and SD of global_noise will be added to each time point of the input signal.

        is_retrograde (bool):
            if True, flips the target signal in time. for example (1,2,3,0,3,2,1)

        is_wm (bool):
            if True the condition cue signal will be 1.
            For the times you want to train both conditions so the network knows which condition this one is.

        is_match (bool):
            produces the "match" response to the task. Make sure the task is actually a match task.
            for example:
                recall shouldn't have any transformations or retrograde in the match condition.
                match condition for the working memory condition is retrograde.
                you can define your own rules though, for example a working memory match scenario can also be when
                the target is not retrograde but the n_th event is a scaled version of the same event in source.

        is_3d (bool):
            if True, the output will be a block with shape (n_signals, n_trials, trial_duration). I found it easier
            to shuffle trials this way but if False then there will be a long 2d signal with all trials concatenated.

        transformation_kw (Optional[Dict]):
            kwargs for the transformation function, if needed.

        function_kw (Optional[Dict]):
            kwargs for the event generator function, if needed.

        random_seed (int):
            sets numpy.random.seed(). I heard it's not the best way to do it so take care here.

    Returns:
        a block of experiment with the same condition (match/nomatch).
    """
    transformation_kw = transformation_kw if transformation_kw else {}
    function_kw = function_kw if function_kw else {}
    np.random.seed(random_seed) if random_seed else np.random.seed(None)

    source_generator_kw = {"function": event_function,
                           "function_kw": function_kw,
                           "frequencies": None}
    target_generator_kw = source_generator_kw.copy()

    trial_kw = {"source_generator": make_stimulus,
                "source_generator_kw": source_generator_kw,
                "target_generator": make_stimulus,
                "target_generator_kw": target_generator_kw,
                "t_response": t_response,
                "t_silence": t_silence,
                "global_noise": global_noise}

    if is_match:
        template = template_generator(n_trials=n_trials,
                                      transformation=None,
                                      frequency_range=frequency_range,
                                      retrograde=is_retrograde)
    else:
        template = template_generator(n_trials=n_trials,
                                      n_events=n_events,
                                      transformation=transformation,
                                      frequency_range=frequency_range,
                                      transformation_kw=transformation_kw,
                                      retrograde=is_retrograde)

    block = experiment_generator(frequency_mat=template,
                                 trial_generator=trial_generator,
                                 trial_generator_kw=trial_kw,
                                 is_wm=is_wm,
                                 is_match=is_match,
                                 is_3d=is_3d)

    if source_generator_kw["function"] is make_step:
        block = normalize_step_signal(block, frequency_range[1])

    return block

# TODO: refactoring.
# TODO: testing.
# TODO: Not sure if np.seed is a good way to take care of randomness
# TODO: Changing all "frequencies" to just events?
