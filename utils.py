import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, Dict, Tuple, Any
import matplotlib.pyplot as plt


def stylize():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'GothamSSm'  # Will skip if you don't have Gotham typeface family


def train_test_preprocessor(*,
                            match_block: np.ndarray,
                            nomatch_block: np.ndarray,
                            from_to_input_index: Tuple = (0, 2),
                            output_index: int = 2,
                            **sklearn_kw: Optional[Dict]) -> Tuple[Any, Any, Any, Any]:
    """
    Turned out given a 3D dataset, sklearn train_test_split shuffles the data in strange ways (for me!) so I wrote
    this function to manually shuffle the match/nomatch blocks, kinda like a deck of cards (here a block of trials).

    Args:
        match_block:
            A 3D matrix of all "match trials", basically the output of experiment_generator with is_3d = True.

        nomatch_block:
            A 3D matrix of all "nomatch trials".

        from_to_input_index:
            indices of inputs, if you're using the experiment generator as is then the default values here will do.
            0 = signal, 1 = cue so (0, 2).

        output_index:
            index of the output signal, here I'm assuming there's only one output but I have to make it general with
            n outputs. Same as above, default value will work if the experiment generator is used.
            2nd index is the response signal there.

        **sklearn_kw:
            kwargs of sklearn train_test_split. Make sure the shuffle is included and it's False since the default is
            True, and we don't want that otherwise we didn't need this weird function.

    Returns:
        x_train, x_test, y_train, y_test
        sklearn's train_test_split output.

    """
    sklearn_kw = sklearn_kw if sklearn_kw else {}

    # making one block of match/nomatch blocks
    data = np.concatenate((match_block, nomatch_block), axis=1)

    # np.random.shuffle only shuffles the first axis so...
    data = np.transpose(data, axes=(1, 0, 2))
    np.random.shuffle(data)

    # from 3D to 2D
    data = np.hstack(data)

    x_train, x_test, y_train, y_test = train_test_split(data[from_to_input_index[0]:from_to_input_index[1], :].T,
                                                        data[output_index, :].T,
                                                        **sklearn_kw)
    return x_train, x_test, y_train, y_test


def simplified_trial_plotter(frequency_list: np.ndarray, *,
                             t_events: int,
                             t_response: int,
                             t_delay: int,
                             is_match: bool = False,
                             figsize: Tuple[int, int] = (4, 1),
                             ) -> Any:
    """
    A very ugly code to produce that weird way I like to see the trials visualized! It produced a figure with
    2 subplots one showing the trial and the other showing the desired response (given response). The trial is then
    visualized as horizontal lines indicating frequencies over time, which is easier to understand compared to plotting
    the signals themselves. I'm still struggling with using the output in a bigger frame, let's say 4x4 subplots.

    Args:
        frequency_list:
            List of integers (np array) indicating the frequencies used in source and target.
            Basically one row of the frequency template matrix.

        t_events:
            Duration of each event.

        t_response:
            Duration of the response.

        t_delay:
            Duration of the silence in between source and target.

        is_match:
            Desired response.

        figsize:
            The resolution is 72DPI and I suggest to stay below width of 8. That's the width of an A4 paper so probably
            5-6 is already too big!

    Returns:
        A matplotlib figure object. Save it with bbox_inches='tight' otherwise xtick labels might get cutoff.

    """
    ys = []
    xmins = []
    xmaxes = []
    signal = (len(frequency_list) - 1) * t_events + t_delay + t_response
    response = np.hstack((np.zeros(signal - t_response), np.ones(t_response)))
    if not is_match:
        response = response * -1

    # finding y, x_min, and x_max of each event
    for index, frequency in enumerate(frequency_list):
        ys.append(frequency)
        if frequency == 0:
            xmins.append(xmaxes[index - 1])
            xmaxes.append(xmins[index] + t_delay)
        else:
            xmins.append(xmaxes[index - 1] if index > 0 else 0)
            xmaxes.append(xmins[index] + t_events)
    silence_index = int(np.where(np.array(ys) == 0)[0])
    ys.pop(silence_index)
    xmins.pop(silence_index)
    xmaxes.pop(silence_index)

    # setting up the figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, hspace=0.2, height_ratios=[3, 1])
    axs = gs.subplots(sharex=True)

    # plotting the frequencies
    for index, value in enumerate(ys):
        axs[0].hlines(y=value, xmin=xmins[index], xmax=xmaxes[index], linewidth=10, color='k')

    # plotting divider lines
    axs[0].axvline(silence_index * t_events, color='#F53A61')
    axs[0].axvline(silence_index * t_events + t_delay, color='#F53A61')
    axs[0].axvline(xmaxes[-1], color='#F53A61')

    # grid and axes for the upper plot
    axs[0].grid(alpha=0.3)
    axs[0].set_xlim(0, signal)
    axs[0].set_ylim(min(ys) - 1, max(ys) + 1)
    axs[0].set_xticks(np.arange(0, signal + 1, 100))
    axs[0].set_ylabel("Frequency (Hz)")

    # plotting the desired response
    axs[1].plot(response, linewidth=2, color='k')

    # grid and axes for the lower plot
    axs[1].set_xticklabels(np.arange(0, signal + 1, 100), rotation=90)
    axs[1].set_ylabel("Correct\n Response (a.u)")
    axs[1].set_yticks(np.linspace(-1.1, 1.1, 3))
    axs[1].set_yticklabels(np.linspace(-1, 1, 3))
    axs[1].set_xlabel("Time")
    axs[1].grid(alpha=0.3)
    return fig


def normalize_step_signal(signals: np.ndarray, max_value: int) -> np.ndarray:
    """
    It pretty much does what it says it does! Since the frequency templates are integers >= 1 then it might be a
    good idea to bring them between 0 and 1 so assuming "signals" is generated by experiment_generator with
    is_3d = True, this function simply normalizes the values.

    Args:
        signals:
            A 3d matrix of step signals with shape (number of input/outputs, number of trials, duration of trials).

        max_value:
            Maximum value in the frequency range in which the experiment block is generated from.

    Returns:
        Normalized (between 0 and 1) step signals.
    """
    for index, signal in enumerate(signals[0]):
        signals[0][index] = signal/max_value
    return signals
