import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time


def plot_hp_data(ax: plt.Axes, data: pd.Series | list, label: str) -> list[plt.Line2D]:
    """
    Plot hardpoint data on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes on which the data is plotted.
    topic : str
        The topic of the data.
    data : Series or list
        The data points to be plotted.
    label : str
        The label for the plotted data.

    Returns
    -------
    list
        A list containing the Line2D objects representing the plotted data
        lines.
    """
    line = ax.plot(data, "-", label=label, lw=0.5)
    return line


def mark_slew_begin_end(ax: plt.Axes, slew_begin: Time, slew_end: Time) -> plt.Line2D:
    """
    Mark the beginning and the end of a slew with vertical lines on the given
    axes.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes where the vertical lines are drawn.
    slew_begin : astropy.time.Time
        The slew beginning time.
    slew_end : astropy.time.Time
        The slew ending time.

    Returns
    -------
    matplotlib.lines.Line2D
        The Line2D object representing the line drawn at the slew end.
    """
    _ = ax.axvline(slew_begin.datetime, lw=0.5, ls="--", c="k", zorder=-1)
    line = ax.axvline(
        slew_end.datetime, lw=0.5, ls="--", c="k", zorder=-1, label="Slew Start/Stop"
    )
    return line


def mark_padded_slew_begin_end(ax: plt.Axes, begin: Time, end: Time) -> plt.Line2D:
    """
    Mark the padded beginning and the end of a slew with vertical lines.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes where the vertical lines are drawn.
    begin : astropy.time.Time
        The padded slew beginning time.
    end : astropy.time.Time
        The padded slew ending time.

    Returns
    -------
    matplotlib.lines.Line2D
        The Line2D object representing the line drawn at the padded slew end.
    """
    _ = ax.axvline(begin.datetime, alpha=0.5, lw=0.5, ls="-", c="k", zorder=-1)
    line = ax.axvline(
        end.datetime,
        alpha=0.5,
        lw=0.5,
        ls="-",
        c="k",
        zorder=-1,
        label="Padded Slew Start/Stop",
    )
    return line


def customize_hp_plot(ax: plt.Axes, dataset: object, lines: list[plt.Line2D]) -> None:
    """
    Customize the appearance of the hardpoint plot.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes of the plot to be customized.
    dataset : object
        The dataset object containing the data to be plotted and metadata.
    lines : list
        The list of Line2D objects representing the plotted data lines.
    """
    t_fmt = "%Y%m%d %H:%M:%S"
    ax.set_title(
        f"HP Measured Data\n "
        f"DayObs {dataset.event.dayObs} "  # type: ignore
        f"SeqNum {dataset.event.seqNum} "  # type: ignore
        f"v{dataset.event.version}\n "  # type: ignore
        f"{dataset.df.index[0].strftime(t_fmt)} - "  # type: ignore
        f"{dataset.df.index[-1].strftime(t_fmt)}"  # type: ignore
    )
    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("HP Measured Forces [N]")
    ax.grid(":", alpha=0.2)
    ax.legend(ncol=4, handles=lines)


def plot_velocity_data(ax: plt.Axes, dataset: object) -> None:
    """
    Plot the azimuth and elevation velocities on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes on which the velocity data is plotted.
    dataset : object
        The dataset object containing the data to be plotted and metadata.
    """
    ax.plot(dataset.df["az_actual_velocity"], color="royalblue", label="Az Velocity")  # type: ignore
    ax.plot(dataset.df["el_actual_velocity"], color="teal", label="El Velocity")  # type: ignore
    ax.grid(":", alpha=0.2)
    ax.set_ylabel("Actual Velocity\n [deg/s]")
    ax.legend(ncol=2)


def plot_torque_data(ax: plt.Axes, dataset: object) -> None:
    """
    Plot the azimuth and elevation torques on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes on which the torque data is plotted.
    dataset : object
        The dataset object containing the data to be plotted and metadata.
    """
    ax.plot(dataset.df["az_actual_torque"], color="firebrick", label="Az Torque")  # type: ignore
    ax.plot(dataset.df["el_actual_torque"], color="salmon", label="El Torque")  # type: ignore
    ax.grid(":", alpha=0.2)
    ax.set_ylabel("Actual Torque\n [kN.m]")
    ax.legend(ncol=2)


def plot_stable_region(
    fig: plt.figure, begin: Time, end: Time, label: str = "", color: str = "b"
) -> plt.Polygon:
    """
    Highlight a stable region on the plot with a colored span.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the axes on which the stable region is
        highlighted.
    begin : astropy.time.Time
        The beginning time of the stable region.
    end : astropy.time.Time
        The ending time of the stable region.
    label : str, optional
        The label for the highlighted region.
    color : str, optional
        The color of the highlighted region.

    Returns
    -------
    matplotlib.patches.Polygon
        The Polygon object representing the highlighted region.
    """
    for ax in fig.axes:
        span = ax.axvspan(
            begin.datetime, end.datetime, fc=color, alpha=0.1, zorder=-2, label=label
        )
    return span


def finalize_and_save_figure(fig: plt.figure, name: str) -> None:
    """
    Finalize the appearance of the figure and save it to a file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to be finalized and saved.
    name : str
        The name of the file to which the figure is saved.
    """
    fig.tight_layout()
    fig.savefig(
        name.replace("/", "")
        .replace("  ", "_")
        .replace(" ", "_")
        .replace(",", "")
        .replace("%", "pc")
    )

    plt.show()


def plot_hp_measured_data(dataset: object) -> None:
    """
    Create and plot hardpoint measured data, velocity, and torque on subplots.

    Parameters
    ----------
    dataset : object
        The dataset object containing the data to be plotted and metadata.
    """
    figure_name = (
        f"hp_measured_forces_"
        f"{dataset.event.dayObs}_"  # type: ignore
        f"sn{dataset.event.seqNum}_"  # type: ignore
        f"v{dataset.event.version}"  # type: ignore
    )

    fig, (ax_hp, ax_tor, ax_vel) = plt.subplots(
        num=figure_name,
        dpi=120,
        figsize=(9, 6),
        nrows=3,
        sharex=True,
        height_ratios=[2, 1, 1],
    )

    lines = []
    for hp in range(dataset.number_of_hardpoints):  # type: ignore
        topic = dataset.measured_forces_topics[hp]  # type: ignore
        line = plot_hp_data(ax_hp, dataset.df[topic], f"HP{hp+1}")  # type: ignore
        lines.extend(line)

    slew_begin = Time(dataset.event.begin, scale="utc")  # type: ignore
    slew_end = Time(dataset.event.end, scale="utc")  # type: ignore

    mark_slew_begin_end(ax_hp, slew_begin, slew_end)
    mark_slew_begin_end(ax_vel, slew_begin, slew_end)
    line = mark_slew_begin_end(ax_tor, slew_begin, slew_end)
    lines.append(line)

    mark_padded_slew_begin_end(
        ax_hp, slew_begin - dataset.outer_pad, slew_end + dataset.outer_pad  # type: ignore
    )
    mark_padded_slew_begin_end(
        ax_vel, slew_begin - dataset.outer_pad, slew_end + dataset.outer_pad  # type: ignore
    )
    line = mark_padded_slew_begin_end(
        ax_tor, slew_begin - dataset.outer_pad, slew_end + dataset.outer_pad  # type: ignore
    )
    lines.append(line)

    stable_begin, stable_end = dataset.find_stable_region()  # type: ignore
    stat_begin, stat_end = (
        stable_begin + dataset.inner_pad,  # type: ignore
        stable_end - dataset.inner_pad,  # type: ignore
    )

    plot_velocity_data(ax_vel, dataset)
    plot_torque_data(ax_tor, dataset)
    span_stable = plot_stable_region(fig, stable_begin, stable_end, "Stable", color="k")
    span_with_padding = plot_stable_region(
        fig, stat_begin, stat_end, "Stable w/ Padding", color="b"
    )
    lines.extend([span_stable, span_with_padding])

    customize_hp_plot(ax_hp, dataset, lines)
    finalize_and_save_figure(fig, figure_name)
