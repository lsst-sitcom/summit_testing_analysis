
import matplotlib.pyplot as plt
from astropy.time import Time

def plot_hp_data(ax, topic, data, label):
    l = ax.plot(data, "-", label=label, lw=0.5)
    return l

def mark_slew_begin_end(ax, slew_begin, slew_end):
    ax.axvline(slew_begin.datetime, lw=0.5, ls="--", c="k", zorder=-1)
    l = ax.axvline(slew_end.datetime, lw=0.5, ls="--", c="k", zorder=-1, label="Slew Start/Stop")
    return l

def mark_padded_slew_begin_end(ax, begin, end):
    ax.axvline(begin.datetime, alpha=0.5, lw=0.5, ls="-", c="k", zorder=-1)
    l = ax.axvline(end.datetime, alpha=0.5, lw=0.5, ls="-", c="k", zorder=-1, label="Padded Slew Start/Stop")
    return l

def customize_hp_plot(ax, dataset, lines):
    t_fmt = "%Y%m%d %H:%M:%S"
    ax.set_title(
        f"HP Measured Data\n "
        f"DayObs {dataset.event.dayObs} "
        f"SeqNum {dataset.event.seqNum} " 
        f"v{dataset.event.version}\n "
        f"{dataset.df.index[0].strftime(t_fmt)} - "
        f"{dataset.df.index[-1].strftime(t_fmt)}"
    )
    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("HP Measured Forces [N]")
    ax.grid(":", alpha=0.2)
    ax.legend(ncol=4, handles=lines)

def plot_velocity_data(ax, dataset):
    l_az_vel = ax.plot(dataset.df["az_actual_velocity"], color='royalblue', label='Az Velocity')
    l_el_vel = ax.plot(dataset.df["el_actual_velocity"], color='teal', label='El Velocity')
    ax.grid(":", alpha=0.2)
    ax.set_ylabel("Actual Velocity\n [deg/s]")
    ax.legend(ncol=2)

def plot_torque_data(ax, dataset):
    l_az_vel2 = ax.plot(dataset.df["az_actual_torque"], color='firebrick', label='Az Torque')
    l_el_vel2 = ax.plot(dataset.df["el_actual_torque"], color='salmon', label='El Torque')        
    ax.grid(":", alpha=0.2)
    ax.set_ylabel("Actual Torque\n [kN.m]")
    ax.legend(ncol=2)

def plot_stable_region(fig, begin, end, label="", color="b"):
    for ax in fig.axes:
        span = ax.axvspan(begin.datetime, end.datetime, fc=color, alpha=0.1, zorder=-2, label=label)
    return span

        
def finalize_and_save_figure(fig, name):

    fig.tight_layout()
    fig.savefig(
        name
        .replace("/", "")
        .replace("  ", "_")
        .replace(" ", "_")
        .replace(",", "")
        .replace("%", "pc")
    )
    
    plt.show()

def plot_hp_measured_data(dataset):
    
    figure_name = (
        f"hp_measured_forces_"
        f"{dataset.event.dayObs}_"
        f"sn{dataset.event.seqNum}_"
        f"v{dataset.event.version}"
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
    for hp in range(dataset.number_of_hardpoints):
        topic = dataset.measured_forces_topics[hp]
        line = plot_hp_data(ax_hp, topic, dataset.df[topic], f"HP{hp+1}")
        lines.extend(line)
        
    slew_begin = Time(dataset.event.begin, scale="utc")
    slew_end = Time(dataset.event.end, scale="utc")
    
    mark_slew_begin_end(ax_hp, slew_begin, slew_end)
    mark_slew_begin_end(ax_vel, slew_begin, slew_end)
    line = mark_slew_begin_end(ax_tor, slew_begin, slew_end)
    lines.append(line)
    
    mark_padded_slew_begin_end(ax_hp, slew_begin - dataset.outer_pad, slew_end + dataset.outer_pad)
    mark_padded_slew_begin_end(ax_vel, slew_begin - dataset.outer_pad, slew_end + dataset.outer_pad)
    line = mark_padded_slew_begin_end(ax_tor, slew_begin - dataset.outer_pad, slew_end + dataset.outer_pad)
    lines.append(line)
        
    stable_begin, stable_end = dataset.find_stable_region()
    stat_begin, stat_end = stable_begin + dataset.inner_pad, stable_end - dataset.inner_pad
    
    plot_velocity_data(ax_vel, dataset)
    plot_torque_data(ax_tor, dataset)
    span_stable = plot_stable_region(fig, stable_begin, stable_end, "Stable", color="k")
    span_with_padding = plot_stable_region(fig, stat_begin, stat_end, "Stable w/ Padding", color="b")
    lines.extend([span_stable, span_with_padding])
        
    customize_hp_plot(ax_hp, dataset, lines)
    finalize_and_save_figure(fig, figure_name)