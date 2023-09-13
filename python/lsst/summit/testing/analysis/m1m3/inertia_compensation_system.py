#!/usr/bin/python
import logging
import pandas as pd

from lsst.summit.utils.tmaUtils import TMAEventMaker, TMAEvent

# TODO @bquint: fix logger - not working now.
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
formatter.datefmt = "%Y-%m-%d %H:%M:%S"

handler = logging.StreamHandler()
handler.setLevel(logging.ERROR)
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class InertiaCompensationSystemAnalysis:
    """
    Evaluate the M1M3 Inertia Compensation System's performance by
    calculating the minima, maxima and peak-to-peak values during a
    slew. In addition, calculates the mean, median and standard deviation
    when the slew has contant velocity or zero acceleration.

    Parameters
    ----------
    event : `lsst.summit.utils.tmaUtils.TMAEvent`
        Abtract representation of a slew event.
    """

    # ToDo @bquint
    def __init__(event: TMAEvent):
        raise NotImplementedError

    @staticmethod
    def get_minmax_from_series(s):
        """
        Calculate minimum, maximum, and peak-to-peak values for a data-series.

        Parameters
        ----------
        s : pandas.Series
            The input pandas Series containing data.

        Returns
        -------
        pandas.Series
            A Series containing the following calculated values for the two
            halves of the input Series:
            - min: Minimum value of the Series.
            - max: Maximum value of the Series.
            - ptp: Peak-to-peak (ptp) value of the Series (abs(max - min)).
        """
        result = pd.Series(
            data=[s.min(), s.max(), np.ptp(s)],
            index=["min", "max", "ptp"],
            name=s.name,
        )
        return result

    @staticmethod
    def get_avgmedstd_from_series(s):
        """
        Calculate average, median, and standard deviation for a data-series.

        Parameters
        ----------
        s : pandas.Series
            The input pandas Series containing data.

        Returns
        -------
        pandas.Series
            A Series containing the following calculated values for the two
            halves of the input Series:
            - mean: The mean value of the Series.
            - median: The median value of the Series.
            - std: The standard deviation of the Series.
        """
        result = pd.Series(
            data=[s.mean(), s.median(), s.std()],
            index=["mean", "median", "std"],
            name=s.name,
        )
        return result


def get_tma_slew_event():
    """
    Retrieve Telescope Mount Assembly (TMA) slew events within a specified time
    range.

    Parameters
    ----------
    dayObs : int
        Observation day in the YYYYMMDD format.
    seqNum : int
        Sequence number associated with the slew event.

    Returns
    -------
    lsst.summit.utils.tmaUtils.TMAEvent
        A TMA slew events that occurred within the specified time range.

    Notes
    -----
    This function retrieves TMA slew events occurring between the specified
    start and end times. It uses the TMAEventMaker class to obtain events for
    the specified day of observation (dayObs). The events are filtered to
    include only those that start after 1 second before the specified start time
    and end before 1 second after the specified end time.

    Example
    -------
    >>>
    >>>
    >>>
    """
    events = event_maker.getEvents(day_obs_begin)

    assert len(events) == 1
    return events[0]


def evaluate_single_slew(day_obs, seq_number):
    """
    Evaluates the M1M3 Inertia Compensation System in a single slew with a
    `seqNumber` sequence number and observed during `dayObs`.

    Parameters
    ----------
    day_obs : int
        Observation day in the YYYYMMDD format.
    seq_number : int
        Sequence number associated with the slew event.

    Returns
    -------
    lsst.summit.utils.tmaUtils.TMAEvent
        A TMA slew events that occurred within the specified time range.
    """
    event_maker = TMAEventMaker()

    logger.info(f"Query events in {day_obs}")
    events = event_maker.getEvents(day_obs)

    logger.info(f"Found {len(events)} events.")
    single_event = [e for e in events if e.seqNum == seq_number]

    logger.info(f"Found {len(single_event)} matching event(s).")
    if len(single_event) > 1:
        raise ValueError(
            f"Expected a single event for {day_obs}. "
            f"Found {len(single_event)} events."
        )

    if len(single_event) == 0:
        raise ValueError(f"Could not find any events for {day_obs}. ")

    return single_event[0]


if __name__ == "__main__":
    logger.info("Start")
    evaluate_single_slew(20230802, 38)
