#!/usr/bin/python
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from lsst.summit.utils.efdUtils import getEfdData
from lsst.summit.utils.tmaUtils import TMAEvent, TMAEventMaker
from plots import inertia_compensation_system

# TMAEventMaker needs to be instantiated only once.
event_maker = TMAEventMaker()

__all__ = ["M1M3ICSAnalysis"]


class M1M3ICSAnalysis:
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

    def __init__(
        self,
        event: TMAEvent,
        inner_pad: float = 0.0,
        outer_pad: float = 0.0,
        n_sigma: float = 1.0,
        logger: logging.Logger | None = None,
    ):
        if logger is None:
            self._make_logger()
        else:
            self.logger = logger

        self.event = event
        self.inner_pad = inner_pad * u.second
        self.outer_pad = outer_pad * u.second
        self.n_sigma = n_sigma
        self.client = event_maker.client

        self.number_of_hardpoints = 6
        self.measured_forces_topics = [
            f"measuredForce{i}" for i in range(self.number_of_hardpoints)
        ]

        self.logger.info("Query datasets")
        self.df = self.query_dataset()

        self.logger.info("Calculate statistics")
        self.stats = self.get_stats()

        self.logger.info("Pack results into a Series")
        self.stats = self.pack_stats_series()

    def _make_logger(self) -> None:
        """Create a logger object for the ICSAnalysis class."""
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        formatter.datefmt = "%Y-%m-%d %H:%M:%S"

        handler = logging.StreamHandler()
        handler.setLevel(logging.ERROR)
        handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

    def find_stable_region(self) -> tuple[Time, Time]:
        """
        Find the stable region of the dataset. By stable, we mean the region
        where the torque is within n_sigma of the mean.

        Returns
        -------
        tuple[Time, Time]:
            The begin and end times of the stable region.
        """
        az_torque = self.df["az_actual_torque"]
        az_torque_regions = find_adjacent_true_regions(
            np.abs(az_torque - az_torque.mean()) < self.n_sigma * az_torque.std()
        )

        el_torque = self.df["el_actual_torque"]
        el_torque_regions = find_adjacent_true_regions(
            np.abs(el_torque - el_torque.mean()) < self.n_sigma * el_torque.std()
        )

        stable_begin = max([reg[0] for reg in az_torque_regions + el_torque_regions])
        stable_begin = Time(stable_begin, scale="utc")

        stable_end = min([reg[-1] for reg in az_torque_regions + el_torque_regions])
        stable_end = Time(stable_end, scale="utc")

        return stable_begin, stable_end

    def query_dataset(self) -> pd.DataFrame:
        """
        Query all the relevant data, resample them to have the same requency
        and merge them in a single dataframe.

        Returns
        -------
        pd.DataFrame
        """
        self.logger.info("Querying dataset: m1m3 hp meadures forces")
        hp_measured_forces = getEfdData(
            self.client,
            "lsst.sal.MTM1M3.hardpointActuatorData",
            columns=self.measured_forces_topics,
            event=self.event,
            prePadding=self.outer_pad,
            postPadding=self.outer_pad,
            warn=False,
        )

        self.logger.info("Querying dataset: mtmount azimuth torque and velocity")
        tma_az = getEfdData(
            self.client,
            "lsst.sal.MTMount.azimuth",
            columns=["timestamp", "actualTorque", "actualVelocity"],
            event=self.event,
            prePadding=self.outer_pad,
            postPadding=self.outer_pad,
            warn=False,
        )

        tma_az = tma_az.rename(
            columns={
                "actualTorque": "az_actual_torque",
                "actualVelocity": "az_actual_velocity",
            }
        )
        tma_az["timestamp"] = Time(
            tma_az["timestamp"], format="unix_tai", scale="utc"
        ).datetime
        tma_az.set_index("timestamp", inplace=True)
        tma_az.index = tma_az.index.tz_localize("UTC")

        self.logger.info("Querying dataset: mtmount elevation torque and velocity")
        tma_el = getEfdData(
            self.client,
            "lsst.sal.MTMount.elevation",
            columns=["timestamp", "actualTorque", "actualVelocity"],
            event=self.event,
            prePadding=self.outer_pad,
            postPadding=self.outer_pad,
            warn=False,
        )

        tma_el = tma_el.rename(
            columns={
                "actualTorque": "el_actual_torque",
                "actualVelocity": "el_actual_velocity",
            }
        )
        tma_el["timestamp"] = Time(
            tma_el["timestamp"], format="unix_tai", scale="utc"
        ).datetime
        tma_el.set_index("timestamp", inplace=True)
        tma_el.index = tma_el.index.tz_localize("UTC")

        merge_cfg = {
            "left_index": True,
            "right_index": True,
            "tolerance": timedelta(seconds=1),
            "direction": "nearest",
        }

        merged_df = pd.merge_asof(hp_measured_forces, tma_az, **merge_cfg)
        merged_df = pd.merge_asof(merged_df, tma_el, **merge_cfg)
        merged_df[["az_actual_torque", "el_actual_torque"]] = (
            1e-3 * merged_df[["az_actual_torque", "el_actual_torque"]]
        )

        return merged_df

    def get_midppoint(self) -> Time:
        """Return the halfway point between begin and end."""
        return self.df.index[len(self.df.index) // 2]

    def get_stats(self) -> pd.DataFrame:
        """
        Calculate statistics for each column in a given dataset.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing calculated statistics for each column in the
            dataset. For each column, the statistics include minimum, maximum,
            and peak-to-peak values.

        Notes
        -----
        This function computes statistics for each column in the provided
        dataset. It utilizes the `get_minmax` function to calculate minimum,
        maximum, and peak-to-peak values for each column's data.
        """
        cols = self.measured_forces_topics
        full_slew_stats = pd.DataFrame(
            data=[self.get_slew_minmax(self.df[col]) for col in cols], index=cols
        )
        self.logger.info("Find stable time window")
        begin, end = self.find_stable_region()

        self.logger.debug("Update begin and end times")
        begin = begin + self.inner_pad
        end = end - self.inner_pad

        self.logger.debug("Calculate statistics in stable time window")
        stable_slew_stats = pd.DataFrame(
            data=[
                self.get_stats_in_torqueless_interval(
                    self.df[col].loc[begin.isot : end.isot]
                )
                for col in cols
            ],
            index=cols,
        )

        self.logger.debug("Concatenate statistics")
        stats = pd.concat((full_slew_stats, stable_slew_stats), axis=1)

        return stats

    @staticmethod
    def get_stats_in_torqueless_interval(s: pd.Series) -> pd.Series:
        """
        Calculate statistical measures within a torqueless interval.

        This static method computes descriptive statistics for a given pandas
        Series within a torqueless interval. The torqueless interval represents
        a period of the data analysis when no external torque is applied.

        Parameters
        ----------
        s : pandas.Series
            A pandas Series containing data values for analysis.

        Returns
        -------
        pandas.Series
            A pandas Series containing the following statistical measures:
            - Mean: The arithmetic mean of the data.
            - Median: The median value of the data.
            - Standard Deviation (Std): The standard deviation of the data.
        """
        result = pd.Series(
            data=[s.mean(), s.median(), s.std()],
            index=["mean", "median", "std"],
            name=s.name,
        )
        return result

    @staticmethod
    def get_slew_minmax(s: pd.Series) -> pd.Series:
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

    def pack_stats_series(self) -> pd.Series:
        """
        Pack the stats DataFrame into a Series with custom index labels.

        This method takes the DataFrame of statistics stored in the 'stats'
        attribute of the current object and reshapes it into a Series where the
        indexes are generated using custom labels based on the column names and
        index positions. The resulting Series combines values from all columns
        of the DataFrame.

        Returns
        -------
        pandas.Series
            A Series with custom index labels based on the column names and
            index positions. The Series contains values from all columns of the
            DataFrame.
        """
        if isinstance(self.stats, pd.Series):
            self.logger.info("Stats are already packed into a Series.")
            return self.stats

        self.logger.info("Packing stats into a Series.")
        df = self.stats.transpose()

        # Define the prefix patterns
        column_prefixes = df.columns
        index_positions = df.index

        # Generate all combinations of prefixes and positions
        index_prefixes = [
            f"measuredForce{stat.capitalize()}{position}"
            for stat in index_positions
            for position, _ in enumerate(column_prefixes)
        ]

        # Flatten the DataFrame and set the new index
        result_series = df.stack().reset_index(drop=True)
        result_series.index = index_prefixes

        # Append the event information to the Series
        event_dict = vars(self.event)
        event_dict = {
            key: val
            for key, val in event_dict.items()
            if key in ["dayObs", "seqNum", "version"]
        }

        # Create a pandas Series from the dictionary
        event_series = pd.Series(event_dict)

        # Concatenate the two Series
        result_series = pd.concat([event_series, result_series])

        # Display the resulting Series
        return result_series


def find_adjacent_true_regions(
    series: pd.Series, min_adjacent: None | int = None
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Find regions in a boolean Series containing adjacent True values.

    Parameters
    ----------
    series : pd.Series
        The boolean Series to search for regions.

    min_adjacent : int, optional
        Minimum number of adjacent True values in a region.
        Defaults to half size of the series.

    Returns
    -------
    list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
        A list of tuples representing the start and end indices of regions
        containing more than or equal to min_adjacent adjacent True values.
    """
    min_adjacent = min_adjacent if min_adjacent else 0.5 * series.size
    regions = []
    for key, group in series.groupby((series != series.shift()).cumsum()):
        if key and len(group) >= min_adjacent:
            region_indices = group.index
            regions.append((region_indices.min(), region_indices.max()))
    return regions


def get_tma_slew_event(day_obs: int, seq_num: int) -> TMAEvent:
    """
    Retrieve all the Telescope Mount Assembly (TMA) slew events in a day and
    select the one that matches the specified sequence number.

    Parameters
    ----------
    day_obs : int
        Observation day in the YYYYMMDD format.
    seq_num : int
        Sequence number associated with the slew event.

    Returns
    -------
    single_event : lsst.summit.utils.tmaUtils.TMAEvent
        A TMA slew events that occurred within the specified time range.

    Raises
    ------
    ValueError
        If no events are found for the provided day_obs.
    ValueError
        If more than one event matching the seq_num is found for day_obs.
    ValueError
        If no events matching the seq_num are found for day_obs.
    """
    logger.info(f"Query events in {day_obs}")
    events = event_maker.getEvents(day_obs)

    if len(events) == 0:
        raise ValueError(f"Could not find any events for {day_obs}. ")

    logger.info(f"Found {len(events)} events.")
    single_event = [e for e in events if e.seqNum == seq_num]

    logger.info(f"Found {len(single_event)} matching event(s).")
    if len(single_event) > 1:
        raise ValueError(
            f"Expected a single event for {day_obs}. "
            f"Found {len(single_event)} events."
        )

    if len(single_event) == 0:
        raise ValueError(
            f"Could not find any events for {day_obs} day_obs "
            f" that match {seq_num} seq_num."
        )

    assert single_event[0].seqNum == seq_num
    return single_event[0]


def evaluate_m1m3_ics_single_slew(
    day_obs: int,
    seq_number: int,
    inner_pad: float = 1.0,
    outer_pad: float = 1.0,
    n_sigma: float = 1.0,
    logger: logging.Logger | None = None,
) -> M1M3ICSAnalysis:
    """
    Evaluate the M1M3 Inertia Compensation System in a single slew with a
    `seqNumber` sequence number and observed during `dayObs`.

    Parameters
    ----------
    day_obs : int
        Observation day in the YYYYMMDD format.
    seq_number : int
        Sequence number associated with the slew event.

    Returns
    -------
    InertiaCompensationSystemAnalysis
        Object containing the results of the analysis.
    """
    logger.info("Retrieving TMA slew event.")  # type: ignore
    event = get_tma_slew_event(day_obs, seq_number)

    logger.info("Start inertia compensation system analysis.")  # type: ignore
    event = get_tma_slew_event(day_obs, seq_number)
    logger.info("Start inertia compensation system analysis.")  # type: ignore
    performance_analysis = M1M3ICSAnalysis(
        event,
        inner_pad=inner_pad,
        outer_pad=outer_pad,
        n_sigma=n_sigma,
        logger=logger,
    )

    return performance_analysis


def create_logger(name: str) -> logging.Logger:
    """
    Create a logger object with the specified name and returns it.

    Parameters
    ----------
    name : str
        The name of the logger object.

    Returns
    -------
    logger : logging.Logger
        The logger object with the specified name.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    formatter.datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler()
    handler.setLevel(logging.ERROR)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    return logger


if __name__ == "__main__":
    logger = create_logger("M1M3ICSAnalysis")
    logger.info("Start")
    results = evaluate_m1m3_ics_single_slew(20230802, 38, logger=logger)
    inertia_compensation_system.plot_hp_measured_data(results)
    logger.info("End")
