#!/usr/bin/python
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from lsst.summit.testing.analysis.m1m3.plots import inertia_compensation_system
from lsst.summit.testing.analysis.utils import create_logger
from lsst.summit.utils.efdUtils import EfdClient, getEfdData
from lsst.summit.utils.tmaUtils import TMAEvent, TMAEventMaker

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
    efd_client : EfdClient
        Client to access the EFD.
    inner_pad : float, optional
        Time padding inside the stable time window of the slew.
    outer_pad : float, optional
        Time padding outside the slew time window.
    n_sigma : float, optional
        Number of standard deviations to use for the stable region.
    log : logging.Logger, optional
        Logger object to use for logging messages.
    """

    def __init__(
        self,
        event: TMAEvent,
        efd_client: EfdClient,
        inner_pad: float = 0.0,
        outer_pad: float = 0.0,
        n_sigma: float = 1.0,
        log: logging.Logger | None = None,
    ):
        self.log = (
            log.getChild(type(self).__name__)
            if log is not None
            else logging.getLogger(type(self).__name__)
        )

        self.event = event
        self.inner_pad = inner_pad * u.second
        self.outer_pad = outer_pad * u.second
        self.n_sigma = n_sigma
        self.client = efd_client

        self.number_of_hardpoints = 6
        self.measured_forces_topics = [
            f"measuredForce{i}" for i in range(self.number_of_hardpoints)
        ]

        self.log.info(f"Query datasets for {event.dayObs} {event.seqNum}")
        self.df = self.query_dataset()

        self.log.info("Calculate statistics")
        self.stats = self.get_stats()

        self.log.info("Pack results into a Series")
        self.stats = self.pack_stats_series()

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
        evt = self.event
        query_config = {
            "hp_measured_forces": {
                "topic": "lsst.sal.MTM1M3.hardpointActuatorData",
                "columns": self.measured_forces_topics,
                "err_msg": f"No hard-point data found for event {evt.seqNum} on {evt.dayObs}",
            },
            "tma_az": {
                "topic": "lsst.sal.MTMount.azimuth",
                "columns": ["timestamp", "actualTorque", "actualVelocity"],
                "err_msg": f"No TMA azimuth data found for event {evt.seqNum} on {evt.dayObs}",
                "reset_index": True,
                "rename_columns": {
                    "actualTorque": "az_actual_torque",
                    "actualVelocity": "az_actual_velocity",
                },
            },
            "tma_el": {
                "topic": "lsst.sal.MTMount.elevation",
                "columns": ["timestamp", "actualTorque", "actualVelocity"],
                "err_msg": f"No TMA elevation data found for event {evt.seqNum} on {evt.dayObs}",
                "reset_index": True,
                "rename_columns": {
                    "actualTorque": "el_actual_torque",
                    "actualVelocity": "el_actual_velocity",
                },
            },
        }

        # Query datasets
        queries = {key: self.query_efd_data(**cfg) for key, cfg in query_config.items()}  # type: ignore

        # Merge datasets
        df = self.merge_datasets(queries)

        # Convert torque from Nm to kNm
        cols = ["az_actual_torque", "el_actual_torque"]
        df.loc[:, cols] *= 1e-3

        return df

    def merge_datasets(self, queries: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple datasets based on their timestamps.

        Parameters
        ----------
        queries (dict[str, pd.DataFrame]):
            A dictionary of dataframes to be merged.

        Returns
        -------
        df : pd.DataFrame
            A merged dataframe.
        """
        merge_cfg = {
            "left_index": True,
            "right_index": True,
            "tolerance": timedelta(seconds=1),
            "direction": "nearest",
        }

        self.log.info("Merging datasets")
        df_list = [df for _, df in queries.items()]
        merged_df = df_list[0]

        for df in df_list[1:]:
            merged_df = pd.merge_asof(merged_df, df, **merge_cfg)

        return merged_df

    def query_efd_data(
        self,
        topic: str,
        columns: list[str],
        err_msg: str,
        reset_index: bool = False,
        rename_columns: dict | None = None,
    ) -> pd.DataFrame:
        """
        Query the EFD data for a given topic and return a dataframe.

        Parameters
        ----------
        topic : str
            The topic to query.
        columns : List[str]
            The columns to query.
        err_msg : str
            The error message to raise if no data is found.
        reset_index : bool, optional
            Whether to reset the index of the dataframe.
        rename_columns : dict, optional
            A dictionary of column names to rename.

        Returns
        -------
        df : pd.DataFrame
            A dataframe containing the queried data.
        """
        self.log.info(f"Querying dataset: {topic}")
        df = getEfdData(
            self.client,
            topic,
            columns=columns,
            event=self.event,
            prePadding=self.outer_pad,
            postPadding=self.outer_pad,
            warn=False,
        )

        self.log.debug(f"Queried {df.index.size} rows from {topic}")
        if df.index.size == 0:
            self.log.error(err_msg)
            raise ValueError(err_msg)

        if rename_columns is not None:
            df = df.rename(columns=rename_columns)

        if reset_index:
            df["timestamp"] = Time(
                df["timestamp"], format="unix_tai", scale="utc"
            ).datetime
            df.set_index("timestamp", inplace=True)
            df.index = df.index.tz_localize("UTC")

        return df

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
        self.log.info("Find stable time window")
        begin, end = self.find_stable_region()

        self.log.debug("Update begin and end times")
        begin = begin + self.inner_pad
        end = end - self.inner_pad

        self.log.debug("Calculate statistics in stable time window from M1M3")
        stable_slew_stats = pd.DataFrame(
            data=[
                self.get_stats_in_torqueless_interval(
                    self.df[col].loc[begin.isot : end.isot]
                )
                for col in cols
            ],
            index=cols,
        )

        self.log.debug("Concatenate statistics")
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
            self.log.info("Stats are already packed into a Series.")
            return self.stats

        self.log.info("Packing stats into a Series.")
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
        event_keys = [
            "dayObs",
            "seqNum",
            "version",
            "begin",
            "end",
            "duration",
            "type",
            "endReason",
        ]
        event_dict = vars(self.event)
        event_dict = {key: val for key, val in event_dict.items() if key in event_keys}

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


def evaluate_m1m3_ics_single_slew(
    day_obs: int,
    seq_number: int,
    event_maker: TMAEventMaker,
    inner_pad: float = 1.0,
    outer_pad: float = 1.0,
    n_sigma: float = 1.0,
    log: logging.Logger | None = None,
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
    event_maker : TMAEventMaker
        Object to retrieve TMA events.
    inner_pad : float, optional
        Time padding inside the stable time window of the slew.
    outer_pad : float, optional
        Time padding outside the slew time window.
    n_sigma : float, optional
        Number of standard deviations to use for the stable region.
    log : logging.Logger, optional
        Logger object to use for logging messages.

    Returns
    -------
    InertiaCompensationSystemAnalysis
        Object containing the results of the analysis.
    """
    log = log.getChild(__name__) if log is not None else logging.getLogger(__name__)

    log.info("Retrieving TMA slew event.")
    event = event_maker.getEvent(day_obs, seq_number)
    if event is None:
        raise ValueError(f"Could not find event with {seq_number} in {day_obs}")

    log.info("Start inertia compensation system analysis.")
    performance_analysis = M1M3ICSAnalysis(
        event,
        event_maker.client,
        inner_pad=inner_pad,
        outer_pad=outer_pad,
        n_sigma=n_sigma,
        log=log,
    )

    return performance_analysis


def evaluate_m1m3_ics_day_obs(
    day_obs: int,
    event_maker: TMAEventMaker,
    inner_pad: float = 1.0,
    outer_pad: float = 1.0,
    n_sigma: float = 1.0,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Evaluate the M1M3 Inertia Compensation System in every slew event during a
    `dayObs`.

    Parameters
    ----------
    day_obs : int
        Observation day in the YYYYMMDD format.
    event_maker : TMAEventMaker
        Object to retrieve TMA events.
    inner_pad : float, optional
        Time padding inside the stable time window of the slew.
    outer_pad : float, optional
        Time padding outside the slew time window.
    n_sigma : float, optional
        Number of standard deviations to use for the stable region.
    log : logging.Logger, optional
        Logger object to use for logging messages.

    Returns
    -------
    pd.DataFrame
        Data-frame containing statistical summary of the analysis.
    """
    log = log.getChild(__name__) if log is not None else logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    log.info("Retrieving TMA slew events.")
    events = event_maker.getEvents(day_obs)
    log.info(f"Found {len(events)} events for day {day_obs}")

    stats = None
    for event in events:
        log.info(f"Start inertia compensation system analysis on {event.seqNum}.")

        try:
            performance_analysis = M1M3ICSAnalysis(
                event,
                event_maker.client,
                inner_pad=inner_pad,
                outer_pad=outer_pad,
                n_sigma=n_sigma,
                log=log,
            )
            log.info(
                f"Complete inertia compensation system analysis on {event.seqNum}."
            )
        except ValueError:
            log.warning(f"Missing data for {event.seqNum} on {event.dayObs}")
            continue

        if stats is None:
            stats = performance_analysis.stats
        else:
            stats = pd.concat((stats.T, performance_analysis.stats), axis=1).T

    assert isinstance(stats, pd.DataFrame)
    stats = stats.set_index("seqNum", drop=False)
    return stats


if __name__ == "__main__":
    dayObs = 20230802

    log = create_logger("M1M3ICSAnalysis")
    log.info("Start - Single Slew")
    event_maker = TMAEventMaker()
    results = evaluate_m1m3_ics_single_slew(dayObs, 38, event_maker, log=log)
    log.debug(f"Result Series:\n{results.stats}")
    inertia_compensation_system.plot_hp_measured_data(results, log=log)
    log.info("End - Single Slew")

    log.info("Start - Day Obs")
    results_df = evaluate_m1m3_ics_day_obs(dayObs, event_maker, log=log)
    results_df.to_csv(f"m1m3_ics_{dayObs}.csv", index=False)
    log.info("End - Day Obs")
