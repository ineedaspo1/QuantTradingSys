# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:19:51 2018

@author: kruegkj

time_utils.py
"""

import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.offsets import BDay

us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def print_beLongs(df):
    print("beLong counts: ")
    print(df['beLong'].value_counts())
    print("==========================")


class TimeUtility:

    def is_oos_data_split(self, issue, pivot_date, is_oos_ratio, oos_months, segments):
        return_dates = ()
        is_months = is_oos_ratio * oos_months
        months_to_load = oos_months + segments * is_months
        is_start_date = pivot_date - \
            relativedelta(months=segments * oos_months + is_months)
        oos_start_date = pivot_date - \
            relativedelta(months=oos_months * segments)
        data_start_date = is_start_date - relativedelta(months=1)
        is_end_date = is_start_date + relativedelta(months=is_months)
        oos_end_date = oos_start_date + relativedelta(months=oos_months)

        print('{0:>30} {1:}'.format("Segments: ", segments))
        print('{0:>30} {1:}'.format("IS OOS Ratio: ", is_oos_ratio))
        print('{0:>30} {1:}'.format("OOS months: ", oos_months))
        print('{0:>30} {1:}'.format("IS Months: ", is_months))
        print('{0:>30} {1:}'.format("Months to load: ", months_to_load))
        print('{0:>30} {1:}'.format("Data Load Date: ", data_start_date))
        print('{0:>30} {1:}'.format("IS Start  Date: ", is_start_date))
        print('{0:>30} {1:}'.format("OOS Start Date: ", oos_start_date))
        print('{0:>30} {1:}'.format("Pivot Date: ", pivot_date))
        return_dates = (data_start_date,
                        is_start_date,
                        oos_start_date,
                        is_months,
                        is_end_date,
                        oos_end_date)
        return return_dates


if __name__ == "__main__":

    from plot_utils import *
    from retrieve_data import DataRetrieve, ComputeTarget
    timeUtil = TimeUtility()
    plotIt = PlotUtility()

    issue = "tlt"
    pivotDate = datetime.date(2018, 4, 2)
    is_oos_ratio = 4
    oos_months = 3
    segments = 3

    isOosDates = timeUtil.is_oos_data_split(
        issue, pivotDate, is_oos_ratio, oos_months, segments)
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    is_months = isOosDates[3]
    is_end_date = isOosDates[4]
    oos_end_date = isOosDates[5]

    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate, pivotDate)

    # set beLong level
    beLongThreshold = 0.0
    ct = ComputeTarget()
    dataSet = ct.setTarget(dataSet, "Long", beLongThreshold)

    for i in range(segments):
        modelData = dataSet[is_start_date:is_end_date]
        print("IN SAMPLE")
        print_beLongs(modelData)
        plotIt.plot_beLongs("In Sample", issue, modelData,
                            is_start_date, is_end_date)
        is_start_date = is_start_date + \
            relativedelta(months=oos_months) + BDay(1)
        is_end_date = is_start_date + relativedelta(months=is_months) - BDay(1)

        # OOS
        modelData = dataSet[oos_start_date:oos_end_date]
        print("OUT OF SAMPLE")
        print_beLongs(modelData)
        plotIt.plot_beLongs("Out of Sample", issue,
                            modelData, oos_start_date, oos_end_date)
        oos_start_date = oos_end_date + BDay(1)
        oos_end_date = oos_end_date + \
            relativedelta(months=oos_months) - BDay(1)
