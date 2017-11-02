#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = zip(ages, net_worths, abs(predictions - net_worths))

    # sort the list by absolute error, ascending.
    cleaned_data.sort(key = lambda x: x[2])

    percent = 0.9
    final_idx = int(len(ages)*percent)

    # since it's sorted ascending, slice from 0 to idx.
    return cleaned_data[:final_idx]
