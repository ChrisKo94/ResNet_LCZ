import numpy as np
import pandas as pd

def get_avg_accuracy(actual, predicted, acc_table):

    df = pd.DataFrame(np.transpose(
        np.vstack(
            (actual, np.array(actual == predicted))
        )
    ),
        columns=["class", "correct"])

    aggr_df = df.groupby(["class"]).sum()
    aggr_df = aggr_df.add_suffix('_sum').reset_index()

    join = pd.merge(acc_table, aggr_df,
                    how="left",
                    on=["class"]).fillna(0)

    join["correct_sum"] = join["correct_sum_x"] + join["correct_sum_y"]
    join_new = join[["class", "correct_sum"]]

    return join_new