import os
import pandas as pd
from datetime import datetime


def main():
    dfs = []
    file_names = ["candidate_phrases_entire_batch_{}.pkl".format(i) for i in range(1, 14)]
    last_item = 0

    for file_name in file_names:
        # Read the pkl file as a Pandas DataFrame
        current_hour = datetime.now().strftime("%H:%M:%S")
        print(f'{current_hour}:read: ' + f"{file_name}")
        batch_df = pd.read_pickle("./Extracting_phrases/entire/batches_for_extracting_phrases/" + f"{file_name}")
        dfs.append(batch_df)

    print(f'{current_hour}:start saving to pkl: candidate_phrases_entire_data')
    combined_df = pd.concat(dfs)
    combined_df.to_pickle(r"./Extracting_phrases/entire/candidate_phrases_entire_data.pkl")
    print(f'{current_hour}:finished saving to pkl: candidate_phrases_entire_data')


if __name__ == "__main__":
    main()