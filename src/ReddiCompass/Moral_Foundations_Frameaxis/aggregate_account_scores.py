# from processing_tools.tools import aggregate_accounts_scores
import pandas as pd


def stratify(tweets_df):
    # Create a new column with tweet counts per user
    tweet_counts = tweets_df.groupby('user_id').size()
    tweets_df['tweet_count'] = tweets_df['user_id'].map(tweet_counts)

    # Define the tweet count ranges for stratified sampling
    ranges = [(0, 100), (100, 500), (500, 1500), (1500, float('inf'))]

    # Create a new column with the tweet count range for each user
    tweets_df['tweet_count_range'] = pd.cut(tweets_df['tweet_count'],
                                            bins=[r[0]
                                                  for r in ranges] + [float('inf')],
                                            labels=[i for i in range(len(ranges))])

    tweets_df['tweet_count_range'] = pd.to_numeric(
        tweets_df['tweet_count_range'], errors='coerce')

    return tweets_df


def aggregate_accounts_scores(input_file, output_path):
    ''' create a new csv file with mf scores means for each account '''

    print(f"Aggregating {input_file}")

    input_file = f"{input_file}"  # ! REMOVE /TESTING/

    df = pd.read_csv(input_file, on_bad_lines='skip',
                     encoding='utf-8', low_memory=False)
    df = stratify(df)
    df = df.apply(pd.to_numeric, errors='coerce')

    grouped_df = df.groupby(['user_id']).mean(numeric_only=True)
    grouped_df = grouped_df.reset_index()
    print(grouped_df.head())

    grouped_df.to_csv(path_or_buf=output_path, index=None)


if __name__ == "__main__":
    input_files = ["data\\scores\\frame-axis\\raw\\LeaveEUOfficial-regular-frame-axis",
                   "data\\scores\\frame-axis\\raw\\BestForBritain-regular-frame-axis"]

    # e.g. data-collection//data//folder_
    output_path = "data\\scores\\frame-axis\\aggregated"

    for file in input_files:
        aggregate_accounts_scores(file, output_path)
