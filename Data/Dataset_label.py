import pandas as pd


def label_dataset(positive_file, negative_file, output_file):
    """
    Label a dataset with sequences from positive and negative samples.
    Positive samples will be labeled as 1, and negative samples as 0.
    The final dataset will be shuffled to mix positive and negative sequences.

    Parameters:
    - positive_file: Path to the Excel file containing positive samples.
    - negative_file: Path to the Excel file containing negative samples.
    - output_file: Path to save the labeled dataset.
    """
    # Load the positive dataset
    positive_df = pd.read_excel(positive_file)
    positive_df['Label'] = 1

    # Load the negative dataset
    negative_df = pd.read_excel(negative_file)
    negative_df['Label'] = 0

    # Combine the two datasets
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
    shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
    shuffled_df.to_excel(output_file, index=False)

    print(f"Labeled and shuffled dataset saved to {output_file}")

# File paths
positive_file = '/content/gdrive/MyDrive/Phase_separation/Positive_dataset.xlsx'
negative_file = '/content/gdrive/MyDrive/Phase_separation/Negative_dataset.xlsx'
output_file = '/content/gdrive/MyDrive/Phase_separation/labeled_dataset.xlsx'

label_dataset(positive_file, negative_file, output_file)