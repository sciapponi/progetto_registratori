import pandas as pd
import csv # Needed for quoting constants if used explicitly, good practice
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---
INPUT_CSV_FILE = "unbalanced_train_segments.csv"
OUTPUT_CSV_FILE = "rural_without_birds.csv"

# Bird-related labels to exclude
# Using a set for efficient lookup
bird_labels = {
    "/m/015p6",    # Bird vocalization, bird call, bird song
    "/m/020bb7",    # Bird
    "/m/07pggtn",   # Chicken, rooster
    "/m/07sx8x_",   # Crow
    "/m/0h0rv",     # Owl
    "/m/07r_25d",   # Duck
    "/m/04s8yn",    # Goose
    "/m/07r5c2p",   # Gull, seagull
    "/m/09d5_",     # Pigeon, dove
    "/m/07r_80w",   # Turkey
    "/m/05_wcq",     # Fowl
    "/m/09x0r" ,    # Speech
    "/m/07qfr4h",
    "/m/02zsn",
    "/m/05zppz"
    # Add any other bird-related labels if needed
}

# Rural label variants to include (ensure exact match from AudioSet Ontology)
# '/t/dd00129' seems to be the official tag for "Rural environment" or similar
# Including variants just in case, but '/t/dd00129' is likely the one needed.
rural_label_variants = {
    "/t/dd00129" # Primary label often meaning Rural environment/Countryside
    # "/m/06_y0by" # environmental noise
    # "dd00129",   # Less likely to appear without prefix
    # "t/dd00129"  # Less likely to appear without leading slash
}

# --- Helper Function ---
def clean_and_split_labels(label_str):
    """
    Cleans the raw string from the CSV (potentially quoted)
    and splits it into a list of label strings.
    Returns an empty list if input is not a valid string or is empty.
    """
    if not isinstance(label_str, str) or not label_str:
        return []
    # Remove potential surrounding quotes and whitespace
    cleaned_str = label_str.strip().strip('"')
    # Split by comma and strip whitespace from each label
    labels = [label.strip() for label in cleaned_str.split(',') if label.strip()]
    return labels

# --- Main Script Logic ---
logging.info(f"Reading CSV: {INPUT_CSV_FILE}")

try:
    # Read the CSV file:
    # - comment='#': Skips lines starting with # (metadata and commented header)
    # - header=None: Because the real header line starts with # and is skipped
    # - names=[...]: Manually assign the column names
    # - skipinitialspace=True: Handles potential spaces after commas
    # - quoting=csv.QUOTE_MINIMAL: Default, handles quoted fields correctly
    # - on_bad_lines='warn': Warn about bad lines instead of skipping silently or erroring
    df = pd.read_csv(
        INPUT_CSV_FILE,
        comment='#',
        header=None,
        names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
        skipinitialspace=True,
        # quoting=csv.QUOTE_MINIMAL, # Usually default, explicit is fine
        on_bad_lines='warn' # Use 'skip' or 'error' if preferred
    )
    logging.info(f"Successfully read {len(df)} data rows.")

except FileNotFoundError:
    logging.error(f"Error: Input CSV file not found at '{INPUT_CSV_FILE}'")
    exit()
except Exception as e:
    logging.error(f"Error reading CSV file: {e}")
    exit()

# Check if DataFrame is empty
if df.empty:
    logging.warning("CSV file loaded successfully, but resulted in an empty DataFrame. No data to process.")
    # Optionally create an empty output file
    pd.DataFrame(columns=["YTID", "start_seconds", "end_seconds", "positive_labels"]).to_csv(OUTPUT_CSV_FILE, index=False)
    logging.info(f"Empty output file created: {OUTPUT_CSV_FILE}")
    exit()


# Apply label cleaning and splitting
# Ensure the column exists before applying
if "positive_labels" in df.columns:
    logging.info("Processing 'positive_labels' column...")
    # Use .fillna('') before apply to handle potential NaN values gracefully
    df["labels_list"] = df["positive_labels"].fillna('').apply(clean_and_split_labels)
    logging.info("Finished processing labels.")
else:
    logging.error("Error: 'positive_labels' column not found after reading CSV. Check CSV format and reader parameters.")
    exit()


# --- Filtering Logic ---
logging.info("Applying filtering conditions...")

# Condition 1: Row must contain at least one rural label
contains_rural = df["labels_list"].apply(lambda label_list:
    any(label in rural_label_variants for label in label_list)
)

# Condition 2: Row must NOT contain any bird labels
contains_no_birds = df["labels_list"].apply(lambda label_list:
    not any(label in bird_labels for label in label_list)
)

# Combine conditions
filtered_df = df[contains_rural & contains_no_birds]

logging.info(f"Filtering complete.")
logging.info(f"Initial rows: {len(df)}")
logging.info(f"Rows matching conditions (rural AND no birds): {len(filtered_df)}")


# --- Output Results ---
# Show result summary (optional)
if not filtered_df.empty:
    logging.info("Sample of filtered data (first 3 rows):")
    # Display original labels for verification if needed
    print(filtered_df.head(3)[["YTID", "start_seconds", "end_seconds", "positive_labels"]])
else:
    logging.info("No rows matched the filtering criteria.")

# Save the result
# Select only the original columns for the output CSV
output_columns = ["YTID", "start_seconds", "end_seconds", "positive_labels"]
logging.info(f"Saving filtered data to: {OUTPUT_CSV_FILE}")
filtered_df[output_columns].to_csv(OUTPUT_CSV_FILE, index=False, quoting=csv.QUOTE_MINIMAL)
logging.info("Script finished successfully.")