"""
clean_data.py - Clean password dataset

Logic explained (beginner-friendly):

1. Input file data/data.csv has 3 row formats with the same header 'password,strength':
   - Normal (2 fields):      password, strength
   - Five-field (5 fields):  username, email, IP, password, strength
   - Seven-field (7 fields): username, email, IP, password, extra, extra, strength
   We only care about the password and the strength value.

2. How we extract:
   - If a row has 2 columns -> password is column 0, strength is column 1
   - If a row has 5 columns -> password is column 3, strength is column 4 (last)
   - If a row has 7 columns -> password is column 3, strength is column 6 (last)
   This avoids ever storing username/email/IP.

3. Validation:
   - Skip the header row
   - Strip whitespace from password and strength
   - Strength must be exactly '0' or '1' or '2' (invalid otherwise)
   - Password must not be empty
   - Any row with unexpected column count or invalid strength is counted as skipped

4. Output:
   - Write only password,strength to data/clean_passwords.csv (with header)
   - Never write to or modify data/data.csv (open input as read-only)
   - At the end, print: rows processed, valid rows written, skipped/invalid rows

No machine learning is done here - just cleaning.
"""

import csv
import os

# File paths - relative to project root
INPUT_FILE = os.path.join("data", "data.csv")
OUTPUT_FILE = os.path.join("data", "clean_passwords.csv")

# Strength values that are allowed
VALID_STRENGTHS = {"0", "1", "2"}

def clean_data():
    rows_processed = 0
    valid_rows = 0
    skipped_rows = 0

    # Open input for reading only, output for writing
    with open(INPUT_FILE, "r", newline="", encoding="utf-8", errors="replace") as infile, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Skip header row from input
        try:
            header = next(reader)
        except StopIteration:
            print("Input file is empty.")
            return

        # Write clean header to output
        writer.writerow(["password", "strength"])

        # Process each row
        for row in reader:
            rows_processed += 1

            # Extract password and strength based on row length
            password = None
            strength = None

            if len(row) == 2:
                # Normal format: password, strength
                password = row[0]
                strength = row[1]
            elif len(row) == 5:
                # Five-field format: username, email, IP, password, strength
                password = row[3]
                strength = row[4]
            elif len(row) == 7:
                # Seven-field format: username, email, IP, password, extra, extra, strength
                password = row[3]
                strength = row[6]
            else:
                # Unexpected format -> skip
                skipped_rows += 1
                continue

            # Clean whitespace
            # Only password and strength are kept - username/email/IP are never stored
            password = password.strip() if password else ""
            strength = strength.strip() if strength else ""

            # Validate strength is 0, 1, or 2
            if strength not in VALID_STRENGTHS:
                skipped_rows += 1
                continue

            # Validate password is not empty
            if not password:
                skipped_rows += 1
                continue

            # Row is valid - write it
            writer.writerow([password, strength])
            valid_rows += 1

    # Print summary
    print(f"Rows processed: {rows_processed}")
    print(f"Valid rows written: {valid_rows}")
    print(f"Skipped/invalid rows: {skipped_rows}")
    print(f"Clean file written to: {OUTPUT_FILE}")
    print(f"Original file untouched: {INPUT_FILE}")

if __name__ == "__main__":
    clean_data()
