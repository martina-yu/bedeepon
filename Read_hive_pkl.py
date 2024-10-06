import pandas as pd
import pickle

def process_pkl_data(pkl_data, seq_df, start_position=-19, sequence_length=50):
    result_df = pd.DataFrame()

    for name, df in pkl_data.items():
        # Check if the name exists in the seq_df
        ref_row = seq_df[seq_df['ID'] == name]
        if ref_row.empty:
            print(f"Reference sequence not found for {name}")
            continue
        
        reference_sequence = ref_row['Reference'].values[0]
        
        # Ensure the reference sequence length is enough for the 50bp range
        if len(reference_sequence) < sequence_length:
            print(f"Reference sequence for {name} is shorter than the required sequence length.")
            continue
        
        # Go through each row in the DataFrame corresponding to the name
        for _, row in df.iterrows():
            sequence = list(reference_sequence[:sequence_length])  # Initialize with reference sequence
            
            for col in df.columns[:-3]:  # Exclude last three columns ('Count', 'Frequency', 'Y')
                if '-' in col:
                    try:
                        pos = -int(col.split('-')[1])
                    except ValueError:
                        print(f"Invalid position format in column name: {col}")
                        continue
                else:
                    try:
                        pos = int(col[1:])
                    except ValueError:
                        print(f"Invalid position format in column name: {col}")
                        continue

                index_in_sequence = pos - start_position
                if 0 <= index_in_sequence < sequence_length:
                    # Replace the base in the sequence
                    sequence[index_in_sequence] = row[col]
            
            # Convert list back to string
            modified_sequence = ''.join(sequence)

            # Add the result to the final DataFrame
            result_row = {
                'ID': name,
                'Reference': modified_sequence,
                'Count': row['Count'],       # Keep original Count
                'Frequency': row['Frequency'], # Keep original Frequency
                'Y': row['Y']                # Keep original Y
            }
            
            result_df = result_df.append(result_row, ignore_index=True)
    
    return result_df