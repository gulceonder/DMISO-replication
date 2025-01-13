def convert_miraw_to_dmiso(input_file, output_file):
    """
    Converts miRAW training data to DMISO format with labels:
    miRNA_id    target_id    miRNA_seq    target_seq    label
    """
    converted_count = 0
    positive_count = 0
    negative_count = 0
    skipped_count = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        # Skip header if exists
        header = f_in.readline()
        
        for line in f_in:
            if not line.strip():
                continue
                
            parts = line.strip().split('\t')
            if len(parts) < 6:  # Need all 6 columns from miRAW format
                print(f"Skipping malformed line: {line.strip()}")
                skipped_count += 1
                continue
                
            try:
                # Parse miRAW format (mirna_name, mirna_seq, target_id, target_seq, label, split)
                mirna_name = parts[0]
                mirna_seq = parts[1]
                target_id = parts[2]
                target_seq = parts[3]
                label = parts[4]
                
                # Include both positive (1) and negative (0) interactions
                if label in ["0", "1"]:
                    # Write in DMISO format with label
                    f_out.write(f"{mirna_name}\t{target_id}\t{mirna_seq}\t{target_seq}\t{label}\n")
                    converted_count += 1
                    if label == "1":
                        positive_count += 1
                    else:
                        negative_count += 1
                else:
                    print(f"Invalid label value: {label}")
                    skipped_count += 1
                    
            except Exception as e:
                print(f"Error processing line: {line.strip()}")
                print(f"Error: {str(e)}")
                skipped_count += 1
                continue

    print(f"\nConversion complete:")
    print(f"Successfully converted: {converted_count} lines")
    print(f"  Positive examples: {positive_count}")
    print(f"  Negative examples: {negative_count}")
    print(f"Skipped: {skipped_count} lines")

# Usage
input_file = "/Users/gulceonder/Documents/School/TEZ/DMISO-main/Utils/concatenated_file.txt"
output_file = "dmiso_format_data_shuffled.txt"
convert_miraw_to_dmiso(input_file, output_file)





def convert_miraw_to_dmiso_fasta(input_file, mirna_file, target_file):
    """
    Converts miRAW training data to DMISO FASTA format
    """
    mirnas = set()
    targets = set()
    
    # First pass to collect unique sequences
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            mirnas.add(parts[0])
            targets.add(parts[1])
    
    # Write miRNA FASTA file
    with open(mirna_file, 'w') as f:
        for i, seq in enumerate(mirnas):
            f.write(f">mirna_{i}\n{seq}\n")
    
    # Write target FASTA file
    with open(target_file, 'w') as f:
        for i, seq in enumerate(targets):
            f.write(f">target_{i}\n{seq}\n")

# Usage
input_file = "/Users/gulceonder/Documents/School/TEZ/DMISO-main/TARGETNET-data/miRAW_Train_Validation.txt"
mirna_file = "dmiso_mirnas.fa"
target_file = "dmiso_targets.fa"
convert_miraw_to_dmiso_fasta(input_file, mirna_file, target_file)

def convert_miraw_to_train_test(input_file, train_output, test_output, test_size=0.2):
    """
    Converts miRAW training data to DMISO format and splits into train/test sets
    while maintaining class balance.
    
    Args:
        input_file: Input file path
        train_output: Output file path for training data
        test_output: Output file path for test data
        test_size: Proportion of data to use for testing (default: 0.2)
    """
    # Store positive and negative examples separately
    positive_examples = []
    negative_examples = []
    skipped_count = 0
    
    with open(input_file, 'r') as f_in:
        # Skip header if exists
        header = f_in.readline()
        
        for line in f_in:
            if not line.strip():
                continue
                
            parts = line.strip().split('\t')
            if len(parts) < 6:
                print(f"Skipping malformed line: {line.strip()}")
                skipped_count += 1
                continue
                
            try:
                # Parse miRAW format
                mirna_name = parts[0]
                mirna_seq = parts[1]
                target_id = parts[2]
                target_seq = parts[3]
                label = parts[4]
                
                # Create DMISO format line
                dmiso_line = f"{mirna_name}\t{target_id}\t{mirna_seq}\t{target_seq}\t{label}\n"
                
                # Add to appropriate list based on label
                if label == "1":
                    positive_examples.append(dmiso_line)
                elif label == "0":
                    negative_examples.append(dmiso_line)
                else:
                    print(f"Invalid label value: {label}")
                    skipped_count += 1
                    
            except Exception as e:
                print(f"Error processing line: {line.strip()}")
                print(f"Error: {str(e)}")
                skipped_count += 1
                continue
    
    # Calculate split sizes
    n_pos_test = int(len(positive_examples) * test_size)
    n_neg_test = int(len(negative_examples) * test_size)
    
    # Randomly shuffle examples
    np.random.shuffle(positive_examples)
    np.random.shuffle(negative_examples)
    
    # Split into train and test sets
    pos_train = positive_examples[n_pos_test:]
    pos_test = positive_examples[:n_pos_test]
    neg_train = negative_examples[n_neg_test:]
    neg_test = negative_examples[:n_neg_test]
    
    # Write training data
    with open(train_output, 'w') as f_train:
        for line in pos_train + neg_train:
            f_train.write(line)
            
    # Write test data
    with open(test_output, 'w') as f_test:
        for line in pos_test + neg_test:
            f_test.write(line)
    
    # Print statistics
    print(f"\nData split complete:")
    print(f"Training set:")
    print(f"  Positive examples: {len(pos_train)}")
    print(f"  Negative examples: {len(neg_train)}")
    print(f"Test set:")
    print(f"  Positive examples: {len(pos_test)}")
    print(f"  Negative examples: {len(neg_test)}")
    print(f"Skipped: {skipped_count} lines")

# Usage
import numpy as np

input_file = "/Users/gulceonder/Documents/School/TEZ/DMISO-main/Utils/concatenated_file.txt"
train_output = "dmiso_format_train.txt"
test_output = "dmiso_format_test.txt"
convert_miraw_to_train_test(input_file, train_output, test_output)