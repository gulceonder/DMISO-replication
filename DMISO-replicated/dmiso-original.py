import numpy as np
import sys, os, argparse, getopt, pdb
from itertools import islice
from collections import defaultdict

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate true positives, false positives, true negatives, false negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity
    }

def write_metrics(metrics, output_path):
    """Write metrics to the output file."""
    metrics_text = "\n\nPerformance Metrics:\n"
    metrics_text += f"Accuracy: {metrics['accuracy']:.4f}\n"
    metrics_text += f"Precision: {metrics['precision']:.4f}\n"
    metrics_text += f"Recall: {metrics['recall']:.4f}\n"
    metrics_text += f"F1 Score: {metrics['f1']:.4f}\n"
    metrics_text += f"Specificity: {metrics['specificity']:.4f}\n"
    
    with open(output_path, 'a') as f:
        f.write(metrics_text)
        
# -----------------------------------------------
def writeFile(data, filename, mode="w"):
    d = os.path.dirname(filename)

    if d != "":
        if not os.path.exists(d):
            os.makedirs(d)

    fl = open(filename, mode)
    fl.write(data)
    fl.close()

# -----------------------------------------------
def writeDataTableAsText(data, filename, mode="w"):
    text = formatDataTable(data, "\t", "\n")

    writeFile(text, filename, mode)

# -----------------------------------------------
def formatDataTable(data, col_sep="\t", row_sep="\n"):
    return row_sep.join([col_sep.join([str(item1) for item1 in item]) for item in data])

# -----------------------------------------------------------------
def addPadding(seq, length):

    total_pad = abs(len(seq)-length)
    left_pad = total_pad//2
    right_pad = total_pad-left_pad

    if len(seq) > length:
        return seq[left_pad:left_pad+length]

    return seq + ''.join(['N']*total_pad)
    #return ''.join(['N']*left_pad) + seq + ''.join(['N']*right_pad)

# -----------------------------------------------------------------
def oneHotEncode(seq):

    nt_map = {'A':0, 'T':1, 'U':1, 'C':2, 'G':3}

    code = np.zeros((len(seq), 4))

    for i in range(len(seq)):
        if seq[i] == 'N': code[i] = [0.25]*4
        else:
            assert seq[i] in nt_map, '{} not recognized for one hot coding'.format(seq[i])
            code[i, nt_map[seq[i]]] = 1

    return code

#------------------------------------------------------------------------------
def processData(interactions):

    X_mirnas, X_targets = [], []

    for mirna_id, target_id, mirna_seq, target_seq in interactions:

        mirna_seq_ext = addPadding(mirna_seq, len_mirna)
        mirna_code = oneHotEncode(mirna_seq_ext)
        target_seq_ext = addPadding(target_seq, len_target)
        target_code = oneHotEncode(target_seq_ext)
        X_mirnas.append(mirna_code)
        X_targets.append(target_code)

    return X_mirnas, X_targets

#------------------------------------------------------------------------------
def loadModel(model_path):

    json_file = open(model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_path + ".h5")
    print("Loaded model from disk")
    return model

# -------------------------------------------------------------------------
def showArgError(parser):
    parser.parse_args(['-h'])
    exit()

# -------------------------------------------------------------------------
def showError(msg):
    print('----------- Error !!! ------------')
    print(msg)
    print('----------------------------------')

# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Predicts interactions between miRNA and target pair. ' + \
                                             'It takes an input file of query miRNA/isomiR and target pairs and predict the interactions between the pairs. ' + \
                                             'Alternatively, it can take separate files for miRNA/isomiR and target sequences and outputs all the possible interactions.', \
                                 epilog='Example: "python3 dmiso.py -p examples/test_pairs.txt -o examples/test_output.txt" OR ' + \
                                                  '"python3 dmiso.py -m examples/test_miRNAs.fa -t examples/test_mRNAs.fa -o examples/test_output.txt"')
required_arg = parser.add_argument_group('required arguments')
required_arg.add_argument('-p', help="Path for miRNA (or isomiR) and mRNA pair file in a tsv format. The file must contain data in the following order: miRNA id, target id, miRNA sequence, target sequence", required=True,
                          metavar="PAIR")
required_arg.add_argument('-m', help="Path for miRNA or isomiR sequence in fasta format.", required=True, metavar="MIRNA/ISOMIR")
required_arg.add_argument('-t', help="Path for mRNA sequence in fasta format.", required=True, metavar="MRNA")
parser.add_argument('-o', help='Path for DMISO outputs', metavar="OUTPUT")

if '-h' in sys.argv[1:]:
    showArgError(parser)

opts, args = getopt.getopt(sys.argv[1:], 'p:m:t:o:')

pair_file_path = ''
mi_seq_file_path = ''
m_seq_file_path = ''
output_path = ''

if len(opts) == 0: showArgError(parser)

for i in opts:
    if i[0] == '-p':
        pair_file_path = i[1]
        if pair_file_path == '': showArgError(parser)
    elif i[0] == '-m':
        mi_seq_file_path = i[1]
        if mi_seq_file_path == '': showArgError(parser)
    elif i[0] == '-t':
        m_seq_file_path = i[1]
        if m_seq_file_path == '': showArgError(parser)
    elif i[0] == '-o':
        output_path = i[1]
        if output_path == '': showArgError(parser)
    else:
        showArgError(parser)

is_valid_pair_path = os.path.isfile(pair_file_path)
is_valid_mi_seq_path = os.path.isfile(mi_seq_file_path)
is_valid_m_seq_path = os.path.isfile(m_seq_file_path)

# Validate pair path
if not (is_valid_pair_path or (is_valid_mi_seq_path and is_valid_m_seq_path)):
    if not is_valid_pair_path:
        showError('The following file path for miRNA target pair file does not exist' + '\n' + pair_file_path)
    elif not is_valid_mi_seq_path:
        showError('The following file path for miRNA (isomiR) sequence file does not exist' + '\n' + mi_seq_file_path)
    elif not is_valid_m_seq_path:
        showError('The following file path for mRNA sequence file does not exist' + '\n' + m_seq_file_path)
    exit()

if output_path == '': output_path = os.path.join(os.path.dirname(pair_file_path), 'predictions_dmiso.txt')

from keras.models import model_from_json

tool_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(tool_path, 'models/model_top')
model = loadModel(model_path)

def get_layer_shape(layer):
    shape = layer.output_shape
    if isinstance(shape, tuple):
        return shape[1] if len(shape) > 1 else shape[0]
    return shape[0][1] if isinstance(shape, list) else None

mirna_layer = model.get_layer('mirna')
target_layer = model.get_layer('target')
len_mirna = get_layer_shape(mirna_layer)
len_target = get_layer_shape(target_layer)

if len_mirna is None or len_target is None:
    raise ValueError("Could not determine input layer shapes")

def writeResults(output, output_path, first_write=False):
    if first_write:
        # First write - overwrite the file
        writeDataTableAsText(output, output_path, "w")
    else:
        # Subsequent writes - append without extra newline
        writeDataTableAsText(output, output_path, "a")

# First write the header
header = [['miRNA ID', 'Target ID', 'miRNA Sequence', 'Target Sequence', 'Prediction Score', 'Prediction']]
writeResults(header, output_path, first_write=True)

# Create a dictionary to track processed pairs
processed_pairs = defaultdict(bool)

if is_valid_pair_path:

    with open(pair_file_path) as f:
        while True:
            file_chunk = list(islice(f, 1000000))
            if not file_chunk:
                break

            interactions = []
            chunk_labels = []
            actual_predictions = []
            actual_labels = []

            for item in file_chunk:
                item = item.strip()
                parts = item.split('\t')
                if len(parts) >= 5:  # Check if we have label
                    mi_id, m_id, mi_seq, m_seq, label = parts[:5]
                    chunk_labels.append(float(label))
                else:
                    mi_id, m_id, mi_seq, m_seq = parts[:4]
                    chunk_labels.append(None)
                    
                # Instead of splitting long sequences, just take the first len_target nucleotides
                if len(m_seq) > len_target:
                    m_seq = m_seq[:len_target]
                
                interactions.append([mi_id, m_id, mi_seq, m_seq])

            X_mirnas, X_targets = processData(interactions)
            y_pred_pp = model.predict([X_mirnas, X_targets])[:, 0]
            y_pred = (y_pred_pp > 0.5).astype(int)

            # Store predictions and true labels only for original pairs
            for i in range(len(interactions)):
                if chunk_labels[i] is not None:
                    actual_predictions.append(y_pred[i])
                    actual_labels.append(chunk_labels[i])

            output = np.c_[interactions, y_pred_pp, y_pred]
            writeResults(output, output_path)

            # After processing all chunks
            if actual_predictions:
                metrics = calculate_metrics(actual_labels, actual_predictions)
                write_metrics(metrics, output_path)
                
                print("\nPerformance Metrics:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1']:.4f}")
                print(f"Specificity: {metrics['specificity']:.4f}")

elif is_valid_mi_seq_path and is_valid_m_seq_path:

    chunk_size = 2000000

    file_chunk_mi = []

    with open(mi_seq_file_path) as f_mi:
        while True:
            file_chunk_mi += list(islice(f_mi, chunk_size))
            if not file_chunk_mi:
                break

            if len(file_chunk_mi) >= chunk_size:
                for i in range(len(file_chunk_mi) - 1, -1, -1):
                    if file_chunk_mi[i][0] == '>':
                        break
                seq_info = ''.join(file_chunk_mi[:i]).split('>')
                file_chunk_mi = file_chunk_mi[i:]
            else:
                seq_info = ''.join(file_chunk_mi).split('>')
                file_chunk_mi = []

            mi_info = {}
            for item in seq_info[1:]:
                lines = item.split('\n')
                id = lines[0]
                seq = ''.join(lines[1:-1])
                mi_info[id] = seq.upper()

            file_chunk_m = []

            with open(m_seq_file_path) as f_m:
                while True:
                    file_chunk_m += list(islice(f_m, chunk_size))
                    if not file_chunk_m:
                        break

                    if len(file_chunk_m) >= chunk_size:
                        for i in range(len(file_chunk_m) - 1, -1, -1):
                            if file_chunk_m[i][0] == '>':
                                break
                        seq_info = ''.join(file_chunk_m[:i]).split('>')
                        file_chunk_m = file_chunk_m[i:]
                    else:
                        seq_info = ''.join(file_chunk_m).split('>')
                        file_chunk_m = []

                    m_info = {}
                    for item in seq_info[1:]:
                        lines = item.split('\n')
                        id = lines[0]
                        seq = ''.join(lines[1:-1])
                        m_info[id] = seq.upper()

            interactions = []
            for mi_id, mi_seq in mi_info.items():
                for m_id, m_seq in m_info.items():
                    while len(m_seq) > len_target*1.5:
                        m_sub_seq = m_seq[:len_target]
                        m_seq = m_seq[len_target:]
                        pair_key = f"{mi_id}_{m_id}_{mi_seq}_{m_sub_seq}"
                        if not processed_pairs[pair_key]:
                            interactions.append([mi_id, m_id, mi_seq, m_sub_seq])
                            processed_pairs[pair_key] = True
                        interactions.append([mi_id, m_id, mi_seq, m_seq])
            X_mirnas, X_targets = processData(interactions)
            # Convert lists to numpy arrays first
            X_mirnas = np.array(X_mirnas)
            X_targets = np.array(X_targets)

            # Get the minimum length between the two arrays
            min_samples = min(X_mirnas.shape[0], X_targets.shape[0])

            # Trim both arrays to the same length
            X_mirnas = X_mirnas[:min_samples]
            X_targets = X_targets[:min_samples]

            # Now try prediction
            y_pred_pp = model.predict([X_mirnas, X_targets])[:, 0]
            y_pred = (y_pred_pp > 0.5).astype(int)
            output = np.c_[interactions, y_pred_pp, y_pred]
            output = output[y_pred == 1]

            writeResults(output, output_path)

            # After processing all chunks
            if actual_predictions:
                metrics = calculate_metrics(actual_labels, actual_predictions)
                write_metrics(metrics, output_path)
                
                print("\nPerformance Metrics:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1']:.4f}")
                print(f"Specificity: {metrics['specificity']:.4f}")

print('Prediction complete !!')
print('Please check the prediction results at ' + os.path.abspath(output_path))

