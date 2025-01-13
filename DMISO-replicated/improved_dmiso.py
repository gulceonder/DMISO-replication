import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Conv1D, MaxPooling1D, Flatten, 
                                   concatenate, Activation, BatchNormalization, 
                                   Dropout, LSTM, Bidirectional)
from tensorflow.keras.regularizers import l1_l2, l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Reuse helper functions from original implementation
def addPadding(seq, length):
    total_pad = abs(len(seq)-length)
    left_pad = total_pad//2
    right_pad = total_pad-left_pad
    
    if len(seq) > length:
        return seq[left_pad:left_pad+length]
    return seq + ''.join(['N']*total_pad)

def oneHotEncode(seq):
    nt_map = {'A':0, 'T':1, 'U':1, 'C':2, 'G':3}
    code = np.zeros((len(seq), 4))
    
    for i in range(len(seq)):
        if seq[i] == 'N': 
            code[i] = [0.25]*4
        else:
            assert seq[i] in nt_map, f'{seq[i]} not recognized for one hot coding'
            code[i, nt_map[seq[i]]] = 1
    return code

def create_improved_dmiso_model(mirna_length=30, target_length=60):
    # Input layers
    mirna_input = Input(shape=(mirna_length, 4), name='mirna')
    target_input = Input(shape=(target_length, 4), name='target')
    
    # miRNA branch with increased complexity
    mirna_conv1 = Conv1D(32, 4, activation='linear')(mirna_input)
    mirna_bn1 = BatchNormalization()(mirna_conv1)
    mirna_act1 = Activation('relu')(mirna_bn1)
    mirna_pool1 = MaxPooling1D(2)(mirna_act1)
    mirna_drop1 = Dropout(0.2)(mirna_pool1)
    
    mirna_conv2 = Conv1D(64, 4, activation='linear')(mirna_drop1)
    mirna_bn2 = BatchNormalization()(mirna_conv2)
    mirna_act2 = Activation('relu')(mirna_bn2)
    mirna_pool2 = MaxPooling1D(2)(mirna_act2)
    mirna_drop2 = Dropout(0.2)(mirna_pool2)
    
    # Target branch with increased complexity
    target_conv1 = Conv1D(32, 4, activation='linear')(target_input)
    target_bn1 = BatchNormalization()(target_conv1)
    target_act1 = Activation('relu')(target_bn1)
    target_pool1 = MaxPooling1D(2)(target_act1)
    target_drop1 = Dropout(0.2)(target_pool1)
    
    target_conv2 = Conv1D(64, 4, activation='linear')(target_drop1)
    target_bn2 = BatchNormalization()(target_conv2)
    target_act2 = Activation('relu')(target_bn2)
    target_pool2 = MaxPooling1D(2)(target_act2)
    target_drop2 = Dropout(0.2)(target_pool2)
    
    # Process branches separately with LSTM
    mirna_lstm = Bidirectional(LSTM(32, return_sequences=True))(mirna_drop2)
    target_lstm = Bidirectional(LSTM(32, return_sequences=True))(target_drop2)
    
    # Flatten and merge branches
    mirna_flat = Flatten()(mirna_lstm)
    target_flat = Flatten()(target_lstm)
    merged = concatenate([mirna_flat, target_flat])
    
    # Dense layers with residual connections and regularization
    x = Dense(256, kernel_regularizer=l2(0.01))(merged)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Residual block 1
    res1 = Dense(128, kernel_regularizer=l2(0.01))(x)
    res1 = BatchNormalization()(res1)
    res1 = Activation('relu')(res1)
    res1 = Dropout(0.3)(res1)
    
    # Residual block 2
    res2 = Dense(128, kernel_regularizer=l2(0.01))(res1)
    res2 = BatchNormalization()(res2)
    res2 = Activation('relu')(res2)
    res2 = Dropout(0.3)(res2)
    
    # Add residual connection
    x = concatenate([res1, res2])
    
    # Final dense layers
    x = Dense(64, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[mirna_input, target_input], outputs=output)
    
    # Simplified learning rate configuration
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer,
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def prepare_training_data(data_file):
    # Reuse from original implementation
    X_mirnas, X_targets = [], []
    y_train = []
    
    with open(data_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
                
            mirna_id, target_id, mirna_seq, target_seq, label = parts[:5]
            
            try:
                label = float(label)
                if label not in [0, 1]:
                    continue
            except ValueError:
                continue
            
            mirna_seq_ext = addPadding(mirna_seq, 30)
            target_seq_ext = addPadding(target_seq, 60)
            
            mirna_code = oneHotEncode(mirna_seq_ext)
            target_code = oneHotEncode(target_seq_ext)
            
            X_mirnas.append(mirna_code)
            X_targets.append(target_code)
            y_train.append(label)
    
    return np.array(X_mirnas), np.array(X_targets), np.array(y_train)

if __name__ == "__main__":
    print("Starting improved DMISO training process...")
    
    data_file = "Utils/dmiso_format_train.txt"
    print(f"Loading data from {data_file}")
    X_mirnas, X_targets, y_train = prepare_training_data(data_file)
    
    print("Creating improved model...")
    model = create_improved_dmiso_model()
    
    print("Training model...")
    X_mirnas_train, X_mirnas_val, X_targets_train, X_targets_val, y_train, y_val = train_test_split(
        X_mirnas, X_targets, y_train, 
        test_size=0.2, 
        stratify=y_train,
        random_state=42
    )
    
    # Create model checkpoint callback
    checkpoint_path = "models/best_model.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        [X_mirnas_train, X_targets_train],
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=([X_mirnas_val, X_targets_val], y_val),
        verbose=1,
        callbacks=callbacks
    )
    
    # Save training results
    print("Saving results...")
    with open("training_results.txt", 'w') as f:
        f.write("Training Results\n")
        f.write("================\n\n")
        for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
            history.history['loss'],
            history.history['accuracy'],
            history.history['val_loss'],
            history.history['val_accuracy']
        )):
            f.write(f"Epoch {epoch + 1}:\n")
            f.write(f"  Loss: {loss:.4f}\n")
            f.write(f"  Accuracy: {acc:.4f}\n")
            f.write(f"  Validation Loss: {val_loss:.4f}\n")
            f.write(f"  Validation Accuracy: {val_acc:.4f}\n\n")
    
    print("Training complete! Results saved to training_results.txt") 
    
    # Test the model on test data
    print("\nEvaluating model on test data...")
    test_file = "Utils/dmiso_format_test.txt"
    X_mirnas_test, X_targets_test, y_test = prepare_training_data(test_file)
    
    test_results = model.evaluate(
        [X_mirnas_test, X_targets_test],
        y_test,
        verbose=1
    )
    
    # Get predictions
    y_pred_prob = model.predict([X_mirnas_test, X_targets_test])
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Save test results
    with open("test_results.txt", 'w') as f:
        f.write("Test Results\n")
        f.write("============\n\n")
        f.write(f"Test Loss: {test_results[0]:.4f}\n")
        f.write(f"Test Accuracy: {test_results[1]:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
    
    print("\nTest results saved to test_results.txt") 