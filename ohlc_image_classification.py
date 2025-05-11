import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split # Consider TimeSeriesSplit for financial data
from sklearn.preprocessing import MinMaxScaler
from pyts.image import MarkovTransitionField

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
OHLC_FILE_PATH = -r"C:\Users\spenc\Downloads\Dev Files\img-class\training-data\ohlc.csv"
IMAGE_SIZE = 40 # Size of the generated MTF image
WINDOW_SIZE = IMAGE_SIZE # Use window size equal to image size for MTF
TARGET_COLUMN = 'Close' # Column to encode and use for training

# Labeling parameters (adjust as needed, based on classification.ipynb)
LABEL_NBARS = 3
LABEL_WINDOW = 5
LABEL_TARGET_POS = 3
LABEL_QSIZE = 0.125 # Quantile size for bucketing returns

# Training parameters
NUM_EPOCHS = 50 # Adjust as needed
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1291

# --- Seeding for reproducibility ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # May need these for full determinism, can impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

set_seed(SEED)

# --- Data Loading and Preprocessing ---

def load_ohlc_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded data shape: {df.shape}")
    return df

def create_labels(df, n_bars=LABEL_NBARS, window=LABEL_WINDOW, target_bar_pos=LABEL_TARGET_POS, qsize=LABEL_QSIZE):
    print("Creating labels...")
    # Calculate future percentage change in High price (adjust logic if needed)
    pct_changes = ((df["High"].shift(-(target_bar_pos + n_bars)).rolling(window, center=True).mean() - df["High"]) / df["High"])
    
    # Handle NaNs introduced by rolling/shift
    pct_changes = pct_changes.dropna()
    
    # Quantize the changes
    qs = np.arange(0, 1 + qsize, qsize)
    # Use labels=False to get integer codes directly
    qranges, bins = pd.qcut(pct_changes, q=qs, retbins=True, duplicates='drop', labels=False) 
    
    print("Quantile bins:")
    print(bins)
    print("Value counts per quantile bin:")
    print(qranges.value_counts())

    # Map quantile bins to 0 (Hold), 1 (Buy - top quantiles), 2 (Sell - bottom quantiles)
    num_quantiles = len(qranges.cat.categories)
    buy_threshold = num_quantiles - 2 # Example: Top 2 quantiles for Buy
    sell_threshold = 1             # Example: Bottom 2 quantiles for Sell

    labels = qranges.map(lambda x: 1 if x >= buy_threshold else (2 if x <= sell_threshold else 0))
    
    # Align labels with the original dataframe (dropping NaNs from pct_changes means labels start later)
    aligned_labels = pd.Series(labels, index=pct_changes.index).reindex(df.index)
    print("Label distribution (0:Hold, 1:Buy, 2:Sell):")
    print(aligned_labels.value_counts())
    
    return aligned_labels

def encode_mtf(series, image_size=IMAGE_SIZE):
    # Scale data to [0, 1] for MTF - crucial!
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    
    mtf = MarkovTransitionField(image_size=image_size, n_bins=5) # n_bins can be tuned
    mtf_image = mtf.fit_transform(np.array([scaled_series]))
    return mtf_image[0]

def prepare_image_data(df, labels, target_col=TARGET_COLUMN, window_size=WINDOW_SIZE, image_size=IMAGE_SIZE):
    print(f"Preparing image data using '{target_col}' with window size {window_size}...")
    images = []
    valid_labels = []
    valid_indices = []

    # Ensure labels Series is aligned with df's index before iterating
    labels = labels.reindex(df.index)

    for i in tqdm(range(len(df) - window_size + 1)):
        window_end = i + window_size
        window_slice = df[target_col].iloc[i:window_end]
        
        # Get label corresponding to the *end* of the window
        current_label = labels.iloc[window_end - 1] 

        # Only proceed if the window is full and label is not NaN
        if len(window_slice) == window_size and not pd.isna(current_label):
            mtf_image = encode_mtf(window_slice, image_size)
            images.append(mtf_image)
            valid_labels.append(int(current_label)) # Ensure label is integer
            valid_indices.append(df.index[window_end - 1]) # Store index of the label

    print(f"Generated {len(images)} images.")
    return np.array(images), np.array(valid_labels), pd.DatetimeIndex(valid_indices)

# --- PyTorch Dataset ---

class OhlcImageDataset(Dataset):
    def __init__(self, images, labels):
        # Add channel dimension for grayscale CNN input
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) 
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# --- PyTorch Model ---

class OhlcCnn(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE, num_classes=3):
        super(OhlcCnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # Adjusted padding
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # Adjusted padding
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size dynamically
        self._flattened_size = self._get_conv_output_size(image_size)
        
        self.fc1 = nn.Linear(self._flattened_size, 64) 
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def _get_conv_output_size(self, image_size):
        # Helper to calculate size after convolutions/pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, image_size, image_size)
            x = self.pool1(self.relu1(self.conv1(dummy_input)))
            x = self.pool2(self.relu2(self.conv2(x)))
            return int(np.prod(x.size()[1:])) # product of C * H * W

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Training Loop ---

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    print(f"Starting training on {device}...")
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            train_pbar.set_postfix({'loss': running_loss / total_samples, 'acc': running_corrects.double() / total_samples})

        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = running_corrects.double() / total_samples
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item()) # Store as float

        # --- Validation Phase ---
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)
                
                val_pbar.set_postfix({'loss': running_loss / total_samples, 'acc': running_corrects.double() / total_samples})


        epoch_val_loss = running_loss / total_samples
        epoch_val_acc = running_corrects.double() / total_samples
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item()) # Store as float

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # Save model state if needed
            # torch.save(model.state_dict(), 'best_ohlc_cnn_model.pth')
            # print("Best model saved.")

    print(f"Training finished. Best Val Acc: {best_val_acc:.4f}")
    return model, history

# --- Plotting ---
def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training acc')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    ohlc_df = load_ohlc_data(OHLC_FILE_PATH)

    # 2. Create Labels
    labels = create_labels(ohlc_df)
    
    # Align df and labels (drop rows where labels couldn't be calculated)
    common_index = ohlc_df.index.intersection(labels.dropna().index)
    ohlc_df_aligned = ohlc_df.loc[common_index]
    labels_aligned = labels.loc[common_index]
    print(f"Data shape after aligning with labels: {ohlc_df_aligned.shape}")


    # 3. Prepare Image Data
    # Note: Using WINDOW_SIZE samples before the label date to generate the image
    images, image_labels, image_indices = prepare_image_data(ohlc_df_aligned, labels_aligned, 
                                                             target_col=TARGET_COLUMN,
                                                             window_size=WINDOW_SIZE, 
                                                             image_size=IMAGE_SIZE)

    # 4. Split Data
    # IMPORTANT: For financial time series, random splitting can lead to lookahead bias.
    # Use TimeSeriesSplit or manual splitting by date for a more robust evaluation.
    # Example using train_test_split (use with caution):
    X_train, X_val, y_train, y_val = train_test_split(images, image_labels, 
                                                    test_size=0.2, 
                                                    random_state=SEED, 
                                                    stratify=image_labels) # Stratify for balanced classes
    
    print(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}")
    print(f"Train label distribution: {np.bincount(y_train)}")
    print(f"Validation label distribution: {np.bincount(y_val)}")

    # 5. Create Datasets and DataLoaders
    train_dataset = OhlcImageDataset(X_train, y_train)
    val_dataset = OhlcImageDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Initialize Model, Loss, Optimizer
    model = OhlcCnn(image_size=IMAGE_SIZE, num_classes=3).to(DEVICE)
    print("Model Architecture:")
    print(model)
    
    # Calculate class weights for imbalanced datasets (optional but recommended)
    class_counts = np.bincount(y_train)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(np.unique(y_train)) # Normalize
    print(f"Using class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. Train Model
    trained_model, history = train_model(model, train_loader, val_loader, 
                                        criterion, optimizer, NUM_EPOCHS, DEVICE)

    # 8. Plot Results
    plot_history(history)

    # Optional: Evaluate further or save model
    # torch.save(trained_model.state_dict(), 'final_ohlc_cnn_model.pth')
    # print("Final model saved.") 