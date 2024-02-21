import sys
import platform
import torch
import sklearn as sk

has_gpu = torch.cuda.is_available()
has_mps = getattr(torch,'has_mps',False)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")
print('==================================START==================================')

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

class CustomAttentionClassifier(nn.Module):
    def __init__(self, model_path, length):
        super(CustomAttentionClassifier, self).__init__()
        
        # Load the pre-trained model
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, output_attentions=True)
        
        # Define a CNN to process the attention maps
        # Assuming attention maps are of size [12, 12, length, length]
        self.cnn = nn.Sequential(
            nn.Conv2d(144, 64, kernel_size=3, padding=1), # 144 is 12 * 12, as we're going to reshape the attention maps
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (length//4) * (length//4), 128), # Replace `length` with the actual length
            nn.ReLU(),
            nn.Linear(128, length*length)
        )
        
    def forward(self, **kwargs):
        # Pass input through the pre-trained model
        outputs = self.model(**kwargs)
        
        # Extract attention maps and reshape
        attention_maps = torch.stack(outputs['attentions']).squeeze(2).view(1, 144, length, length) # Replace `length` with the actual length

        # Pass attention maps through CNN
        logits = self.cnn(attention_maps)
        
        return logits

# Load ready dataset
import pickle
print('Loading dataset..')

load_path = "./dataset.pkl" # Select input directory
with open(load_path, "rb") as file:
    dataset = pickle.load(file)

from torch.utils.data import TensorDataset, DataLoader

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

from torch.utils.data import SubsetRandomSampler

train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size

train_sampler = SubsetRandomSampler(range(train_size))
valid_sampler = SubsetRandomSampler(range(train_size, train_size + valid_size))
test_sampler = SubsetRandomSampler(range(train_size + valid_size, len(dataset)))

train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=4, sampler=valid_sampler)
test_loader = DataLoader(dataset, batch_size=4, sampler=test_sampler)

print("Number of training batches:",len(train_loader),
    "\nNumber of valifation batches:", len(valid_loader),
    "\nNumber of test batches:", len(test_loader))

def remove_padding(binary_mask, padded_map):
    binary_mask = binary_mask.bool()  # Convert to boolean (T / F)
    
    # Calculate the indices of the unpadded region
    indices = torch.nonzero(binary_mask)
    
    # Get the minimum and maximum indices along each dimension
    min_indices, _ = torch.min(indices, dim=0)
    max_indices, _ = torch.max(indices, dim=0)
    
    # Extract the unpadded region from the padded map
    unpadded_map = padded_map[
        min_indices[0]:max_indices[0] + 1,
        min_indices[1]:max_indices[1] + 1
    ]
    
    return unpadded_map

def weigth_calculator(unpadded_target):
    
    n_samples = unpadded_target.shape[0]
    n_classes = 2

    weights = n_samples / (n_classes * torch.bincount(unpadded_target.reshape(-1).round().int()))
    if weights.shape == torch.Size([1]):
        weight = torch.ones([1])
    else: 
        weight = weights[1]
    return weight # they represent: [0, 1]

#Discard gpu fork errors
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print('Training:')

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

# Initialize the CustomAttentionClassifier
# num_output_classes is the number of output classes for your classification problem
model_path = "./model/DNA_bert_3/"
length = 512  # Set the actual length here
custom_model = CustomAttentionClassifier(model_path, length)
custom_model.to(device)

# Define your optimizer
optimizer = AdamW(custom_model.parameters(), lr=2e-5)

# Define your training parameters
num_epochs = 10 
patience = 3
best_loss = float('inf')
num_epochs_without_improvement = 0

# Training loop
for epoch in range(num_epochs):
    custom_model.train().to(device)
    
    loss = 0.0
    for batch in tqdm(train_loader):
        batch_loss = 0.0
        for i in range(len(batch[0])):

            inputs = batch[0][i].unsqueeze(0).to(device)  # Get the input tensors from the batch
            targets = batch[1][i].unsqueeze(0).to(device)  # Get the output tensors from the batch

            masks = batch[2][i].to(device)  # Get the mask tensors from the batch {those are for the lm}
            lm_masks = batch[2][i].unsqueeze(0).to(device)  # Get the mask tensors from the batch {those are for the lm}
            structure_masks = torch.mm(masks.float().unsqueeze(1), masks.float().unsqueeze(0))
            #print(structure_masks.shape)
            
            # Forward pass
            logits = custom_model(input_ids=inputs, attention_mask=lm_masks).reshape(-1, 512, 512)  # Pass the input tensors and masks to the model
            
            unpadded_targets = remove_padding(structure_masks, targets[0]).unsqueeze(0)  # Remove padding from the output tensors
            unpadded_logits = remove_padding(structure_masks, logits[0]).unsqueeze(0)  # Remove padding from the logits
            #print(unpadded_targets.shape, unpadded_logits.shape)

            weight = weigth_calculator(unpadded_targets[0])
            
            criterion = nn.BCEWithLogitsLoss(pos_weight = weight).to(device)
            #print(weight)

            loss = criterion(unpadded_logits, unpadded_targets)
            batch_loss += loss.item()
        
        loss += (batch_loss / len(batch[0]))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate on validation set
    custom_model.eval()
    with torch.no_grad():

        loss = 0.0
        for batch in valid_loader:
            batch_loss = 0.0
            for i in range(len(batch[0])):

                inputs = batch[0][i].unsqueeze(0).to(device)  # Get the input tensors from the batch
                targets = batch[1][i].unsqueeze(0).to(device)  # Get the output tensors from the batch

                masks = batch[2][i].to(device)  # Get the mask tensors from the batch {those are for the lm}
                lm_masks = batch[2][i].unsqueeze(0).to(device)  # Get the mask tensors from the batch {those are for the lm}
                structure_masks = torch.mm(masks.float().unsqueeze(1), masks.float().unsqueeze(0))
                #print(structure_masks.shape)
                
                # Forward pass
                logits = custom_model(input_ids=inputs, attention_mask=lm_masks).reshape(-1, 512, 512)  # Pass the input tensors and masks to the model
                
                unpadded_targets = remove_padding(structure_masks, targets[0]).unsqueeze(0)  # Remove padding from the output tensors
                unpadded_logits = remove_padding(structure_masks, logits[0]).unsqueeze(0)  # Remove padding from the logits
                #print(unpadded_targets.shape, unpadded_logits.shape)

                weight = weigth_calculator(unpadded_targets[0])
                
                criterion = nn.BCEWithLogitsLoss(pos_weight = weight).to(device)
                #print(weight)

                loss = criterion(unpadded_logits, unpadded_targets)
                batch_loss += loss.item()
            
            loss += (batch_loss / len(batch[0]))
        
        avg_loss = loss / len(valid_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_loss}")

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        num_epochs_without_improvement = 0
        # Save the best model
        torch.save(custom_model.state_dict(), 'best_tune_100x.pt')
    else:
        num_epochs_without_improvement += 1
        if num_epochs_without_improvement >= patience:
            print("Early stopping! No improvement for", patience, "epochs.")
            break

print('Evaluating..')

from sklearn.metrics import f1_score

custom_model.eval()
with torch.no_grad():

    total_f1 = 0
    for batch in tqdm(test_loader):

        batch_f1 = 0
        for i in range(len(batch[0])):

            inputs = batch[0][i].unsqueeze(0).to(device)  # Get the input tensors from the batch
            targets = batch[1][i].unsqueeze(0).to(device)  # Get the output tensors from the batch

            masks = batch[2][i].to(device)  # Get the mask tensors from the batch {those are for the lm}
            lm_masks = batch[2][i].unsqueeze(0).to(device)  # Get the mask tensors from the batch {those are for the lm}
            structure_masks = torch.mm(masks.float().unsqueeze(1), masks.float().unsqueeze(0))
            #print(structure_masks.shape)
            
            # Forward pass
            logits = custom_model(input_ids=inputs, attention_mask=lm_masks).reshape(-1, 512, 512)  # Pass the input tensors and masks to the model
            
            unpadded_targets = remove_padding(structure_masks, targets[0]).unsqueeze(0)  # Remove padding from the output tensors
            unpadded_logits = remove_padding(structure_masks, logits[0]).unsqueeze(0)  # Remove padding from the logits
            #print(unpadded_targets.shape, unpadded_logits.shape)
        
            threshold = 0.5
            unpadded_predictions = (unpadded_logits > 0.5).float()

            unpadded_targets = unpadded_targets.squeeze(0).cpu()
            unpadded_predictions = unpadded_predictions.squeeze(0).cpu()
            
            f1 = f1_score(unpadded_targets, unpadded_predictions, average='macro', zero_division=1)
            batch_f1 += f1
        
        total_f1 += (batch_f1 / len(batch[0]))
    avg_f1 = total_f1 / len(test_loader)

print(f"Average F1 score: {avg_f1}")

print('===================================END===================================')
