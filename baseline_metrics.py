'''
CE Loss
AUC-ROC
Accuracy
'''
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torch.optim.lr_scheduler import StepLR
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import accuracy_score

# # **Step 1: Prepare Data - Create Pairs (Question + Choice)**
# train_pairs = []
# train_labels = []

# val_pairs = []
# val_labels = []

# for i, row in train_df.iterrows():
#     question = row["question"]
#     choices = row["choices"]
#     correct_idx = row["correct_choice_idx"]  # This is the correct choice index

#     for idx, choice in enumerate(choices):
#         text_input = question + " " + choice  # Combine Question + Choice
#         train_pairs.append(text_input)
#         train_labels.append(1 if idx == correct_idx else 0)  # Label: 1 if correct, 0 otherwise

# for i, row in val_df.iterrows():
#     question = row["question"]
#     choices = row["choices"]
#     correct_idx = row["correct_choice_idx"]

#     for idx, choice in enumerate(choices):
#         text_input = question + " " + choice
#         val_pairs.append(text_input)
#         val_labels.append(1 if idx == correct_idx else 0)

# # **Step 2: Compute TF-IDF Embeddings**
# vectorizer = TfidfVectorizer(max_features=10000)
# X_train = vectorizer.fit_transform(train_pairs).toarray()
# X_val = vectorizer.transform(val_pairs).toarray()

# y_train = np.array(train_labels)
# y_val = np.array(val_labels)

# # **Step 3: Convert Data to PyTorch Tensors**
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # Binary classification

# X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# # **Step 4: Create DataLoader**
# batch_size = 128
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# # **Step 5: Define MLP Model for Scoring Choices**
# class ChoiceScoringMLP(nn.Module):
#     def __init__(self, input_dim):
#         super(ChoiceScoringMLP, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 1)  # Output a single score for the choice
#         )

#     def forward(self, x):
#         return self.model(x).squeeze(1)  # Ensure output shape is (batch,)

# # **Step 6: Initialize Model**
# input_dim = X_train.shape[1]  # TF-IDF feature size
# model_mlp = ChoiceScoringMLP(input_dim).to(device)

# # **Step 7: Define Loss, Optimizer, and Scheduler**
# criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
# optimizer = optim.Adam(model_mlp.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

# # **Step 8: Train MLP with DataLoader & Scheduler**
# num_epochs = 40
# for epoch in range(num_epochs):
#     model_mlp.train()
#     epoch_loss = 0

#     for batch_X, batch_y in train_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)

#         outputs = model_mlp(batch_X)  # Get scores
#         loss = criterion(outputs, batch_y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#     scheduler.step()
#     current_lr = scheduler.get_last_lr()[0]
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, LR: {current_lr:.6f}")

# **Step 9: Evaluate Model**
# model_mlp.eval()
# y_pred_scores = []

# with torch.no_grad():
#     for batch_X, _ in val_loader:
#         batch_X = batch_X.to(device)
#         batch_scores = model_mlp(batch_X).cpu().numpy()
#         y_pred_scores.extend(batch_scores)

# # **Step 10: Compute Accuracy (Choice Selection)**
# val_choice_indices = []  # Stores predicted choice index for each question
# val_correct_indices = []  # Stores ground-truth indices

# i = 0
# for idx, row in val_df.iterrows():
#     num_choices = len(row["choices"])
#     scores = y_pred_scores[i:i + num_choices]  # Get scores for this question's choices
#     pred_choice_idx = np.argmax(scores)  # Pick highest-scoring choice
#     val_choice_indices.append(pred_choice_idx)
#     val_correct_indices.append(row["correct_choice_idx"])  # Ground-truth index
#     i += num_choices

# Final Accuracy Computation
bert_accuracy = accuracy_score(val_correct_indices, val_choice_indices)
print(f"TF-IDF + Choice-Based MLP Accuracy: {bert_accuracy:.2%}")

# Convert scores to probabilities using Sigmoid since BCEWithLogitsLoss was used
y_pred_probs = torch.sigmoid(torch.tensor(y_pred_scores)).numpy()

# Cross-Entropy Loss (Log Loss)
cross_entropy = log_loss(y_val, y_pred_probs)
print(f"Cross-Entropy Loss: {cross_entropy:.4f}")

# Precision, Recall, and F1 Score

precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, y_pred_probs)
best_threshold = thresholds[np.argmax(precision_vals * recall_vals)]  # F1-optimal threshold
y_pred_binary = (y_pred_probs >= best_threshold).astype(int)

precision = precision_score(y_val, y_pred_binary)
recall = recall_score(y_val, y_pred_binary)
f1 = f1_score(y_val, y_pred_binary)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")