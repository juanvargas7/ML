import torch
from torch.utils.tensorboard import SummaryWriter

class MyNetwork(torch.nn.Module):

    def __init__(self, nInputChannels, nOutputClasses, learningRate=1E-2, nEpochs=100):
        super().__init__()

        self.nInputChannels = nInputChannels
        self.nOutputClasses = nOutputClasses
        self.trainingDevice = 'cuda'
        
        self.nEpochs = nEpochs
        self.learningRate = learningRate

        

        ## Image input is nSamples x 3 x 28 x 28


        ## Add your convolutional architecture
        
        self.conv = torch.nn.Sequential(
            
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Output size: [batch_size, 32, 28, 28]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output size: [batch_size, 64, 14, 14]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64,2,1)
        )

        self.conv = torch.nn.Sequential(
            # First Convolutional Block
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Output: [batch_size, 32, 28, 28]
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Additional Conv Layer
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2,2),  # Output: [batch_size, 32, 14, 14]
            torch.nn.Dropout(0.1),  # Dropout for Regularization

            # Second Convolutional Block
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: [batch_size, 64, 14, 14]
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Additional Conv Layer
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # Output: [batch_size, 64, 7, 7]
            torch.nn.Dropout(0.1),  # Dropout for Regularization

            # Third Convolutional Block
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: [batch_size, 128, 7, 7]
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Additional Conv Layer
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # Output: [batch_size, 128, 3, 3]
            torch.nn.Dropout(0.1),  # Dropout for Regularization

            # Final Convolutional Layer
            torch.nn.Conv2d(128, 20, kernel_size=3),
            torch.nn.ReLU()  
        )



        ## Add your fully connected architecture
        self.fc = torch.nn.Sequential(torch.nn.Linear(20, 32),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(32,32),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(32,15),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(15, self.nOutputClasses)
        )

    def forward(self, x):

        # Calculating convolutional features
        x = self.conv(x)

        # Flattening to feed to fully connected layers
        x = x.view(x.size(0), -1)

        # Making predictions
        x = self.fc(x)

        return x

 

   
    def trainModel(self, trainLoader, validationLoader,logPath):


        # Moving to training device
        device = torch.device(self.trainingDevice)
        self.to(device=device)

        ## Define your loss and optimizer here
        loss = torch.nn.CrossEntropyLoss()
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.learningRate)
        optimizer = torch.optim.Adam(self.parameters(), lr= self.learningRate)

        # Creating logger
        writer = SummaryWriter(logPath)

        total_loss = 0.0

        ## Iterating through epochs
        for epoch in range(self.nEpochs):

            epoch_loss = 0
            epoch_accuracy = 0
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            ## TRAINING PHASE
            self.train()
            for batch, (inputs, targets) in enumerate(trainLoader):
                batchLoss = 0
                optimizer.zero_grad()
                targets = targets.squeeze(1)
                # Making targets a vector with labels instead of a matrix and sending to gpu
                inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)

                ## Forward operation here
                y_pred = self.forward(inputs)

                ##Compute loss 
                batchLoss = loss(y_pred, targets)

                ## Bakpropagation operations here
                
                batchLoss.backward()


                ## Parameter updates here
                
                optimizer.step()


                ## Log information (screen and/or Tensorboard)
                total_loss += batchLoss.item()  

                _, predicted = torch.max(y_pred.data, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
            epoch_loss = total_loss / len(trainLoader)
            epoch_accuracy = 100.0 * total_correct / total_samples
            print(f"Epoch: {epoch} training Batch Loss: {epoch_loss}  Accuracy: {epoch_accuracy}")


            # Training summary writes
            writer.add_scalar('Loss/train', epoch_loss, epoch) # on validation
            writer.add_scalar('Accuracy/train', epoch_accuracy, epoch) #on validation






            ## VALIDATION PHASE
            self.eval()
            epoch_loss_val = 0
            epoch_accuracy_val = 0
            total_val_samples = 0
            total_val_correct = 0

            for inputs, targets in validationLoader:    
                with torch.no_grad():
                    inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)
                    targets = targets.squeeze(1)
                    y_pred = self.forward(inputs)

                    # Compute loss
                    evalLoss = loss(y_pred, targets)
                    epoch_loss_val += evalLoss.item() * inputs.size(0)  # Accumulate the loss

                    # Calculate accuracy
                    _, predicted = torch.max(y_pred, 1)
                    total_val_correct += (predicted == targets).sum().item()
                    total_val_samples += targets.size(0)

                # Calculate average loss and accuracy over the epoch
            epoch_loss_val /= total_val_samples
            epoch_accuracy_val = 100.0 * total_val_correct / total_val_samples

                        # Validation
            writer.add_scalar('Accuracy/val', epoch_accuracy_val, epoch)
            writer.add_scalar('Loss/val', epoch_loss_val, epoch) # on validation

            print(f'Validation Loss: {epoch_loss_val:.4f}, Validation Accuracy: {epoch_accuracy_val:.2f}%')
        writer.close()


    def save(self, path):

        ## Saving the model
        torch.save(self.state_dict(), path)