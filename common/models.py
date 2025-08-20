"""---------------------------------------
Define various types of models
---------------------------------------"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    """ CNN处理obs """

    def __init__(self, h, w, outputs, dropout_rate=0.25):
        super(Qnet, self).__init__()
        
        # --- Convolutional Layers Definition ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=0)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Calculate Flattened Size ---
        flattened_features = self._get_conv_output_flattened_size(h, w)
        
        # --- Fully Connected Layers Definition ---
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=flattened_features, out_features=512)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(in_features=512, out_features=outputs)

    def _get_conv_output_flattened_size(self, h, w):
        with torch.no_grad():
            dummpy_input = torch.zeros(1, 3, h, w)
            x = self.pool1(self.bn1(self.conv1(dummpy_input)))
            x = self.pool2(self.bn2(self.conv2(x)))
            x = self.pool3(self.bn3(self.conv3(x)))

        return int(x.numel()) # 计算张量的元素个数

    def forward(self, x):
        # 输入已经转换为 [Batch, Height, Width, Channels]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(F.relu(self.bn4(self.fc1(x))))
        return self.fc2(x)

class VAQnet(nn.Module):
    """ CNN处理obs """

    def __init__(self, h, w, outputs, dropout_rate=0.25):
        super(VAQnet, self).__init__()
        
        # --- Convolutional Layers Definition ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=0)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Calculate Flattened Size ---
        flattened_features = self._get_conv_output_flattened_size(h, w)
        
        # --- Fully Connected Layers Definition ---
        self.flatten = nn.Flatten()

        self.adv_fc1  = nn.Linear(in_features=flattened_features, out_features=512)
        self.adv_bn   = nn.BatchNorm1d(num_features=512)
        self.adv_drop = nn.Dropout(dropout_rate)
        self.adv_fc2  = nn.Linear(in_features=512, out_features=outputs)

        self.val_fc1  = nn.Linear(in_features=flattened_features, out_features=512)
        self.val_bn   = nn.BatchNorm1d(num_features=512)
        self.val_drop = nn.Dropout(dropout_rate)
        self.val_fc2  = nn.Linear(in_features=512, out_features=1)

    def _get_conv_output_flattened_size(self, h, w):
        with torch.no_grad():
            dummpy_input = torch.zeros(1, 3, h, w)
            x = self.pool1(self.bn1(self.conv1(dummpy_input)))
            x = self.pool2(self.bn2(self.conv2(x)))
            x = self.pool3(self.bn3(self.conv3(x)))

        return int(x.numel()) # 计算张量的元素个数

    def forward(self, x):
        # 输入已经转换为 [Batch, Height, Width, Channels]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)

        advantage_fc1 = self.adv_drop(F.relu(self.adv_bn(self.adv_fc1(x))))
        advantage = self.adv_fc2(advantage_fc1)

        value_fc1 = self.val_drop(F.relu(self.val_bn(self.val_fc1(x))))
        value = self.val_fc2(value_fc1)

        return value + advantage - advantage.mean(dim=1, keepdim=True) 

class ActorSoftmax_net(nn.Module):
    """ CNN处理obs """

    def __init__(self, h, w, outputs, dropout_rate=0.25):
        super(ActorSoftmax_net, self).__init__()
        
        # --- Convolutional Layers Definition ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=0)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Calculate Flattened Size ---
        flattened_features = self._get_conv_output_flattened_size(h, w)
        
        # --- Fully Connected Layers Definition ---
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=flattened_features, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=outputs)

    def _get_conv_output_flattened_size(self, h, w):
        with torch.no_grad():
            dummpy_input = torch.zeros(1, 3, h, w)
            x = self.pool1(self.bn1(self.conv1(dummpy_input)))
            x = self.pool2(self.bn2(self.conv2(x)))
            x = self.pool3(self.bn3(self.conv3(x)))

        return int(x.numel()) # 计算张量的元素个数

    def forward(self, x):
        # 输入已经转换为 [Batch, Height, Width, Channels]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        probs = F.softmax(self.fc2(x), dim=1)
        return probs

class ActorCriticNet(nn.Module):
    """ CNN base shared between Actor and Critic, with separate heads. """
    def __init__(self, h, w, action_dim, dropout_rate=0.25) -> None:
        super(ActorCriticNet, self).__init__()
        self.action_dim = action_dim

        # --- Convolutional Layers Definition (Shared Base) ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=0)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Calculate Flattened Size ---
        flattened_features = self._get_conv_output_flattened_size(h, w)

        # --- Fully Connected Layers Definition (Shared Base) ---
        self.flatten = nn.Flatten()
        self.fc_shared = nn.Linear(in_features=flattened_features, out_features=512)
        self.bn_shared = nn.BatchNorm1d(num_features=512) # BatchNorm for shared layer
        self.dropout = nn.Dropout(dropout_rate)

        # --- Actor Head ---
        self.actor_head = nn.Linear(512, action_dim)

        # --- Critic Head ---
        self.critic_head = nn.Linear(512, 1) # Outputs a single state value

    def _get_conv_output_flattened_size(self, h, w):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, h, w) # Use C, H, W format
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return int(x.numel())
    
    def forward(self, x):
        # Shared CNN Layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)

        # Shared FC layer
        # Handle batch size 1 for BatchNorm1d during training (if applicable, though updates are batched here)
        if self.training and x.shape[0] == 1:
            # Skip BatchNorm and Dropout if batch size is 1 during training inference step
            shared_features = F.relu(self.fc_shared(x))
        elif not self.training and x.shape[0] == 1:
            # Use BatchNorm running stats in eval mode even for batch size 1
            shared_features = F.relu(self.bn_shared(self.fc_shared(x)))
        else: # Batch size > 1 or eval mode with batch size > 1
            shared_features = self.dropout(F.relu(self.bn_shared(self.fc_shared(x))))
        
        # Actor head -> action logits
        action_logits = self.actor_head(shared_features)
        action_probs  = F.softmax(action_logits, dim=1)

        # Critic head -> state value
        state_value = self.critic_head(shared_features)

        return action_probs, state_value

class ActorCriticNetMinigird(nn.Module):
    """ CNN base shared between Actor and Critic, with separate heads. """
    def __init__(self, h, w, action_dim, dropout_rate=0.25) -> None:
        super(ActorCriticNetMinigird, self).__init__()
        self.action_dim = action_dim

        # --- Convolutional Layers Definition (Shared Base) ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # --- Calculate Flattened Size ---
        flattened_features = self._get_conv_output_flattened_size(h, w)

        # --- Fully Connected Layers Definition (Shared Base) ---
        self.flatten = nn.Flatten()
        self.fc_shared = nn.Linear(in_features=flattened_features, out_features=512)
        self.bn_shared = nn.BatchNorm1d(num_features=512) # BatchNorm for shared layer
        self.dropout = nn.Dropout(dropout_rate)

        # --- Actor Head ---
        self.actor_head = nn.Linear(512, action_dim)

        # --- Critic Head ---
        self.critic_head = nn.Linear(512, 1) # Outputs a single state value

    def _get_conv_output_flattened_size(self, h, w):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, h, w) # Use C, H, W format
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        return int(x.numel())
    
    def forward(self, x):
        # Shared CNN Layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)

        # Shared FC layer
        # Handle batch size 1 for BatchNorm1d during training (if applicable, though updates are batched here)
        if self.training and x.shape[0] == 1:
            # Skip BatchNorm and Dropout if batch size is 1 during training inference step
            shared_features = F.relu(self.fc_shared(x))
        elif not self.training and x.shape[0] == 1:
            # Use BatchNorm running stats in eval mode even for batch size 1
            shared_features = F.relu(self.bn_shared(self.fc_shared(x)))
        else: # Batch size > 1 or eval mode with batch size > 1
            shared_features = self.dropout(F.relu(self.bn_shared(self.fc_shared(x))))
        
        # Actor head -> action logits
        action_logits = self.actor_head(shared_features)
        action_probs  = F.softmax(action_logits, dim=1)

        # Critic head -> state value
        state_value = self.critic_head(shared_features)

        return action_probs, state_value

class ActorNet(nn.Module):
    """ CNN base with a policy head (Actor). """
    def __init__(self, h, w, action_dim, dropout_rate=0.25) -> None:
        super(ActorNet, self).__init__()
        self.actiond_dim = action_dim

        # --- Convolutional Layers Definition (Base) ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # --- Calculate Flattened Size ---
        flattened_features = self._get_conv_output_flattened_size(h, w)

        # --- Fully Connected Layers Definition (Base) ---
        self.flatten = nn.Flatten()
        self.fc_base = nn.Linear(in_features=flattened_features, out_features=512)
        self.bn_base = nn.BatchNorm1d(num_features=512) # BatchNorm for base FC layer
        self.dropout = nn.Dropout(dropout_rate)

        # --- Actor Head ---
        self.actor_head = nn.Linear(512, action_dim)

        # Initialize weights (optional, but good practice)
        self._initialize_weights()
    
    def _get_conv_output_flattened_size(self, h, w):
        with torch.no_grad():
            # Create dummy input with Batch, Channel, Height, Width
            dummy_input = torch.zeros(1, 3, h, w)
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        return int(x.numel()) # Total number of elements after conv layers
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)

        x_fc = self.fc_base(x)
        if self.training:
            if x_fc.shape[0] > 1:
                base_features = self.dropout(F.relu(self.bn_base(x_fc)))
            else:
                # Skip BatchNorm and Dropout for batch_size 1 during training
                # to avoid error and because dropout isn't meaningful.
                base_features = F.relu(x_fc)
        else: # Evaluation mode
            base_features = F.relu(self.bn_base(x_fc)) 
        
        action_logits = self.actor_head(base_features)
        action_probs = F.softmax(action_logits, dim=1)

        return action_probs

class CriticNet(nn.Module):
    """ CNN base with a value head (Critic). """
    def __init__(self, h, w, dropout_rate=0.25) -> None:
        super(CriticNet, self).__init__()

        # --- Convolutional Layers Definition (Base) ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # --- Calculate Flattened Size ---
        flattened_features = self._get_conv_output_flattened_size(h, w)

        # --- Fully Connected Layers Definition (Base) ---
        self.flatten = nn.Flatten()
        self.fc_base = nn.Linear(in_features=flattened_features, out_features=512)
        self.bn_base = nn.BatchNorm1d(num_features=512) # BatchNorm for base FC layer
        self.dropout = nn.Dropout(dropout_rate)

        # --- Critic Head ---
        self.critic_head = nn.Linear(512, 1) # Outputs a single state value

        # Initialize weights (optional, but good practice)
        self._initialize_weights()

    def _get_conv_output_flattened_size(self, h, w):
        with torch.no_grad():
            # Create dummy input with Batch, Channel, Height, Width
            dummy_input = torch.zeros(1, 3, h, w)
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        return int(x.numel()) # Total number of elements after conv layers
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)

        x_fc = self.fc_base(x)
        if self.training:
            if x_fc.shape[0] > 1:
                base_features = self.dropout(F.relu(self.bn_base(x_fc)))
            else:
                base_features = F.relu(x_fc) # Skip BN and Dropout
        else: # Evaluation mode
            base_features = F.relu(self.bn_base(x_fc))

        state_value = self.critic_head(base_features)

        return state_value

if __name__ == "__main__":
    test_q = Qnet(56, 56, 7)