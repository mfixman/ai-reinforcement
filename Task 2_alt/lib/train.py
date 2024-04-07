import torch
import torch.nn as nn


def optimize_model(memory, transition, policy_net, target_net, optimizer, gamma, batch_size, device, loss_func):
    if len(memory) > batch_size:
        transitions = memory.sample(batch_size)

        batch = transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)

        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        loss_func = loss_func
        loss = loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

        # Back propogation
        optimizer.zero_grad()
        loss.backward()

        # Clips gradient
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        
        # Step next for optimizer
        optimizer.step()
