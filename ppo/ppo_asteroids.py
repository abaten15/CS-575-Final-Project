import gym

import datetime
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

##### CONSTANTS
clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95

# Advantage Calculations
def get_advantages(values, masks, rewards):
  returns = []
  gae = 0
  for i in reversed(range(len(rewards))):
    delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
    gae = delta + gamma * lmbda * masks[i] * gae
    returns.insert(0, gae + values[i])

  adv = np.array(returns) - values[:-1]
  return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

## PPO Loss function

def ppo_loss(oldpolicy_probs, advantages, rewards, values):
  def loss(y_true, y_pred):
    newpolicy_probs = y_pred
    ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
    p1 = ratio * advantages
    p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
    actor_loss = -K.mean(K.minimum(p1, p2))
    critic_loss = K.mean(K.square(rewards - values))
    total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean( -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
    return total_loss

  return loss



# Actor model
def get_model_actor_simple(input_dims, output_dims):
  state_input = Input(shape=input_dims)
  oldpolicy_probs = Input(shape=(1, output_dims,))
  advantages = Input(shape=(1, 1,))
  rewards = Input(shape=(1, 1,))
  values = Input(shape=(1, 1,))
  n_actions = output_dims

  x = Dense(128, activation='relu', name='fc1')(state_input)
  x = Dense(32, activation='relu', name='fc2')(x)
  out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

#  model = Model(inputs=[state_input], outputs=[out_actions])
  model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values], outputs=[out_actions])
#  model.compile(optimizer=Adam(lr=1e-4), loss='mse')
  model.compile(optimizer=Adam(lr=1e-3), loss=[ppo_loss(
    oldpolicy_probs=oldpolicy_probs,
    advantages=advantages,
    rewards=rewards,
    values=values)])

  return model

# Critic model
def get_model_critic_simple(input_dims):
  state_input = Input(shape=input_dims)

  x = Dense(128, activation='relu', name='fc1')(state_input)
  x = Dense(32, activation='relu', name='fc2')(x)
  out_actions = Dense(1, activation='tanh')(x)

  model = Model(inputs=[state_input], outputs=[out_actions])
  model.compile(optimizer=Adam(lr=1e-3), loss='mse') 

  return model

def test_reward(env, actor_model, dummy_n, dummy_1):
    state = env.reset()
    done = False
    total_reward = 0
    print('testing...')
    limit = 0
    while not done:
        state_input = K.expand_dims(state, 0)
#        action_probs = actor_model.predict([state_input], steps=1)
        action_probs = actor_model.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        action = np.argmax(action_probs)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        limit += 1
        if limit >= 200:
          break
    return total_reward

def plot_list(list):
  filename = "asteroids_plot.png"
  array = np.array(list)
  plt.plot(array)
  plt.xlabel("iteration")
  plt.ylabel("time (microseconds)")
  plt.savefig(filename)

###############
# PPO Algorithm
###############

def ppo_algorithm():

  env = gym.make('Asteroids-ram-v0')
  state = env.reset()

  state_dims = env.observation_space.shape
  n_actions = env.action_space.n

  dummy_n = np.zeros((1,1,n_actions))
  dummy_1 = np.zeros((1,1,1))


  actor_model = get_model_actor_simple(state_dims, n_actions)
  critic_model = get_model_critic_simple(state_dims)

  ppo_steps = 256
  target_reached = False
  best_reward = 0
  iters = 0
  max_iters = 500

  total_time_elapsed = 0
  average_time_per_iter = 0

  average_time_array = []

  while not target_reached and iters < max_iters:

    start_time = datetime.datetime.now()

    states = []
    actions = []
    values = []
    masks = []
    rewards = []
    actions_probs = []
    actions_onehot = []
    state_input = None

    for itr in range(ppo_steps):
      env.render()
      state_input = K.expand_dims(state, 0)
#      action_dist = actor_model.predict([state_input], steps=1)
      action_dist = actor_model.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
      q_value = critic_model.predict([state_input], steps=1)
      action = np.random.choice(n_actions, p=action_dist[0, :])
      action_onehot = np.zeros(n_actions)
      action_onehot[action] = 1

      observation, reward, done, info = env.step(action)
      mask = not done

      states.append(state)
      actions.append(action)
      actions_onehot.append(action_onehot)
      values.append(q_value)
      masks.append(mask)
      rewards.append(reward)
      actions_probs.append(action_dist)
      
      state = observation
      if done:
        env.reset()

    q_value = critic_model.predict(state_input, steps=1)
    values.append(q_value)
    returns, advantages = get_advantages(values, masks, rewards)
    
#    actor_loss = actor_model.fit([states], [(np.reshape(actions_onehot, newshape=(-1, n_actions)))], verbose=True, shuffle=True, epochs=20)

    actor_loss = actor_model.fit(
      [states, actions_probs, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
      [(np.reshape(actions_onehot, newshape=(-1, n_actions)))], verbose = True, shuffle=True, epochs=20)
    critic_loss = critic_model.fit([states], [np.reshape(returns, newshape=(-1,1))], shuffle=True, epochs=20, verbose=True)

    iters = iters + 1

    end_time = datetime.datetime.now()

    elapsed = (end_time - start_time)
    micros_elapsed = elapsed.microseconds
    total_time_elapsed = total_time_elapsed + micros_elapsed
    average_time_per_iter = total_time_elapsed / iters
    average_time_array.append(average_time_per_iter)
    print("average_time_per_iter = {}\n".format(average_time_per_iter))

    avg_reward = np.mean([test_reward(env, actor_model, dummy_n, dummy_1) for _ in range(5)])
    print('total test reward=' + str(avg_reward))
    if avg_reward > best_reward:
        print('best reward=' + str(avg_reward))
        best_reward = avg_reward
    if best_reward > 1200 or iters > max_iters:
        target_reached = True
    env.reset()

  
  ave_file = open("average_time_file.txt", "w")
  for i in range(len(average_time_array)):
    ave_file.write("{}\n".format(average_time_array[i]))
  ave_file.write("total_time = {}\n".format(total_time_elapsed))
  ave_file.close()  

  plot_list(average_time_array)
    
  actor_model.save("actor_{}_its.hdf5".format(iters))
    
  critic_model.save("critic_{}_its.hdf5".format(iters))
        



ppo_algorithm()


