import datetime
import functools
import json
import os
import re
import sys
import time

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

from absl import app
from absl import flags
import google.generativeai as palm
import numpy as np
import openai
import matplotlib.pyplot as plt
from bandits import BernoulliBandit

from opro import prompt_utils

_OPENAI_API_KEY = flags.DEFINE_string(
    "openai_api_key", "", "The OpenAI API key."
)

_PALM_API_KEY = flags.DEFINE_string("palm_api_key", "", "The PaLM API key.")

_OPTIMIZER = flags.DEFINE_string(
    "optimizer", "gpt-3.5-turbo", "The name of the optimizer LLM."
)


#Solver object taken from multi-armed bandit repo
class Solver(object):
    def __init__(self, bandit):
        """
        bandit (Bandit): the target bandit to solve.
        """
        assert isinstance(bandit, BernoulliBandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.n
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.

    def update_regret(self, i):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_proba - self.bandit.probas[i]
        self.regrets.append(self.regret)

    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)

class GPT(Solver):
    def __init__(self, bandit):
        super(GPT, self).__init__(bandit)

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self, input):
        i = input
        r = self.bandit.generate_reward(i)
        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i


#Plot function taken from multi-armed bandit repo
def plot_results(solvers, solver_names, figname):
    """
    Plot the results by multi-armed bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit

    fig = plt.figure(figsize=(14, 4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Sub.fig. 1: Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 2: Probabilities estimated by solvers.
    sorted_indices = sorted(range(b.n), key=lambda x: b.probas[x])
    ax2.plot(range(b.n), [b.probas[x] for x in sorted_indices], 'k--', markersize=12)
    for s in solvers:
        ax2.plot(range(b.n), [s.estimated_probas[x] for x in sorted_indices], 'x', markeredgewidth=2)
    ax2.set_xlabel('Actions sorted by ' + r'$\theta$')
    ax2.set_ylabel('Estimated')
    ax2.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 3: Action counts
    for s in solvers:
        ax3.plot(range(b.n), np.array(s.counts) / float(len(solvers[0].regrets)), ls='steps', lw=2)
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('Frac. # trials')
    ax3.grid('k', ls='--', alpha=0.3)

    plt.savefig(figname)




def main(_):
  # ============== set optimization experiment configurations ================
  num_steps = 1
  K = 3
  b = BernoulliBandit(K)
  print ("Randomly generated Bernoulli bandit has reward probabilities:\n", b.probas)
  print ("The best machine has index: {} and proba: {}".format(
        max(range(K), key=lambda i: b.probas[i]), max(b.probas)))

  # ================ load LLM settings ===================
  optimizer_llm_name = _OPTIMIZER.value
  assert optimizer_llm_name in {
      "text-bison",
      "gpt-3.5-turbo",
      "gpt-4",
  }
  openai_api_key = _OPENAI_API_KEY.value
  palm_api_key = _PALM_API_KEY.value

  if optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4"}:
    assert openai_api_key, "The OpenAI API key must be provided."
    openai.api_key = openai_api_key
  else:
    assert optimizer_llm_name == "text-bison"
    assert (
        palm_api_key
    ), "A PaLM API key is needed when prompting the text-bison model."
    palm.configure(api_key=palm_api_key)

  # =================== create the result directory ==========================
  datetime_str = (
      str(datetime.datetime.now().replace(microsecond=0))
      .replace(" ", "-")
      .replace(":", "-")
  )

  save_folder = os.path.join(
      OPRO_ROOT_PATH,
      "outputs",
      "optimization-results",
      f"multi_armed_bandit-o-{optimizer_llm_name}-{datetime_str}/",
  )
  os.makedirs(save_folder)
  print(f"result directory:\n{save_folder}")


  # ====================== optimizer model configs ============================
  if optimizer_llm_name.lower() == "text-bison":
    # when prompting text-bison with Cloud API
    optimizer_finetuned_palm_temperature = 1.0
    optimizer_finetuned_palm_max_decode_steps = 1024
    optimizer_finetuned_palm_batch_size = 1
    optimizer_finetuned_palm_num_servers = 1
    optimizer_finetuned_palm_dict = dict()
    optimizer_finetuned_palm_dict["temperature"] = (
        optimizer_finetuned_palm_temperature
    )
    optimizer_finetuned_palm_dict["batch_size"] = (
        optimizer_finetuned_palm_batch_size
    )
    optimizer_finetuned_palm_dict["num_servers"] = (
        optimizer_finetuned_palm_num_servers
    )
    optimizer_finetuned_palm_dict["max_decode_steps"] = (
        optimizer_finetuned_palm_max_decode_steps
    )

    call_optimizer_finetuned_palm_server_func = functools.partial(
        prompt_utils.call_palm_server_from_cloud,
        model="text-bison-001",
        temperature=optimizer_finetuned_palm_dict["temperature"],
        max_decode_steps=optimizer_finetuned_palm_dict["max_decode_steps"],
    )

    optimizer_llm_dict = {
        "model_type": optimizer_llm_name.lower(),
    }
    optimizer_llm_dict.update(optimizer_finetuned_palm_dict)
    call_optimizer_server_func = call_optimizer_finetuned_palm_server_func

  else:
    assert optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4"}
    optimizer_gpt_max_decode_steps = 1024
    optimizer_gpt_temperature = 1.0

    optimizer_llm_dict = dict()
    optimizer_llm_dict["max_decode_steps"] = optimizer_gpt_max_decode_steps
    optimizer_llm_dict["temperature"] = optimizer_gpt_temperature
    optimizer_llm_dict["batch_size"] = 1
    call_optimizer_server_func = functools.partial(
        prompt_utils.call_openai_server_func,
        model=optimizer_llm_name,
        max_decode_steps=optimizer_gpt_max_decode_steps,
        temperature=optimizer_gpt_temperature,
    )

  # ====================== try calling the servers ============================
  print("\n======== testing the optimizer server ===========")
  optimizer_test_output = call_optimizer_server_func(
      "Does the sun rise from the north? Just answer yes or no.",
      temperature=1.0,
  )
  print(f"optimizer test output: {optimizer_test_output}")
  print("Finished testing the optimizer server.")
  print("\n=================================================")


  # ====================== utility functions ============================
# Define your action sequences and regrets
  actions_0 = [0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  regret_0 = [0.0, 0.0, 0.0, 0.1487980469591451, 0.2975960939182902, 0.4165547722561752, 0.4165547722561752, 0.4165547722561752, 0.4165547722561752, ...]
  cumulative_regret_0 = 0.4165547722561752

  actions_2 = [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 0, 0, 1, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0, 1, 1, 0, 2]
  regret_2 = [0.0, 0.0, 0.1487980469591451, 0.2677567252970301, 0.2677567252970301, 0.2677567252970301, 0.2677567252970301, ...]
  cumulative_regret_2 = 3.1828425387959474

  actions_1 = [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  regret_1 = [0.0, 0.0, 0.1487980469591451, 0.2975960939182902, 0.4165547722561752, 0.5355134505940602, 0.6544721289319452, ...]
  cumulative_regret_1 = 2.2904530531882314

  prev_actions = [[actions_0, regret_0, cumulative_regret_0], [actions_1, regret_1, cumulative_regret_1], [actions_2, regret_2, cumulative_regret_2]]

  new_actions = []  # List for new sequence
  
  def gen_meta_prompt(
      prev_three_actions,
      max_num_pairs=3,
  ):
    """Generate the meta-prompt for optimization.

    Args:
     old_value_pairs_set (set): the set of old results (sequence, regret).
     max_num_pairs (int): the maximum number of exemplars in the meta-prompt.

    Returns:
      meta_prompt (str): the generated meta-prompt.
    """

    # Generate new sequence based on the logic provided above
    meta_prompt = f"""
    Given the following sequences of actions and their corresponding cumulative regrets, where lower cumulative regret is better, generate a new sequence of size 50 that differs from the previous ones, and results in a lower cumulative regret than the previous sequences:

    Trial 0:
    Actions: {prev_three_actions[0][0]}
    Regret: {prev_three_actions[0][1]}
    Cumulative regret: {prev_three_actions[0][2]}

    Trial 1:
    Actions: {prev_three_actions[1][0]}
    Regret: {prev_three_actions[1][1]}
    Cumulative regret: {prev_three_actions[1][2]}

    Trial 2:
    Actions: {prev_three_actions[2][0]}
    Regret: {prev_three_actions[2][1]}
    Cumulative regret: {prev_three_actions[2][2]}

    Generate three new sequences of 50 actions each (between 0, 1, or 2) that does not repeat the previous sequences but results in a lower cumulative regret than {cumulative_regret_2}.

    The output should be a simple list of actions, e.g., [0, 1, 2, 0, 1, ...]. Nothing else.
    """
    print(meta_prompt)

    return meta_prompt

  # ====================== run optimization ============================
  def extract_string(input_string):
    start_string = "["
    end_string = "]"
    if start_string not in input_string:
      return ""
    input_string = input_string[input_string.index(start_string) + len(start_string):]
    if end_string not in input_string:
      return ""
    input_string = input_string[:input_string.index(end_string)]
    parsed_list = []
    for p in input_string.split(","):
      p = p.strip()
      try:
        p = int(p)
      except:
        continue
      parsed_list.append(p)
    return parsed_list
  
  for i_step in range(num_steps):
    print(f"\nStep {i_step}:")
    meta_prompt = gen_meta_prompt(
        prev_actions
    )
    print(f"meta_prompt:\n{meta_prompt}")
    raw_outputs = []
    parsed_outputs = []
    while len(parsed_outputs) < 3:
        GPT = GPT(b)
        raw_output = call_optimizer_server_func(meta_prompt)
        for string in raw_output:
            print("raw output:\n", string)
            try:
                parsed_output = extract_string(string)
                parsed_outputs.append(parsed_output)
                raw_outputs.append(string)
            except:
                pass
        raw_outputs.append(raw_output)
        parsed_outputs.append(parsed_output)
    print(f"proposed points: {parsed_outputs}")
    for output in parsed_output:
        


if __name__ == "__main__":
  app.run(main)
