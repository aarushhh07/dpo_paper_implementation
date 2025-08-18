# Direct Preference Optimization : Low-level Paper Implementation
> Link to the paper : [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- This paper introduced a new, less computationally demanding technique to finetune Language Models for human preferences as pposed to RLHF, over a vareity of tasks like summarization, generative sentiment classificaton, single-dialogue tasks etc.
- The proposed loss parameterized the reward model in terms of the language model's parameters, thus directly modifying the parameters under the Bradley-Terry model.
- This is an low-level implementation of the paper, using the same core concepts but lighter models. The original paper used GPT-2 as the policy model, whereas this uses GPT-2 Medium as the policy model
- The task implemented is using the dataset Anthropic HH (HH-RLHF) - One of the (>3) used in the paper for single-turn dialogue generation i.e. x is a human query, which may be anything from a question about astrophysics
to a request for relationship advice. A policy must produce an engaging and helpful response y to
a user’s query. 
  
  ## Training the Reference Model
  - The authors mention that there is no SFT model available for this task so fine-tuning from scratch has to be done, which they have done using instruction output pairs of the form (prompt, y_w) in the Anthropic dataset, and then taking the MLE as the reference model.
  
  - The reference model for this implementation is GPT-2 Medium, in the file "reference_model.py"
  
  ## DPO Fine-Tuning
  - The policy model is initialized to be the reference model and the updates are carried out using the Adam Optimizer
  - Due to GPU constraints, the max sequence length for all sequences is capped out at 128 tokens (instead of the original 512)
  - This caused significant decrease in the training time: from an initial 208 hours to 7 hours
  - The policy model has been trained in the file "policy_model.py"
  >Note: I have loaded the reference model in "policy_model.py"