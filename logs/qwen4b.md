Tl;dr
Qwen-4B necessitates longer 'max_new_tokens' to solve the same problems compared to 0.6B model. 
Strangely, stronger model needs more tokens to solve the same problems, and WHEN we restrict them in token counts, 
they CAN'T solve the same problem which smaller model can. 
So, in a resource (token count) constraint environment, smaller model is definitively better.

- max length: 512
======================================================================
Model: /workspace/arl/output/grpo_qwen3_4b
Samples: 50  |  Correct: 4  |  Incorrect: 46
Accuracy: 8.0%
======================================================================

--- Scalar metrics ---
                        Correct  Incorrect      Delta
-------------------------------------------------------
        mean_logprob    -0.1116    -0.1408    +0.0292
            mean_mbe     1.2104     1.1847    +0.0257
         end_cos_sim     0.0671     0.0865    -0.0194
      completion_len   485.5000   512.0000   -26.5000

--- 1. Confidence (log-prob) (higher = better) ---
  Position    5%   15%   25%   35%   45%   55%   65%   75%   85%   95%
----------------------------------------------------------------------
   Correct -0.08 -0.08 -0.11 -0.15 -0.11 -0.10 -0.09 -0.18 -0.13 -0.09  (n=4)
 Incorrect -0.08 -0.09 -0.13 -0.13 -0.14 -0.15 -0.18 -0.17 -0.17 -0.17  (n=46)
     Delta +0.01 +0.01 +0.01 -0.02 +0.03 +0.05 +0.09 -0.01 +0.03 +0.08

--- 2. MBE (representation diversity) (higher = better) ---
  Position    5%   15%   25%   35%   45%   55%   65%   75%   85%   95%
----------------------------------------------------------------------
   Correct  1.27  1.22  1.23  1.20  1.21  1.20  1.16  1.16  1.21  1.25  (n=4)
 Incorrect  1.26  1.24  1.21  1.19  1.20  1.18  1.12  1.15  1.15  1.16  (n=46)
     Delta +0.01 -0.02 +0.01 +0.01 +0.01 +0.02 +0.04 +0.01 +0.06 +0.08

--- 3. P(correct answer | prefix) (higher = better) ---
  Position    5%   15%   25%   35%   45%   55%   65%   75%   85%   95%
----------------------------------------------------------------------
   Correct -34.02 -33.48 -32.23 -33.15 -32.58 -35.34 -33.68 -32.04 -29.88 -25.77  (n=4)
 Incorrect -34.00 -35.41 -34.05 -31.11 -31.44 -30.20 -28.69 -30.79 -26.74 -25.21  (n=46)
     Delta -0.02 +1.93 +1.82 -2.04 -1.14 -5.14 -5.00 -1.25 -3.14 -0.56

--- 4. End-of-trace cosine similarity to correct answer embedding ---
  Correct:   mean=0.0671  min=0.0619  max=0.0727
  Incorrect: mean=0.0865  min=0.0037  max=0.2805

Saved to /workspace/arl/output/grpo_qwen3_4b/rollout_analysis.json

>>> All done.
>>> Logs:
>>>   vLLM:     /workspace/arl/output/grpo_qwen3_4b/vllm.log
>>>   Training: /workspace/arl/output/grpo_qwen3_4b/train.log
>>>   Analysis: /workspace/arl/output/grpo_qwen3_4b/analysis.log



  Qwen3-4B GRPO Analysis                                                                                                                                               
                                                                                                                                                                       
  The Problem: 4B is barely learning (reward flat ~0.03-0.06)                                                                                                          
                                                                                                                                                                       
  ┌────────────────────────────────┬─────────────┬─────────────┐                                                                                                       
  │             Metric             │    0.6B     │     4B      │                                                                                                       
  ├────────────────────────────────┼─────────────┼─────────────┤                                                                                                       
  │ Reward trajectory              │ 0.13 → 0.73 │ 0.04 → 0.06 │                                                                                                       
  ├────────────────────────────────┼─────────────┼─────────────┤                                                                                                       
  │ Completions clipped at max len │ 0/20 steps  │ 20/20 steps │                                                                                                       
  ├────────────────────────────────┼─────────────┼─────────────┤                                                                                                       
  │ Format reward (#### pattern)   │ 0           │ 0           │
  ├────────────────────────────────┼─────────────┼─────────────┤
  │ Accuracy (analysis)            │ —           │ 8%          │
  ├────────────────────────────────┼─────────────┼─────────────┤
  │ Mean terminated length         │ varies      │ 0–21 tokens │
  └────────────────────────────────┴─────────────┴─────────────┘

  Root Cause: max_completion_length=256 is too short for 4B

  The 4B model generates longer reasoning chains (Qwen3's think tokens). 100% of completions are hitting the 256 token ceiling and getting truncated before the model
  can output #### answer. The model literally never gets to finish, so:
  - Format reward is always 0 (no #### pattern found)
  - Correctness reward is near 0 (answer truncated)
  - No learning signal → reward stays flat

  The 0.6B run used max_completion_length=512 and had 0% clipping.

  Recommended Hyperparameter Changes

● Read 1 file (ctrl+o to expand)    
                       
● Here's what to change for the next 4B run:                                                                                                                           
                                  
  1. max_completion_length 256 → 1024 — the critical fix, let the model finish its reasoning                                                                           
  2. num_generations 4 → 8 — more rollouts per prompt gives better advantage estimation                                                                                
  3. gradient_accumulation_steps 16 → 8 — compensate for longer completions eating more memory                                                                         
  4. per_device_train_batch_size 4 → 4 — keep as is (minimum for num_gen=4, or bump to 8 if using num_gen=8)                                                           
  5. learning_rate 3e-6 → 5e-6 — slightly higher, the model was barely moving   



- Extend 'max_length' to 1024 

======================================================================
Model: /workspace/arl/output/grpo_qwen3_4b_long_cot
Samples: 50  |  Correct: 6  |  Incorrect: 44
Accuracy: 12.0%
======================================================================

--- Scalar metrics ---
                        Correct  Incorrect      Delta
-------------------------------------------------------
        mean_logprob    -0.1232    -0.1351    +0.0119
            mean_mbe     1.2920     1.1975    +0.0945
         end_cos_sim     0.0662     0.0902    -0.0240
      completion_len   483.0000   512.0000   -29.0000

--- 1. Confidence (log-prob) (higher = better) ---
  Position    5%   15%   25%   35%   45%   55%   65%   75%   85%   95%
----------------------------------------------------------------------
   Correct -0.08 -0.08 -0.10 -0.09 -0.13 -0.13 -0.21 -0.18 -0.12 -0.11  (n=6)
 Incorrect -0.08 -0.09 -0.13 -0.13 -0.13 -0.15 -0.15 -0.15 -0.16 -0.18  (n=44)
     Delta +0.01 +0.01 +0.02 +0.04 +0.01 +0.02 -0.06 -0.02 +0.03 +0.07

--- 2. MBE (representation diversity) (higher = better) ---
  Position    5%   15%   25%   35%   45%   55%   65%   75%   85%   95%
----------------------------------------------------------------------
   Correct  1.27  1.30  1.30  1.30  1.24  1.26  1.33  1.27  1.30  1.33  (n=6)
 Incorrect  1.26  1.24  1.22  1.19  1.20  1.20  1.18  1.22  1.15  1.14  (n=44)
     Delta +0.00 +0.06 +0.08 +0.11 +0.05 +0.06 +0.15 +0.05 +0.15 +0.19

--- 3. P(correct answer | prefix) (higher = better) ---
  Position    5%   15%   25%   35%   45%   55%   65%   75%   85%   95%
----------------------------------------------------------------------
   Correct -35.14 -36.61 -30.82 -31.43 -26.09 -28.63 -27.95 -31.17 -29.36 -25.80  (n=6)
 Incorrect -34.39 -36.08 -34.15 -31.25 -29.82 -29.56 -29.07 -29.31 -29.39 -22.93  (n=44)
     Delta -0.75 -0.54 +3.32 -0.18 +3.73 +0.93 +1.13 -1.86 +0.03 -2.88

--- 4. End-of-trace cosine similarity to correct answer embedding ---
  Correct:   mean=0.0662  min=0.0490  max=0.0770
  Incorrect: mean=0.0902  min=0.0101  max=0.2762

Saved to /workspace/arl/output/grpo_qwen3_4b_long_cot/rollout_analysis.json