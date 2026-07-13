# TBD
# (1). with every GRPO objective, add-in 'forgettable' query's cached rollout's policy gradient
#      (composite objective) 
# (2). with every GRPO group, add-in 'forgettable' queries. 
# (3). with every GRPO group, add-in the queries from 'forgettable' queries set with the closest 
#      representation to the current query in-group. 

# [Insight I like: compression -> interference]
# churn effect comes from compression of representations
# - when 2 queries got similar representation but different continuation / prediction
# this is bad 'churning'
# - when 2 queries got similar representation AND similar continuations, learn on one generalize to another
# this is good 'churning'

# [Experimental Validation]
# XOR training reveals 2 insights: 
# - when similar input necessitates different target, churning occurs
# - when similar input necessitates different target, interleaving conflicting samples are critical 
#   for learning success (avoidance of interference)
# A hypothesis here, is that GRPO / SFT suffers from interference, because queries that are 'similar' (in representation) to
# other queries, are not getting constantly interleaved into training batch / steps, this lead to the model's "churning" or 
# "constant forgetting". 