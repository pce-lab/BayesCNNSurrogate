############### Configuration file for Bayesian ###############
layer_type = 'lrt'  # 'bbb' or 'lrt'
activation_type = 'softplus'  # 'softplus' or 'relu'

n_epochs = 500
lr_start = 0.003
num_workers = 4
train_ens = 1
valid_ens = 1
beta_type = 0.0001   # 'Blundell', 'Standard', etc. Use float for const value
