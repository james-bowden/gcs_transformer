# gcs_transformer
Transformer to mimic generation of iterative convex solver solutions from GCS, on UAV navigation task.


# Data 

To tokenize data, use the `data/tokenize_data.ipynb` script

To split the data into train and eval splits, run `python split_data.py --data_folder <your_data_folder> --train_fraq X (--overwrite if overwriting split)

# Training 

To train the policy, first create your own config from the default `config/default.yaml`

Train the policy with `python train.py -c config/<your config>