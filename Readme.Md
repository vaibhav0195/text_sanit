Code for the paper Sanitization is all you need.

data folder contains the raw sentences used for validation, created embeddings for a dataset.

renyi_exps contains the renyi-divergence estimator
dpsgd contains the code to run comparison between dpsgd and our approach
membership inference contains the code to run membership inference attack

To run the renyi divergence experiments please run
    To get the plots for the paper please run "renyi_exps/calc_epsusing_newtheorm.py"
    To create the embeddings please run "renyi_exps/create_embeddings.py"
    To looks for renyi divergence values please run "renyi_exps/run_renyi_final_paper.py"

To run the Dp-SGD comparison please run
    dpsgd/utility_final_seed.py --data=data/sentences/smartMaskingValidationMedalSepTrain_diff_length
    To get results for different datasets please change the data args with
    data/smart_amazon_utility_redaction
    data/smartMaskingPoliticalDataset
    data/smart_masking_redit_suicide

To run membership inference please run
    membership_inference/true_negs_res.py --data=data/sentences/smartMaskingValidationMedalSepTrain_diff_length
    To get results for different datasets please change the data args with
    data/smart_amazon_utility_redaction
    data/smartMaskingPoliticalDataset
    data/smart_masking_redit_suicide
    
