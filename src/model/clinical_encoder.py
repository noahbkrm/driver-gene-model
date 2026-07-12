"""Clinical encoder - arm A (discrete embedding) of the two-arm study. 

Contract
--------
Turns per-patient clinical covariates into a small set of learned token
vectors that live in the same space as the RNA/SNV/CNV tokens, so the
shared attention-pooling module can treat them uniformly. 

Input columns (5, from `cohort.clinical`)
    Categorical (lookup-table embedded):



"""