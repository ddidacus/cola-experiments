#### Changes
- Refactored training code
- implemented hyperparam search 
- fixed inconsistencies in alpha/beta hparams
- fixed bug and added support for arbitrary n-th order HOLA

#### Issues
- incomplete code: 
    - not all the configurations are provided
    - inconsistent use of look ahead rate with two variables: alpha, beta
    - alpha used arbitrarily as both alpha look ahead and consistency loss alpha coefficient
    - independent runs were not running under different seeds, it was just an outer loop
- looks like they cherry-picked the alpha parameter, because it's a specific value in [0.5, 0.6] that causes CGD and SOS to have high variance
- no variance over consistency analysis
- no implementation of COLA_short
- fixed bug from original notebook in loss reporting for IPD 


#### Experiments to run
GAMES = [
    "Matching Pennies", 
    "IPD", 
    "Ultimatum", 
    "Chicken Game", 
    "Tandem", 
    "Balduzzi", 
    "Hamiltonian"
]

- [ ] try smaller LR for consistency losses in non-polynomial games (matching pennies, ipd, ultimatum, chicken game)
- [ ] try smaller look-ahead rate for CDG to see if it is really the case
- [ ] analyze the probabilities learned with Ultimatum
- [ ] smaller look-aheads with Ultimatum?
- [ ] can we do round-robin tournaments and see equivalency of COLA with HOLA?