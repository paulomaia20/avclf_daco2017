--------------------------------------------------------------------------------
Best pipeline: 150467
--------------------------------------------------------------------------------
----------------------  ------------------------------------
Validation performance  Inner-loop scores
----------------------  ------------------------------------
                                    OF0       OF1       OF2
 min    0.9808            min    0.9689    0.9684    0.9683
 25%     0.981            25%    0.9696    0.9689    0.9693
 50%    0.9812            50%    0.9704    0.9693    0.9702
 75%    0.9812            75%    0.9708    0.9695    0.9706
 max    0.9812            max    0.9713    0.9698    0.9711
----------------------  ------------------------------------
mean    0.9811           mean    0.9702    0.9692    0.9699
 std 0.0002203            std   0.00121 0.0006934  0.001408
----------------------  ------------------------------------
 OF0    0.9812            IF0    0.9713    0.9684    0.9702
 OF1    0.9812            IF1    0.9704    0.9698    0.9711
 OF2    0.9808            IF2    0.9689    0.9693    0.9683
----------------------  ------------------------------------
--------------------------------------------------------------------------------
Pipeline steps
---------------
scaler:
StandardScaler(copy=True, with_mean=True, with_std=True)

estimator:
RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
    max_depth=None, max_features=4, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_impurity_split=None,
    min_samples_leaf=1, min_samples_split=10,
    min_weight_fraction_leaf=0.0, n_estimators=22, n_jobs=1,
    oob_score=False, random_state=None, verbose=0,
    warm_start=False)

--------------------------------------------------------------------------------
Nested k-fold cross-validation parameters
-----------------------------------------
scoring metric:                 auc

scoring type:                   median

outer-fold count:               3
inner-fold count:               3

shuffle seed:                   3243
outer-loop split seed:          45
inner-loop split seeds:         62, 207, 516

random combinations:            50
random combination seed:        2374
--------------------------------------------------------------------------------
