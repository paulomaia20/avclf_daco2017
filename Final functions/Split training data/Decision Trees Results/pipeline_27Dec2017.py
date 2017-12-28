--------------------------------------------------------------------------------
Best pipeline: 44
--------------------------------------------------------------------------------
----------------------  ------------------------------------
Validation performance  Inner-loop scores
----------------------  ------------------------------------
                                    OF0       OF1       OF2
 min    0.8991            min    0.8678    0.8649    0.8684
 25%    0.9008            25%    0.8684    0.8657    0.8713
 50%    0.9025            50%    0.8691    0.8665    0.8742
 75%     0.906            75%    0.8725     0.867    0.8769
 max    0.9094            max    0.8759    0.8674    0.8797
----------------------  ------------------------------------
mean    0.9037           mean    0.8709    0.8663    0.8741
 std  0.005272            std  0.004371   0.00125  0.005638
----------------------  ------------------------------------
 OF0    0.9025            IF0    0.8759    0.8674    0.8684
 OF1    0.9094            IF1    0.8691    0.8665    0.8742
 OF2    0.8991            IF2    0.8678    0.8649    0.8797
----------------------  ------------------------------------
--------------------------------------------------------------------------------
Pipeline steps
---------------
scaler:
StandardScaler(copy=True, with_mean=True, with_std=True)

feature_selection:
SelectKBest(k=60, score_func=<function f_classif at 0x000002F76D4F0048>)

estimator:
SVC(C=32, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.125, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

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

random combinations:            20
random combination seed:        2374
--------------------------------------------------------------------------------
