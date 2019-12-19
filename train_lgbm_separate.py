import sympy as sp


def make_lgbm_obj(weights):

    m = len(weights)

    p = [sp.Symbol(f'p{i}') for i in range(m)]
    y = [sp.Symbol(f'y{i}') for i in range(m)]
    N = [sp.Symbol(f'N{i}') for i in range(m)]
    w = [sp.Symbol(f'w{i}') for i in range(m)]

    # This the loss defined on Kaggle written with Sympy
    loss = -sum([
        (wi * yi / Ni * sp.log(sp.exp(pi) / sum(map(sp.exp, p)))) / sum(w)
        for pi, yi, Ni, wi in zip(p, y, N, w)
    ])

    grad = [loss.diff(pi) for pi in p]
    hess = [loss.diff(pi).diff(pi) for pi in p]

    grad = [sp.lambdify([*p, *y, *N, *w], gi, 'numpy') for gi in grad]
    hess = [sp.lambdify([*p, *y, *N, *w], hi, 'numpy') for hi in hess]

    def obj(preds, dataset):

        # Extract the labels and the predictions
        y_true = pd.get_dummies(dataset.get_label()).values.astype(np.float64)
        y_pred = preds.reshape(y_true.shape[0], m, order='F').astype(np.float64)
        y_sums = y_true.sum(axis=0)

        gradient = np.array([gi(*y_pred.T, *y_true.T, *y_sums, *weights) for gi in grad]).T
        hessian = np.array([hi(*y_pred.T, *y_true.T, *y_sums, *weights) for hi in hess]).T

        gradient *= float(m) / (1 / y_sums).sum()
        hessian *= float(m) / (1 / y_sums).sum()

        return gradient.flatten(order='F'), hessian.flatten(order='F')

    return obj


def make_lgbm_eval(weights):

    m = len(weights)

    p = [sp.Symbol(f'p{i}') for i in range(m)]
    y = [sp.Symbol(f'y{i}') for i in range(m)]
    N = [sp.Symbol(f'N{i}') for i in range(m)]
    w = [sp.Symbol(f'w{i}') for i in range(m)]

    # This the loss defined on Kaggle written with Sympy
    loss = -sum([
        (wi * yi / Ni * sp.log(sp.exp(pi) / sum(map(sp.exp, p)))) / sum(w)
        for pi, yi, Ni, wi in zip(p, y, N, w)
    ])
    loss = sp.lambdify([*p, *y, *N, *w], loss, 'numpy')

    def eval(preds, dataset):

        # Extract the labels and the predictions
        y_true = pd.get_dummies(dataset.get_label()).values.astype(np.float64)
        y_pred = preds.reshape(y_true.shape[0], m, order='F').astype(np.float64)
        y_sums = y_true.sum(axis=0)

        return 'loss', loss(*y_pred.T, *y_true.T, *y_sums, *weights).sum(), False

    return eval


import pandas as pd


train = pd.read_feather('data/train.fth').set_index('object_id')
test = pd.read_feather('data/test.fth').set_index('object_id')

for f in ['ratio', 'width']:
    train = train.join(pd.read_hdf(f'data/iprapas/{f}_train.h5'), on='object_id')
    test = test.join(pd.read_hdf(f'data/iprapas/{f}_test.h5'), on='object_id')

X_train = train.drop(columns='target')
y_train = train['target']
X_test = test

submission = pd.DataFrame(0.0, index=X_test.index, columns=y_train.cat.categories)
submission['class_99'] = 0.0

class_weights = {c: 1 for c in y_train.cat.categories}
class_weights['class_64'] = 2
class_weights['class_15'] = 2


import numpy as np

X_train_gal = X_train[X_train['is_galactic']].drop(columns='is_galactic')
y_train_gal = y_train[X_train['is_galactic']]
X_test_gal = X_test[X_test['is_galactic']].drop(columns='is_galactic')

class_to_int = {c: i for i, c in enumerate(y_train_gal.unique())}
int_to_class = {i: c for c, i in class_to_int.items()}
weights = np.array([class_weights[int_to_class[i]] for i in sorted(int_to_class)], dtype=np.float64)

def softmax(x, axis=1):
    z = np.exp(x)
    return z / np.sum(z, axis=axis, keepdims=True)


import lightgbm as lgbm
from sklearn import model_selection

params = {
    'application': 'multiclass',
    'boosting_type': 'gbdt',
    'num_classes': y_train_gal.nunique(),
    'metric': 'None',
    'num_threads': -1,
    'num_leaves': 2 ** 3,
    'min_data_in_leaf': 130,
    'feature_fraction': 0.6,
    'feature_fraction_seed': 42,
    #'max_bin': 127,
    #'min_data_in_bin': 20,
    #'min_sum_hessian_in_leaf': 5,
    'learning_rate': 0.06,
    #'bagging_fraction': 1,
    #'bagging_freq': 0,
    #'bagging_seed': 42,
    #'lambda_l1': 0,
    #'lambda_l2': 0,
    'verbosity': -1,
}

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
feature_importances = pd.DataFrame(index=X_train_gal.columns)
gal_fit_scores = np.zeros(cv.n_splits)
gal_val_scores = np.zeros(cv.n_splits)
submission.loc[X_test_gal.index, y_train_gal.unique()] = 0.0


for i, (fit_idx, val_idx) in enumerate(cv.split(X_train_gal, y_train_gal)):

    X_fit = X_train_gal.iloc[fit_idx]
    y_fit = y_train_gal.iloc[fit_idx].map(class_to_int)
    w_fit = y_train_gal.iloc[fit_idx].map(class_weights)
    X_val = X_train_gal.iloc[val_idx]
    y_val = y_train_gal.iloc[val_idx].map(class_to_int)
    w_val = y_train_gal.iloc[val_idx].map(class_weights)

    # Train the model
    fit_set = lgbm.Dataset(X_fit.values, y_fit.values, weight=w_fit.values)
    val_set = lgbm.Dataset(X_val.values, y_val.values, reference=fit_set, weight=w_val.values)

    evals_result = {}
    model = lgbm.train(
        params=params,
        train_set=fit_set,
        fobj=make_lgbm_obj(weights),
        feval=make_lgbm_eval(weights),
        num_boost_round=30000,
        valid_sets=(fit_set, val_set),
        valid_names=('fit', 'val'),
        verbose_eval=20,
        early_stopping_rounds=20,
        evals_result=evals_result
    )

    # Store the feature importances
    feature_importances[f'gain_{i}'] = model.feature_importance('gain')
    feature_importances[f'split_{i}'] = model.feature_importance('split')

    # Store the predictions
    #y_pred = pd.DataFrame(softmax(model.predict(X_test_gal)), index=X_test_gal.index)
    #y_pred.columns = y_pred.columns.map(int_to_class)
    #submission.loc[y_pred.index, y_pred.columns] += y_pred / cv.n_splits

    # Store the scores
    gal_fit_scores[i] = evals_result['fit']['loss'][model.best_iteration - 1]
    gal_val_scores[i] = evals_result['val']['loss'][model.best_iteration - 1]

print(feature_importances.sort_values('gain_0', ascending=False))

print(f'- Train loss: {gal_fit_scores.mean():.3f} (±{gal_fit_scores.std():.3f})')
print(f'- Valid loss: {gal_val_scores.mean():.3f} (±{gal_val_scores.std():.3f})')


X_train_ex = X_train[~X_train['is_galactic']].drop(columns='is_galactic')
y_train_ex = y_train[~X_train['is_galactic']]
X_test_ex = X_test[~X_test['is_galactic']].drop(columns='is_galactic')

class_to_int = {c: i for i, c in enumerate(y_train_ex.unique())}
int_to_class = {i: c for c, i in class_to_int.items()}
weights = np.array([class_weights[int_to_class[i]] for i in sorted(int_to_class)], dtype=np.float64)

params = {
    'application': 'multiclass',
    'boosting_type': 'gbdt',
    'num_classes': y_train_ex.nunique(),
    'metric': 'None',
    'num_threads': -1,
    'num_leaves': 2 ** 3,
    'min_data_in_leaf': 180,
    'feature_fraction': 0.45,
    'feature_fraction_seed': 42,
    #'min_sum_hessian_in_leaf': 5e-2,
    #'max_bin': 127,
    #'min_data_in_bin': 40,
    'learning_rate': 0.08,
    #'bagging_fraction': 1,
    #'bagging_freq': 0,
    #'bagging_seed': 42,
    'lambda_l1': 0.5,
    'lambda_l2': 0.08,
    'verbosity': -1,
}

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
feature_importances = pd.DataFrame(index=X_train_ex.columns)
ex_fit_scores = np.zeros(cv.n_splits)
ex_val_scores = np.zeros(cv.n_splits)
submission.loc[X_test_ex.index, y_train_ex.unique()] = 0.0


for i, (fit_idx, val_idx) in enumerate(cv.split(X_train_ex, y_train_ex)):

    X_fit = X_train_ex.iloc[fit_idx]
    y_fit = y_train_ex.iloc[fit_idx].map(class_to_int)
    w_fit = y_train_ex.iloc[fit_idx].map(class_weights)
    X_val = X_train_ex.iloc[val_idx]
    y_val = y_train_ex.iloc[val_idx].map(class_to_int)
    w_val = y_train_ex.iloc[val_idx].map(class_weights)

    # Train the model
    fit_set = lgbm.Dataset(X_fit.values, y_fit.values, weight=w_fit.values)
    val_set = lgbm.Dataset(X_val.values, y_val.values, reference=fit_set, weight=w_val.values)

    evals_result = {}
    model = lgbm.train(
        params=params,
        train_set=fit_set,
        fobj=make_lgbm_obj(weights),
        feval=make_lgbm_eval(weights),
        num_boost_round=30000,
        valid_sets=(fit_set, val_set),
        valid_names=('fit', 'val'),
        verbose_eval=20,
        early_stopping_rounds=20,
        evals_result=evals_result
    )

    # Store the feature importances
    feature_importances[f'gain_{i}'] = model.feature_importance('gain')
    feature_importances[f'split_{i}'] = model.feature_importance('split')

    # Store the predictions
    #y_pred = pd.DataFrame(softmax(model.predict(X_test_ex.values)), index=X_test_ex.index)
    #y_pred.columns = y_pred.columns.map(int_to_class)
    #submission.loc[y_pred.index, y_pred.columns] += y_pred / cv.n_splits

    # Store the scores
    ex_fit_scores[i] = evals_result['fit']['loss'][model.best_iteration - 1]
    ex_val_scores[i] = evals_result['val']['loss'][model.best_iteration - 1]

print(feature_importances.sort_values('gain_0', ascending=False))

print(f'- Train loss: {gal_fit_scores.mean():.3f} (±{gal_fit_scores.std():.3f})')
print(f'- Valid loss: {gal_val_scores.mean():.3f} (±{gal_val_scores.std():.3f})')

print(f'- Train loss: {ex_fit_scores.mean():.3f} (±{ex_fit_scores.std():.3f})')
print(f'- Valid loss: {ex_val_scores.mean():.3f} (±{ex_val_scores.std():.3f})')

def GenUnknown(data):
    return ((((((data["mymedian"]) + (((data["mymean"]) / 2.0)))/2.0)) + (((((1.0) - (((data["mymax"]) * (((data["mymax"]) * (data["mymax"]))))))) / 2.0)))/2.0)

feats = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
         'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
         'class_92', 'class_95']

y = pd.DataFrame()
y['mymean'] = submission[feats].mean(axis=1)
y['mymedian'] = submission[feats].median(axis=1)
y['mymax'] = submission[feats].max(axis=1)

submission['class_99'] = GenUnknown(y)

name = f'{gal_val_scores.mean():.3f}_{gal_val_scores.std():.3f}_{ex_val_scores.mean():.3f}_{ex_val_scores.std():.3f}'

sample_sub = pd.read_csv('~/projects/kaggle-plasticc-astro-classification/data/kaggle/sample_submission.csv').set_index('object_id')

submission.loc[sample_sub.index, sample_sub.columns].to_csv(f'submissions/{name}.csv.gz', compression='gzip')

print('\a')
