def hyperopt_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, params: Dict, config: Config, max_evals=10):
    X_train, X_test, y_train, y_test = data_split_by_time(X_train, y_train, test_size=0.2)
    X_train, X_val, y_train, y_val = data_split_by_time(X_train, y_train, test_size=0.3)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
        #"max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "max_depth": hp.choice("max_depth", [1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }

    def objective(hyperparams):
        if config.time_left() < 50:
            return {'status': STATUS_FAIL}
        else:
            model = lgb.train({**params, **hyperparams}, train_data, 100,
                          valid_data, early_stopping_rounds=10, verbose_eval=0)
            pred = model.predict(X_test)
            score = roc_auc_score(y_test, pred)

            #score = model.best_score["valid_0"][params["metric"]]

            # in classification, less is better
            return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=max_evals, verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    return hyperparams 