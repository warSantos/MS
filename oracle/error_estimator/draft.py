"""
def global_error_estimation(dataset, probas, name_estimator, oracle_dir, CLFS, confidence=65, load_model=True):

    # For each fold.
    for fold in np.arange(10):
        
        # Building dirs.
        output_dir = f"{oracle_dir}/global_{name_estimator}/error_estimator/{dataset}/{fold}"
        os.makedirs(output_dir, exist_ok=True)
        model_path = f"{output_dir}/model"

        begin = time()
        global_X_train = []
        global_X_test = []
        global_upper_train = []
        global_upper_test = []
        global_y_train = []
        global_y_test = []
        # Loading labels.
        y_train = np.load(
            f"/home/welton/data/datasets/labels/split_10/{dataset}/{fold}/train.npy")
        y_test = np.load(
            f"/home/welton/data/datasets/labels/split_10/{dataset}/{fold}/test.npy")
        # Load fold Meta-Features (Washington).
        dist_train = csr_matrix(np.load(
            f"/home/welton/data/meta_features/features/dist/{fold}/{dataset}/train.npz")["X_train"]).toarray()
        dist_test = csr_matrix(np.load(
            f"/home/welton/data/meta_features/features/dist/{fold}/{dataset}/test.npz")["X_test"]).toarray()
        # Joining all CLFs meta-features.
        for target_clf in CLFS:

            # Building Meta-Features.
            X_train, X_test, upper_train, upper_test = make_mfs(
                probas, target_clf, y_train, y_test, fold, dist_train, dist_test)

            global_X_train.append(X_train)
            global_X_test.append(X_test)

            global_upper_train.append(upper_train)
            global_upper_test.append(upper_test)

            global_y_train.append(y_train)
            global_y_test.append(y_test)

        global_X_train = np.vstack(global_X_train)
        X_test = np.vstack(global_X_test)
        global_upper_train = np.hstack(global_upper_train)
        upper_test = np.hstack(global_upper_test)
        
        # Featuring selection.
        print("Feature Selection...")
        forest_path = f"{output_dir}/forest"
        best_feats, forest = feature_selection(global_X_train, X_test, global_upper_train, upper_test)
        ranking = (1 - forest.feature_importances_).argsort()
        best_feats_set = ranking[:best_feats]
        dump(forest, forest_path)
        
        # Saving optimal number of features.
        with open(f"{output_dir}/fs.json", 'w') as fd:
            fs = {"best_feats": best_feats}
            json.dump(fs, fd)


        # Hyperparameter tuning on GLOBAL Meta-Features (concatenating all CLFs MFs).
        optuna_search = execute_optimization(
            name_estimator,
            model_path,
            global_X_train,
            global_upper_train,
            load_model=True)

        # Gloabl Prediction
        y_pred = optuna_search.predict(X_test)
        
        if load_model and os.path.exists(model_path):
            error_estimator = load(model_path)
        else:

            error_estimator = XGBClassifier(
                n_estimators=300,
                learning_rate=0.11,
                max_depth=11,
                booster="gbtree",
                colsample_bytree=0.650026359170959,
                random_state=42,
                verbosity=0,
                n_jobs=1,
                tree_method='gpu_hist')
    
            error_estimator = GradientBoostingClassifier()
            error_estimator.fit(global_X_train[:, best_feats_set], global_upper_train)
            dump(error_estimator, model_path)
        
        y_pred = error_estimator.predict(X_test[:, best_feats_set])
        pc, rc, f1, acc = get_scores(upper_test, y_pred)
        dict_scores = get_dict_score(pc, rc, f1, acc)
        save_scores(output_dir, dict_scores)
        print(
            f"\nDATASET: {dataset.upper()} / GLOBAL ESTIMATOR / FOLD - {fold} - Prec: {pc:.2f}, Rec: {rc:.2f}, F1: {f1:.2f}, Acc: {acc:.2f}")

        # Local Prediction.
        for X_test, upper_test, alg in zip(global_X_test, global_upper_test, CLFS):

            output_dir = f"{oracle_dir}/global_{name_estimator}/clfs/{dataset}/{alg}/{fold}"
            os.makedirs(output_dir, exist_ok=True)

            y_pred = error_estimator.predict(np.vstack(X_test[:, best_feats_set]))
            pc, rc, f1, acc = get_scores(upper_test, y_pred)
            print(
                f"DATASET: {dataset.upper()} / CLF: {alg} / FOLD - {fold} - Prec: {pc:.2f}, Rec: {rc:.2f}, F1: {f1:.2f}, Acc: {acc:.2f}")

            dict_scores = get_dict_score(pc, rc, f1, acc)
            save_scores(output_dir, dict_scores)

            if f1 < confidence:
                y_pred = np.zeros(y_pred.shape[0]) + 1

            np.savez(f"{output_dir}/test", y=y_pred)
            y_true = np.zeros(y_train.shape[0]) + 1
            np.savez(f"{output_dir}/train", y=y_true)
        end = time()
        print(f"Seconds: {end - begin}s")
"""
