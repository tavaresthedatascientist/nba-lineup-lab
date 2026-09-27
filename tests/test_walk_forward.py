from lineup_lab.model import walk_forward


def test_rolling_folds_are_5_then_10(bundle):
    res = walk_forward(bundle.table, expanding=False)
    for f in res["folds"]:
        lo, hi = f["train_games"]
        tlo, thi = f["test_games"]
        assert hi - lo == 5
        assert tlo == hi and thi - tlo <= 10
    # test blocks tile games 6-82 without overlap
    starts = [f["test_games"][0] for f in res["folds"]]
    assert starts == list(range(5, 82, 10))
    assert res["pooled"]["random_forest"]["games"] > 1000


def test_predictions_are_out_of_sample(bundle):
    res = walk_forward(bundle.table, expanding=True)
    preds = res["predictions"]
    assert preds["GAME_ID"].is_unique
    assert preds["P_REF"].between(0, 1).all()
