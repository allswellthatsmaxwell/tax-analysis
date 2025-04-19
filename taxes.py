import pandas as pd, numpy as np
import re

import plotnine as pn
import lightgbm as lgb

PATH = "data/dndsci_tax.csv"

ITEMS = ['Cockatrice Eye', 'Dragon Head', 'Lich Skull', 'Unicorn Horn', 'Zombie Hand']


def get_total_assessed(assessed_str):
    # assessed_str looks like "18 gp 9 sp"
    # return 18.9
    if not assessed_str:
        return 0
    match = re.match(r"(\d+) gp (\d+) sp", assessed_str)
    if match:
        gp = int(match.group(1))
        sp = int(match.group(2))
        return gp + sp / 10
    else:
        return 0


class Splits:
    def __init__(self, dat):
        self.dat = dat.copy()
        prop = 0.7
        trn_inds = [np.random.uniform() < prop for _ in range(len(dat))]
        self.trn = dat[trn_inds]
        valtst = dat[~np.array(trn_inds)]
        self.val = valtst.sample(frac=0.5)
        self.tst = valtst[~valtst.index.isin(self.val.index)]

        self.trn = self.trn.reset_index(drop=True)
        self.val = self.val.reset_index(drop=True)
        self.tst = self.tst.reset_index(drop=True)

        self.trn_data = lgb.Dataset(self.trn[ITEMS], self.trn['total_tax'])
        self.val_data = lgb.Dataset(self.val[ITEMS], self.val['total_tax'])
        self.tst_data = lgb.Dataset(self.tst[ITEMS], self.tst['total_tax'])


class Data:
    def __init__(self):
        self.dat = pd.read_csv(PATH)
        self.dat['total_tax'] = self.dat['Tax Assessed'].apply(get_total_assessed)

    def get_ranges(self):
        ranges = []
        for item in ITEMS:
            series = self.dat[item]
            ranges.append([item, series.min(), series.max()])
        return pd.DataFrame(ranges, columns=['item', 'min', 'max'])

    def get_splits(self):
        return Splits(self.dat)

    def get_modeler(self):
        return Modeler(self.get_splits())


class Modeler:
    def __init__(self, splits: Splits):
        self.splits = splits
        self.model = None

    def train(self, nleaves: int):
        # 60 gets lowest rmse
        param = {'num_leaves': nleaves, 'objective': 'regression', 'metric': 'rmse'}
        self.model = lgb.train(
            param,
            self.splits.trn_data,
            valid_sets=[self.splits.val_data],
            num_boost_round=500*2,
            callbacks=[#lgb.log_evaluation(period=50),
                       lgb.early_stopping(stopping_rounds=10)])

    def get_tstpreds_dat(self):
        preds = self.model.predict(self.splits.tst[ITEMS])
        return pd.DataFrame(preds, columns=['preds']).assign(
            actual=self.splits.tst['total_tax'].values)

    def get_tstpreds_plot(self):
        tstpreds_dat = self.get_tstpreds_dat()
        rmse = get_rmse(tstpreds_dat)
        return (
                pn.ggplot(tstpreds_dat, pn.aes(x='actual', y='preds')) +
                pn.geom_point(alpha=0.1) +
                pn.geom_abline(slope=1, intercept=0, color='red') +
                pn.annotate(
                    'text', x=20, y=143,
                    label=f"RMSE: {rmse:.2f}",
                    size=10, color='navy') +
                # pn.xlim((0, None)) +
                # pn.ylim((0, None)) +
                pn.theme_bw() +
                pn.theme(figure_size=(4, 3))
        )

ITEMS_TO_ASSIGN = {
    'Cockatrice Eye': 4,
    'Dragon Head': 4,
    'Lich Skull': 5,
    'Unicorn Horn': 7,
    'Zombie Hand': 8
}


class TaxAssignmentsGenerator:
    def generate(self):
        pass


def get_rmse(preds_dat):
    return np.sqrt(np.mean((preds_dat['preds'] - preds_dat['actual']) ** 2))
