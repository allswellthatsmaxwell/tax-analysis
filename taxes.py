import pandas as pd, numpy as np
import re
from typing import List

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

    def find_doubled_row_indexes(self, min_rows_to_find: int):
        doublings = []
        try:
            return pd.read_csv('data/doublings_dat.csv')
        except FileNotFoundError:
            pass

        try:
            seen = set()

            for i, rowi in self.dat[ITEMS].iterrows():
                idstr = counts_to_str(rowi)
                if idstr in seen:
                    continue
                seen.add(idstr)

                doubled = rowi.mul(2)
                for j, rowj in self.dat[ITEMS].iterrows():
                    if (doubled == rowj).all():
                        doublings.append((i, j))
                        break  # no need for multiple of the same
                if i != 0 and i % 10 == 0:
                    print(f"Row {i}; {len(doublings)} doubles found so far.")
                if len(doublings) >= min_rows_to_find:
                    break
        finally:
            dat = pd.DataFrame(doublings, columns=['i', 'j']).drop_duplicates()
            return dat

    def get_differences_from_doublings(self, doublings_dat):
        tax_differences = []
        for _, row in doublings_dat.iterrows():
            i, j = row
            tax_differences.append([i, j, self.dat.iloc[i]['total_tax'],
                                    self.dat.iloc[j]['total_tax']])
        return pd.DataFrame(tax_differences,
                            columns=['i', 'j', 'base', 'doubled']).drop_duplicates()


def plot_doublings(dat):
    return (
            pn.ggplot(dat, pn.aes(x='base', y='doubled')) +
            pn.geom_point() +
            pn.geom_abline(slope=1, intercept=0, color='maroon') +
            pn.geom_abline(slope=2, intercept=0, color='navy') +
            pn.theme_bw())


def counts_to_str(counts):
    return ','.join([str(x) for x in counts])


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
            num_boost_round=500 * 2,
            callbacks=[  # lgb.log_evaluation(period=50),
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


def apply_do_operator_over_ranges(data: Data, modeler: Modeler):
    avg_preds = []
    for item in ITEMS:
        maximum = int(data.get_ranges().query(f"item == '{item}'")['max'].iloc[0])
        df = data.dat.copy()[ITEMS]
        for val in range(maximum):
            df[item] = val
            preds = modeler.model.predict(df)
            avg_preds.append([item, val, np.mean(preds)])
    return pd.DataFrame(avg_preds, columns=['item', 'x', 'pred'])


def get_do_results_plot(do_dat):
    return (
            pn.ggplot(do_dat, pn.aes(x='x', y='pred')) +
            pn.geom_point() +
            pn.facet_wrap('item', scales='free') +
            pn.theme_bw() +
            pn.theme(figure_size=(9, 5)))


ITEMS_TO_ASSIGN = {
    'Cockatrice Eye': 4,
    'Dragon Head': 4,
    'Lich Skull': 5,
    'Unicorn Horn': 7,
    'Zombie Hand': 8
}


class AssignmentAssessor:
    def __init__(self, data: Data, modeler: Modeler):
        self.data = data
        self.modeler = modeler

    def assess(self, records):
        assert (np.sum(np.array(records), axis=0) == np.array(list(ITEMS_TO_ASSIGN.values()))).all()
        dat = (
            pd.DataFrame(records, columns=ITEMS)
            .assign(rowid=lambda d: d.index)
            .merge(self.data.dat, on=ITEMS, how='left')
            .drop_duplicates())
        single_dat = pd.DataFrame(records)
        single_dat.columns = ITEMS
        taxes = self.modeler.model.predict(single_dat)
        return dat.assign(predicted=np.round(taxes, 1)).assign(
            summed_actual=lambda d: d['total_tax'].sum(),
            summed_predicted=lambda d: d['predicted'].sum())





# class TaxAssignmentsGenerator:
#     def __init__(self):
#         self.items = ITEMS_TO_ASSIGN.copy()
#         self.assignments = []
#
#     def generate(self):
#         self._generate(self.items, [])
#         return self.assignments
#
#     def _generate(self, items, assignment: List[int]):
#         if sum(items.values()) == 0:
#             self.assignments.append(assignment)
#         else:
#             for adventurer in range(1, 2):
#                 for item, remaining in items.items():
#                     for n_assigned in range(1, remaining + 1):
#                         remaining_items = items.copy()
#                         remaining_items[item] -= n_assigned
#                         self._generate(remaining_items, assignment + [adventurer * 10 + n_assigned])


def get_rmse(preds_dat):
    return np.sqrt(np.mean((preds_dat['preds'] - preds_dat['actual']) ** 2))

