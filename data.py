import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from patsy import dmatrices
from tqdm import tqdm

PKU_PATH = "./data/PKU-DFIIC_2011_2021.xlsx"
# CHFS_MASTER_PATH = "./data/2019/chfs2019_master_202112.dta"
# CHFS_IND_PATH = "./data/2019/chfs2019_ind_202112.dta"
# CHFS_HH_PATH = "./data/2019/chfs2019_hh_202112.dta"

CHFS_MASTER_PATH = "./data/2017/chfs2017_master_202206.dta"
CHFS_IND_PATH = "./data/2017/chfs2017_ind_202206.dta"
CHFS_HH_PATH = "./data/2017/chfs2017_hh_202206.dta"


YEAR = 2017


def load_DIFI():
    data_PKU = pd.read_excel(PKU_PATH, sheet_name="Provinces")
    data_PKU_group = data_PKU.groupby("year")
    data_DIFI = data_PKU_group.get_group(YEAR)
    return data_DIFI


def load_CHFS():
    data_CHFS_master = pd.read_stata(CHFS_MASTER_PATH)
    data_CHFS_ind = pd.read_stata(CHFS_IND_PATH)
    data_CHFS_hh = pd.read_stata(CHFS_HH_PATH)
    return data_CHFS_master, data_CHFS_ind, data_CHFS_hh


def get_family_keys(data_CHFS_master):
    data_master_family = data_CHFS_master.groupby("hhid")
    family_keys = list(data_master_family.groups.keys())
    return family_keys


def get_ind_keys(data_CHFS_ind, hhid):
    ind_keys = np.where(data_CHFS_ind.hhid == hhid)[0]
    return ind_keys


def find_hhead(data_CHFS_ind, hhid):
    ids = np.where(data_CHFS_ind.hhid == hhid)[0]
    hhead = None
    for id in ids:
        if data_CHFS_ind.iloc[id].hhead == 1:
            hhead = data_CHFS_ind.iloc[id]
            break
    return hhead


def get_family_data(data_CHFS_master, data_CHFS_ind, data_CHFS_hh, hhid):
    if YEAR == 2019 or YEAR == 2017:
        master_index = np.where(data_CHFS_master.hhid == hhid)[0][0]
        data_master = data_CHFS_master.iloc[master_index]
    elif YEAR == 2015:
        data_master = data_CHFS_master
    ind_indexes = get_ind_keys(data_CHFS_ind, hhid)
    data_ind = data_CHFS_ind.iloc[ind_indexes]
    hh_index = np.where(data_CHFS_hh.hhid == hhid)[0][0]
    data_hh = data_CHFS_hh.iloc[hh_index]
    hhead = find_hhead(data_CHFS_ind, hhid)
    return data_master, data_ind, data_hh, hhead


def edu_map(edu):
    if np.isnan(edu):
        return 0
    edu = int(edu)
    if edu == 1:
        return 0
    elif edu == 2:
        return 6
    elif edu == 3:
        return 9
    elif edu == 4:
        return 12
    elif edu == 5:
        return 12
    elif edu == 6:
        return 15
    elif edu == 7:
        return 16
    elif edu == 8:
        return 19
    elif edu == 9:
        return 22
    else:
        return None


def get_size(data_CHFS_hh, hhid):
    ids = np.where(data_CHFS_hh.hhid == hhid)[0]
    return data_CHFS_hh.iloc[ids[0]].a2000


def get_data(master, inds, hh, hhead, PKU_DIFI, hhid):
    # Development Index
    travel_con = master.travel_con if not np.isnan(master.travel_con) else 0
    educ_con = master.educ_con if not np.isnan(master.educ_con) else 0
    medical_con = master.medical_con if not np.isnan(master.medical_con) else 0
    other_con = master.other_con if not np.isnan(master.other_con) else 0

    DR = (travel_con + educ_con + medical_con + other_con) / master.total_consump

    # Deposit account balance
    d3103_imp = hh.d3103_imp if not np.isnan(hh.d3103_imp) else 0
    d1105_imp = hh.d1105_imp if not np.isnan(hh.d1105_imp) else 0
    d2104_imp = hh.d2104_imp if not np.isnan(hh.d2104_imp) else 0
    deposit = d3103_imp + d1105_imp + d2104_imp

    # cash balance
    if YEAR == 2019:
        k1101_imp = hh.k1101_imp if not np.isnan(hh.k1101_imp) else 0
        d7106_imp = hh.d7106ha_imp if not np.isnan(hh.d7106ha_imp) else 0
        cash = k1101_imp + d7106_imp
    elif YEAR == 2017:
        k1101_imp = hh.k1101_imp if not np.isnan(hh.k1101_imp) else 0
        cash = k1101_imp

    # Market stock value
    d3109_imp = hh.d3109_imp if not np.isnan(hh.d3109_imp) else 0
    d3116_imp = hh.d3116_imp if not np.isnan(hh.d3116_imp) else 0
    stock = d3109_imp + d3116_imp

    # Bond value
    if YEAR == 2019:
        d4103_imp = hh.d4103_imp if not np.isnan(hh.d4103_imp) else 0
        bond = d4103_imp
    elif YEAR == 2017:
        d4103_1_imp = hh.d4103_1_imp if not np.isnan(hh.d4103_1_imp) else 0
        d4103_2_imp = hh.d4103_2_imp if not np.isnan(hh.d4103_2_imp) else 0
        d4103_3_imp = hh.d4103_3_imp if not np.isnan(hh.d4103_3_imp) else 0
        d4103_4_imp = hh.d4103_4_imp if not np.isnan(hh.d4103_4_imp) else 0
        d4103_5_imp = hh.d4103_5_imp if not np.isnan(hh.d4103_5_imp) else 0
        bond = d4103_1_imp + d4103_2_imp + d4103_3_imp + d4103_4_imp + d4103_5_imp

    # fund value
    if YEAR == 2019:
        d5107_1_imp = hh.d5107_1_imp if not np.isnan(hh.d5107_1_imp) else 0
        d5107_2_imp = hh.d5107_2_imp if not np.isnan(hh.d5107_2_imp) else 0
        d5107_3_imp = hh.d5107_3_imp if not np.isnan(hh.d5107_3_imp) else 0
        d5107_4_imp = hh.d5107_4_imp if not np.isnan(hh.d5107_4_imp) else 0
        d5107_5_imp = hh.d5107_5_imp if not np.isnan(hh.d5107_5_imp) else 0
        d5107_6_imp = hh.d5107_6_imp if not np.isnan(hh.d5107_6_imp) else 0
        d5107_7_imp = hh.d5107_7_imp if not np.isnan(hh.d5107_7_imp) else 0
        d5107_7777_imp = hh.d5107_7777_imp if not np.isnan(hh.d5107_7777_imp) else 0
        fund = (
            d5107_1_imp
            + d5107_2_imp
            + d5107_3_imp
            + d5107_4_imp
            + d5107_5_imp
            + d5107_6_imp
            + d5107_7_imp
            + d5107_7777_imp
        )
    elif YEAR == 2017:
        d5107_imp = hh.d5107_imp if not np.isnan(hh.d5107_imp) else 0
        fund = d5107_imp

    # Value of financial derivatives
    if YEAR == 2019:
        d6100a = hh.d6100a if not np.isnan(hh.d6100a) else 0
        derivatives = d6100a
    elif YEAR == 2017:
        d6100a_imp = hh.d6100a_imp if not np.isnan(hh.d6100a_imp) else 0
        derivatives = d6100a_imp

    # Value of financial and wealth management products
    d7110a_imp = hh.d7110a_imp if not np.isnan(hh.d7110a_imp) else 0
    bfpp = d7110a_imp

    # Value of Internet financial products
    if YEAR == 2019:
        d7106hb_imp = hh.d7106hb_imp if not np.isnan(hh.d7106hb_imp) else 0
        ifpp = d7106hb_imp
    elif YEAR == 2017:
        d7106h_imp = hh.d7106h_imp if not np.isnan(hh.d7106h_imp) else 0
        ifpp = d7106h_imp

    # Value of non RMB assets
    d8104_imp = hh.d8104_imp if not np.isnan(hh.d8104_imp) else 0
    rnmb = d8104_imp

    # Gold value
    d9103_imp = hh.d9103_imp if not np.isnan(hh.d9103_imp) else 0
    gold = d9103_imp

    # Value of other financial assets
    d9110a_imp = hh.d9110a_imp if not np.isnan(hh.d9110a_imp) else 0
    efasset = d9110a_imp

    # Lending funds
    if YEAR == 2019:
        k2201a_imp = hh.k2201a_imp if not np.isnan(hh.k2201a_imp) else 0
        k2102c_imp = hh.k2102c_imp if not np.isnan(hh.k2102c_imp) else 0
        lending = k2201a_imp + k2102c_imp
    elif YEAR == 2017:
        k2102c_imp = hh.k2102c_imp if not np.isnan(hh.k2102c_imp) else 0
        lending = k2102c_imp

    # Social security account balance
    hhins = (
        master.fina_asset
        - deposit
        - cash
        - stock
        - bond
        - fund
        - derivatives
        - bfpp
        - ifpp
        - rnmb
        - gold
        - efasset
        - lending
    )

    # DEP
    DEP = deposit + cash + bond + fund

    # INS
    INS = hhins

    # STO
    STO = stock + derivatives

    # Housing liabilities
    house_debt = master.house_debt if not np.isnan(master.house_debt) else 0

    # Vehicle liabilities
    vehicle_debt = master.vehicle_debt if not np.isnan(master.vehicle_debt) else 0

    # Education debt
    educ_debt = master.educ_debt if not np.isnan(master.educ_debt) else 0

    # Other Liabilities
    other_debt = master.other_debt if not np.isnan(master.other_debt) else 0

    # Credit card liabilities
    credit_debt = master.credit_debt if not np.isnan(master.credit_debt) else 0

    # DEBT
    DEBT = house_debt + vehicle_debt + educ_debt + other_debt + credit_debt

    # Total household income INC
    INC = master.total_income

    # Head of household education level EDU
    EDU = edu_map(hhead.a2012)

    # Family size SIZE
    if YEAR == 2019:
        SIZE = hh.a2000 if not np.isnan(hh.a2000) else len(inds)
    elif YEAR == 2017:
        a2000a = hh.a2000a if not np.isnan(hh.a2000a) else len(inds)
        a2000b = hh.a2000b if not np.isnan(hh.a2000b) else 0
        SIZE = a2000a + a2000b + 1

    # Age of head of household AGE
    AGE = YEAR - hhead.a2005

    # Is it a rural household registration
    RUR = master.rural

    # DIFI
    DIFI = PKU_DIFI.iloc[
        np.where(PKU_DIFI.prov_name == master.prov)[0]
    ].index_aggregate.iloc[0]

    return {
        "hhid": hhid,
        "DR": DR,
        "DEP": DEP,
        "INS": INS,
        "STO": STO,
        "DEBT": DEBT,
        "INC": INC,
        "EDU": EDU,
        "SIZE": SIZE,
        "AGE": AGE,
        "RUR": RUR,
        "DIFI": DIFI,
    }


def main():
    # Read data
    data_DIFI = load_DIFI()
    data_CHFS_master, data_CHFS_ind, data_CHFS_hh = load_CHFS()
    family_keys = get_family_keys(data_CHFS_hh)
    data_len = len(family_keys)
    heads = [
        "hhid",
        "DR",
        "DEP",
        "INS",
        "STO",
        "DEBT",
        "INC",
        "EDU",
        "SIZE",
        "AGE",
        "RUR",
        "DIFI",
    ]
    data = pd.DataFrame(columns=heads)

    for hhid in tqdm(family_keys):
        data_master, data_ind, data_hh, hhead = get_family_data(
            data_CHFS_master, data_CHFS_ind, data_CHFS_hh, hhid
        )
        loc_data = get_data(data_master, data_ind, data_hh, hhead, data_DIFI, hhid)
        tmp = pd.DataFrame(loc_data, index=[0])
        data = pd.concat([data, tmp], ignore_index=True)

    data.to_excel(f"./data/data_{YEAR}.xlsx", index=False)


if __name__ == "__main__":
    main()
