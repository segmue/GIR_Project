import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

EDINBURGH_FILES_EUROPE = 'edgeoparser_output_pre_cleaned/europe.pkl'
EDINBURGH_FILES_AFRICA = 'edgeoparser_output_pre_cleaned/africa.pkl'

IRCHEL_FILES_EUROPE = 'irchel_geoparser_output/Europe-Central-Asia_2018-2024_Nov22_final.pkl' # Add Batch_no and .pkl later
IRCHEL_FILES_AFRICA = '' # Add Batch_no and .pkl later

ORIGINAL_EUROPE = 'processed_data/Europe-Central-Asia_2018-2024_Nov22.csv'
ORIGINAL_AFRICA = 'processed_data/Africa_1997-2024_Nov29.csv'

CONTINENTS = {'europe': {'edinburgh': EDINBURGH_FILES_EUROPE, 'irchel': IRCHEL_FILES_EUROPE, 'original': ORIGINAL_EUROPE},
              'africa': {'edinburgh': EDINBURGH_FILES_AFRICA, 'irchel': IRCHEL_FILES_AFRICA, 'original': ORIGINAL_AFRICA}}


def load_data(continent):
    """ Loads original data, edinburgh data and irchel data for the given continent. """
    edinburgh = pd.read_pickle(CONTINENTS[continent]['edinburgh'])
    edinburgh = pd.DataFrame(edinburgh)
    irchel = pd.read_pickle(CONTINENTS[continent]['irchel'])
    original = pd.read_csv(CONTINENTS[continent]['original'])
    return edinburgh, irchel, original

def merge_ir_to_orig(ir,orig):
    """ Left joins IR to orig based on the column "event_id_cnty" which is in both dataframes,
    only the additional column geoparser_res should be added to the original data. """
    ir = ir[["event_id_cnty", "geoparser_res"]].rename(columns={"geoparser_res": "irchel_geoparser_res"})
    merged = pd.merge(orig, ir, how='left', on='event_id_cnty')
    return merged

def filter_ed(ed):
    """ Removes all entries, which locations is empty, prints number of removed rows (also in percent),
    also remove row, where locations.latitude or locations.longitude is empty. """
    n = len(ed)

    ed = ed[ed.locations.apply(lambda x: len(x) > 0)]
    ed = ed[ed.locations.apply(lambda x: any([loc["latitude"] != "" and loc["longitude"] != "" for loc in x]))]
    ed["locations"] = ed.locations.apply(lambda x: {loc["name"]: {"latitude": loc["latitude"], "longitude": loc["longitude"]} for loc in x})

    print(f"Removed {n-len(ed)} rows from edinburgh geoparser Results ({(n-len(ed))/n*100:.2f}%)")
    return ed

def filter_id(ir):
    """ Filters based on the geoparser_res column, removes all entries where geoparser_res is empty, or empty dict {} """
    n = len(ir)
    ir = ir[ir.geoparser_res.apply(lambda x: x != {})]
    ir = ir[ir.geoparser_res.apply(lambda x: x != [])]
    print(f"Removed {n-len(ir)} rows from irchel geoparser Results ({(n-len(ir))/n*100:.2f}%)")
    return ir

def add_ed_to_org(ed, orig):
    """ Adds the Edinburgh data to the original data. merge must happen based on the raw_text in ed and "notes" column in org.
     add the ed locations as a column to the original data, but reformat the locations tobe in the following form:
     {name: {"latitude": latitude, "longitude": longitude}, name: ...}"""
    ed = ed[["raw_text", "locations"]]
    ed = ed.rename(columns={"raw_text": "notes", "locations": "edinburgh_geoparser_res"})
    merged = pd.merge(orig, ed, how='left', on='notes')
    return merged

def main(continent):
    if continent not in CONTINENTS:
        raise ValueError(f"Invalid continent: {continent}, must be one of {list(CONTINENTS.keys())}")
    ed, ir, orig = load_data(continent)
    ed = filter_ed(ed)
    ir = filter_id(ir)
    res = merge_ir_to_orig(ir, orig)
    res = add_ed_to_org(ed, res)
    res = res[['event_id_cnty', 'notes',"country","admin1","admin2","admin3", 'location','longitude','latitude', 'edinburgh_geoparser_res', 'irchel_geoparser_res']]

    dir = "result"
    if not os.path.exists(dir):
        os.makedirs(dir)
    res.to_pickle(f"result/{continent}.pkl")

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else 'europe'
    main(arg)