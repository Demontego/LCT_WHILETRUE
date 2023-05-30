from sklearn.preprocessing import LabelEncoder
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from tqdm.notebook import tqdm
import pickle
import os
import pandas as pd
import numpy as np
import ast
from tqdm.notebook import tqdm
import json
from multiprocessing import Pool

def extract_cat(js, cat):
        vals = []
        if js:
            if cat in js:
                for val in js[cat]:
                    try:
                        val = float(val.replace(',', '.').replace('"','').split()[0])
                    except:
                        pass
                    if val in all_categories[cat]:
                        vals.append(val)
                if vals:
                    return vals[0]
        return None

def create_cats(entry):
        if entry:
            global all_categories
            for cat in entry:
                if cat not in all_categories:
                    all_categories[cat] = set()
                for val in entry[cat]:
                    try:
                        val = float(val.replace(',', '.').replace('"','').split()[0])
                    except:

                        pass
                    all_categories[cat].add(val)
            return True

def main():
    train = pd.read_csv('aggregated_attributes_zs.csv')

    graph_whole = train
    graph_whole.shape
    for gg in tqdm(graph_whole.groupby('cat3')):
        cat = gg[0]
        gg_exp = pd.melt(gg[1], id_vars=['variantid'], value_vars=gg[1].columns[1:], var_name='attribute', value_name='value')
        gg_exp = gg_exp[gg_exp['value'] != 'undefined_value_0'] 
        gg_exp = gg_exp[gg_exp['attribute'] != 'cat3']
        gg_exp['attribute_hash'] = gg_exp['attribute'] + '_' + gg_exp['value'].astype(str)
        gg_exp.to_csv(f'./graphdfs/{cat}_graph.csv', index=False)
        
        
    def process_file(file):
        graph_whole = pd.read_csv('./graphdfs/' + file)
        graph_whole = graph_whole[['variantid','attribute_hash']].applymap(str)
        encoder = LabelEncoder()
        combined_values = graph_whole['variantid'].tolist() + graph_whole['attribute_hash'].tolist()
        encoder.fit(combined_values)
        graph_whole['node1_id'] = encoder.transform(graph_whole['variantid'])
        graph_whole['node2_id'] = encoder.transform(graph_whole['attribute_hash'])
        graph_whole = graph_whole[['node1_id','node2_id']]
        graph_whole.to_csv(f'./tmp/{file.split("_")[0].replace("-", "_").replace(" ", "_").replace(",","")}_labeled.csv', header=False, index = False, sep = '\t')
        os.system(f'pecanpy --input ./tmp/{file.split("_")[0].replace("-", "_").replace(" ", "_").replace(",","")}_labeled.csv --output ./tmp/{file.split("_")[0].replace("-", "_").replace(" ", "_").replace(",","")}_tmp.emb --mode SparseOTF')
        with open(f'./tmp/{file.split("_")[0].replace("-", "_").replace(" ", "_").replace(",","")}_tmp.emb', 'r') as fin:
            data = fin.read().splitlines(True)
        with open(f'./embs/{file.split("_")[0].replace("-", "_").replace(" ", "_").replace(",","")}.emb', 'w') as fout:
            fout.writelines(data[1:])
        with open(f'./encs/{file.split("_")[0].replace("-", "_").replace(" ", "_").replace(",","")}_enc.pkl', 'wb') as file:
            pickle.dump(encoder, file)

    p = Pool(len(os.listdir('./graphdfs/')))
    _ = p.map(process_file, os.listdir('./graphdfs/'))
            
    alldata = []
    for file in tqdm(os.listdir('./embs/')):
        data = pd.read_csv('./embs/' + file, sep=' ', header=None)
        data.columns = ['item'] + list(range(128))
        with open('./encs/' + file.split('.emb')[0] + '_enc.pkl', 'rb') as fin:
            enc = pickle.load(fin)
        data['variantid'] = enc.inverse_transform(data.item)
        ids = []
        data['variantid'].map(lambda x: ids.append(x) if all([y in '1234567890' for y in x]) else None)
        data = data[data.variantid.isin(ids)]
        data = data.drop('item', axis=1)
        data.variantid = data.variantid.map(int)
        alldata.append(data)


    tqdm.pandas()

    train = pd.read_parquet('train_data.parquet')
    test = pd.read_parquet('test_data.parquet')
    # train = pd.concat([train, test])

    train["cat3"] = train["categories"].apply(lambda x: json.loads(x)["3"]) 
    test["cat3"] = test["categories"].apply(lambda x: json.loads(x)["3"]) 
    cat3_counts = dict(train['cat3'].value_counts())
    train["cat3"] = train["cat3"].apply(lambda x: x if cat3_counts[x] > 1000 else "rest")
    test["cat3"] = test["cat3"].apply(lambda x: x if cat3_counts[x] > 1000 else "rest")
    train = pd.concat([train, test])

    train.characteristic_attributes_mapping = train.characteristic_attributes_mapping.progress_apply(lambda x: ast.literal_eval(x) if x else None)

    all_categories = {}
    _ = train.characteristic_attributes_mapping.progress_apply(create_cats)

    for x in all_categories:
        float_obj = len([y for y in all_categories[x] if type(y) == float])
        if float_obj / len(all_categories[x]) > 0.9:
            all_categories[x] = [y for y in all_categories[x] if type(y) == float]

    _ = [all_categories.pop(z) for z in set([x for x in all_categories if x not in ['Название модели'] and len([y for y in all_categories[x] if type(y) == str and len(y) > 100]) / len(all_categories[x]) > 0.1] \
        + ['Артикул', 'Диапазон рабочей температуры', 'Вариант', 'Гарантийный срок', 'Комплектация', 'Партномер']) if z in all_categories]

    float_columns = [x for x in all_categories if all([type(y) == float for y in all_categories[x]])]

    catstoknow = ['Цвет товара',
    'Бренд',
    'Страна-изготовитель',
    'Тип'] + float_columns + [p for p in all_categories if 'размер' in p.lower()]

    train = train[['variantid', 'cat3', 'characteristic_attributes_mapping']]

    for cat in tqdm(catstoknow):
        train[cat] = train.characteristic_attributes_mapping.map(lambda p: extract_cat(p, cat))

    def extract_existence(js, cat):
        if js:
            if cat in js:
                return True
            else:
                return 'undefined_value_0'
        return None
    for cat in tqdm(set.difference(set(all_categories), set(catstoknow))):
        train[cat] = train.characteristic_attributes_mapping.map(lambda p: extract_existence(p, cat))

    train = train.dropna(subset=['characteristic_attributes_mapping'])
    train = train.drop('characteristic_attributes_mapping', axis=1)
    train = train.drop_duplicates(subset=['variantid'])

    train.to_csv('aggregated_attributes_zs.csv', index=False)

if __name__ == "__main__":
    main()