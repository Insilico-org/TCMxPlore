# batman_db/batman.py
import pandas as pd
import pickle
import numpy as np
from scipy.stats import fisher_exact
from typing import Optional, Any, Iterable
from copy import copy as cp
import json

from abc import ABC, abstractmethod

class TCMEntity(ABC):
    empty_override = True
    desc = ''
    cid = -1
    entity = 'superclass'

    def __init__(self,
                 pref_name: str,
                 synonyms: Optional[list[str]] = None,
                 desc: str = '',
                 **kwargs):
        self.pref_name = pref_name
        self.desc = desc
        self.synonyms = [] if synonyms is None else [x for x in synonyms if str(x).strip() != 'NA']

        self.targets = {"known": dict(), "predicted": dict()}

        self.formulas = []
        self.herbs = []
        self.ingrs = []

        for k, v in kwargs.items():
            self.__dict__[k] = v

    def serialize(self, links = True):
        init_dict = dict(
            cid=self.cid,
            targets_known=self.targets['known'],
            targets_pred=self.targets['predicted'],
            pref_name=self.pref_name, desc=self.desc,
            synonyms=cp(self.synonyms),
            entity=self.entity
        )
        link_dict = dict()
        if links:
            link_dict = self._get_link_dict()
        out_dict = {"init": init_dict, "links": link_dict}
        return out_dict

    @classmethod
    def load(cls,
             db: 'TCMDB', ser_dict: dict,
             skip_links = True):
        init_args = ser_dict['init']

        if skip_links:
            init_args.update({"empty_override":True})
        else:
            init_args.update({"empty_override": False})

        new_entity = cls(**init_args)
        if not skip_links:
            links = ser_dict['links']
            new_entity._set_links(db, links)
        return (new_entity)

    def _get_link_dict(self)->dict:
        return dict(
            ingrs=[x.cid for x in self.ingrs],
            herbs=[x.pref_name for x in self.herbs],
            formulas=[x.pref_name for x in self.formulas]
        )

    def _set_links(self, db: 'TCMDB', links: dict):
        for ent_type in links:
            self.__dict__[ent_type] = [db.__dict__[ent_type].get(x) for x in links[ent_type]]
            self.__dict__[ent_type] = [x for x in self.__dict__[ent_type] if x is not None]

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.pref_name}>"

class Ingredient(TCMEntity):
    entity: str = 'ingredient'

    def __init__(self, cid: int,
                 targets_pred: Optional[dict] = None,
                 targets_known: Optional[dict] = None,
                 synonyms: Optional[list[str]] = None,
                 pref_name: str = '', desc: str = '',
                 empty_override: bool = True, **kwargs):

        if not empty_override:
            assert targets_known is not None or targets_pred is not None, \
                f"Cant submit a compound with no targets at all (CID:{cid})"

        super().__init__(pref_name=pref_name, synonyms=synonyms, desc=desc, **kwargs)

        self.cid = cid
        self.targets = {
            'known': targets_known if targets_known is not None else {"symbols": [], 'entrez_ids': []},
            'predicted': targets_pred if targets_pred is not None else {"symbols": [], 'entrez_ids': []}
        }

class Herb(TCMEntity):
    entity: str = 'herb'

    def __init__(self, pref_name: str,
                 ingrs: Optional[list[Ingredient]] = None,
                 synonyms: Optional[list[str]] = None,
                 desc: str = '',
                 empty_override: bool = True, **kwargs):

        if ingrs is None:
            ingrs = []

        if not ingrs and not empty_override:
            raise ValueError(f"No ingredients provided for {pref_name}")

        super().__init__(pref_name=pref_name, synonyms=synonyms, desc=desc, **kwargs)

        self.ingrs = ingrs

    def is_same(self, other: 'Herb') -> bool:
        if len(self.ingrs) != len(other.ingrs):
            return False
        this_ingrs = set(x.cid for x in self.ingrs)
        other_ingrs = set(x.cid for x in other.ingrs)
        return this_ingrs == other_ingrs

    def get_target_counts(self,
                          tg_type: str='both',
                          gene_names: str = "symbols") -> dict[str|int, int]:
        """
        Count occurrences of targets across ingredients

        Args:
            tg_type: Type of targets to count ('both', 'known', or 'predicted')
            gene_names: Key for accessing gene identifiers ('symbols' or 'entrez_ids')

        Returns:
            Dictionary mapping targets to their occurrence counts
        """
        tg_counts = dict()
        for i in self.ingrs:
            targets = (
                        set(i.targets["known"][gene_names]) | set(i.targets["predicted"][gene_names]) if
                        tg_type == "both" else
                        set(i.targets[tg_type][gene_names])
                      )
            for tg in targets:
                tg_counts[tg] = tg_counts.get(tg, 0) + 1
        return(tg_counts)

class Formula(TCMEntity):
    entity: str = 'formula'

    def __init__(self, pref_name: str,
                 herbs: Optional[list[Herb]] = None,
                 synonyms: Optional[list[str]] = None,
                 desc: str = '',
                 empty_override: bool = False, **kwargs):

        if herbs is None:
            herbs = []

        if not herbs and not empty_override:
            raise ValueError(f"No herbs provided for {pref_name}")

        super().__init__(pref_name=pref_name, synonyms=synonyms, desc=desc, **kwargs)
        self.herbs = herbs

    def is_same(self, other: 'Formula') -> bool:
        if len(self.herbs) != len(other.herbs):
            return False
        this_herbs = set(x.pref_name for x in self.herbs)
        other_herbs = set(x.pref_name for x in other.herbs)
        return this_herbs == other_herbs

    def get_herbs(self)->list[str]:
        return(list(set([x.pref_name for x in self.herbs])))

    def set_herbs(self, db, herbs: list[str]):
        self.herbs = [db.herbs[x] for x in herbs]

    def get_target_counts(self,
                          tg_type: str='both',
                          gene_names: str = "symbols") -> dict[str|int, int]:
        """
        Count occurrences of targets across a formula's herbs by combining each herb's target counts

        Args:
            tg_type: Type of targets to count ('both', 'known', or 'predicted')
            gene_names: Key for accessing gene identifiers ('symbols' or 'entrez_ids')

        Returns:
            Dictionary mapping targets to their occurrence counts
        """
        tg_counts = dict()
        for h in self.herbs:
            herb_counts = h.get_target_counts(tg_type, gene_names)
            for tg, n in herb_counts.items():
                tg_counts[tg] = tg_counts.get(tg, 0) + n
        return(tg_counts)

class TCMDB:
    hf_repo: str = "f-galkin/batman2"
    hf_subsets: dict[str, str] = {'formulas': 'batman_formulas',
                       'herbs': 'batman_herbs',
                       'ingrs': 'batman_ingredients'}

    def __init__(self, p_batman: str):
        p_batman = p_batman.removesuffix("/") + "/"

        self.batman_files = dict(p_formulas='formula_browse.txt',
                                 p_herbs='herb_browse.txt',
                                 p_pred_by_tg='predicted_browse_by_targets.txt',
                                 p_known_by_tg='known_browse_by_targets.txt',
                                 p_pred_by_ingr='predicted_browse_by_ingredinets.txt',
                                 p_known_by_ingr='known_browse_by_ingredients.txt')

        self.batman_files = {x: p_batman + y for x, y in self.batman_files.items()}

        self.ingrs = None
        self.herbs = None
        self.formulas = None

    @classmethod
    def make_new_db(cls, p_batman: str):
        new_db = cls(p_batman)

        new_db.parse_ingredients()
        new_db.parse_herbs()
        new_db.parse_formulas()

        return (new_db)

    def parse_ingredients(self):

        pred_tgs = pd.read_csv(self.batman_files['p_pred_by_tg'],
                               sep='\t', index_col=None, header=0,
                               na_filter=False)
        known_tgs = pd.read_csv(self.batman_files['p_known_by_tg'],
                                sep='\t', index_col=None, header=0,
                                na_filter=False)
        entrez_to_symb = {int(pred_tgs.loc[x, 'entrez_gene_id']): pred_tgs.loc[x, 'entrez_gene_symbol'] for x in
                          pred_tgs.index}
        # 9927 gene targets
        entrez_to_symb.update({int(known_tgs.loc[x, 'entrez_gene_id']): \
                                   known_tgs.loc[x, 'entrez_gene_symbol'] for x in known_tgs.index})

        known_ingreds = pd.read_csv(self.batman_files['p_known_by_ingr'],
                                    index_col=0, header=0, sep='\t',
                                    na_filter=False)
        # this BATMAN table is badly formatted
        # you cant just read it
        # df_pred = pd.read_csv(p_pred, index_col=0, header=0, sep='\t')
        pred_ingreds = dict()
        with open(self.batman_files['p_pred_by_ingr'], 'r') as f:
            # skip header
            f.readline()
            newline = f.readline()
            while newline != '':
                cid, other_line = newline.split(' ', 1)
                name, entrez_ids = other_line.rsplit(' ', 1)
                entrez_ids = [int(x.split("(")[0]) for x in entrez_ids.split("|") if not x == "\n"]
                pred_ingreds[int(cid)] = {"targets": entrez_ids, 'name': name}
                newline = f.readline()

        all_BATMAN_CIDs = list(set(pred_ingreds.keys()) | set(known_ingreds.index))
        all_BATMAN_CIDs = [int(x) for x in all_BATMAN_CIDs if str(x).strip() != 'NA']

        # get targets for selected cpds
        ingredients = dict()
        for cid in all_BATMAN_CIDs:
            known_name, pred_name, synonyms = None, None, []
            if cid in known_ingreds.index:
                known_name = known_ingreds.loc[cid, 'IUPAC_name']
                known_symbs = known_ingreds.loc[cid, 'known_target_proteins'].split("|")
            else:
                known_symbs = []

            pred_ids = pred_ingreds.get(cid, [])
            if pred_ids:
                pred_name = pred_ids.get('name')
                if known_name is None:
                    cpd_name = pred_name
                elif known_name != pred_name:
                    cpd_name = min([known_name, pred_name], key=lambda x: sum([x.count(y) for y in "'()-[]1234567890"]))
                    synonyms = [x for x in [known_name, pred_name] if x != cpd_name]

                pred_ids = pred_ids.get('targets', [])

            ingredients[cid] = dict(pref_name=cpd_name,
                                    synonyms=synonyms,
                                    targets_known={"symbols": known_symbs,
                                                   "entrez_ids": [int(x) for x, y in entrez_to_symb.items() if
                                                                  y in known_symbs]},
                                    targets_pred={"symbols": [entrez_to_symb.get(x) for x in pred_ids],
                                                  "entrez_ids": pred_ids})
        ingredients_objs = {x: Ingredient(cid=x, **y) for x, y in ingredients.items()}
        self.ingrs = ingredients_objs

    def parse_herbs(self):
        if self.ingrs is None:
            raise ValueError("Herbs cannot be added before the ingredients")
        # load the herbs file
        name_cols = ['Pinyin.Name', 'Chinese.Name', 'English.Name', 'Latin.Name']
        herbs_df = pd.read_csv(self.batman_files['p_herbs'],
                               index_col=None, header=0, sep='\t',
                               na_filter=False)
        for i in herbs_df.index:

            herb_name = herbs_df.loc[i, 'Pinyin.Name'].strip()
            if herb_name == 'NA':
                herb_name = [x.strip() for x in herbs_df.loc[i, name_cols].tolist() if not x == 'NA']
                herb_name = [x for x in herb_name if x != '']
                if not herb_name:
                    raise ValueError(f"LINE {i}: provided a herb with no names")
                else:
                    herb_name = herb_name[-1]

            herb_cids = herbs_df.loc[i, 'ingrs'].split("|")

            herb_cids = [x.split("(")[-1].removesuffix(")").strip() for x in herb_cids]
            herb_cids = [int(x) for x in herb_cids if x.isnumeric()]

            missed_ingrs = [x for x in herb_cids if self.ingrs.get(x) is None]
            for cid in missed_ingrs:
                self.add_ingredient(cid=int(cid), pref_name='',
                                    empty_override=True)
            herb_ingrs = [self.ingrs[int(x)] for x in herb_cids]

            self.add_herb(pref_name=herb_name,
                          ingrs=herb_ingrs,
                          synonyms=[x for x in herbs_df.loc[i, name_cols].tolist() if not x == "NA"],
                          empty_override=True)

    def parse_formulas(self):
        if self.herbs is None:
            raise ValueError("Formulas cannot be added before the herbs")
        formulas_df = pd.read_csv(self.batman_files['p_formulas'], index_col=None, header=0,
                                  sep='\t', na_filter=False)
        for i in formulas_df.index:

            composition = formulas_df.loc[i, 'Pinyin.composition'].split(",")
            composition = [x.strip() for x in composition if not x.strip() == 'NA']
            if not composition:
                continue

            missed_herbs = [x.strip() for x in composition if self.herbs.get(x) is None]
            for herb in missed_herbs:
                self.add_herb(pref_name=herb,
                              desc='Missing in the original herb catalog, but present among formula components',
                              ingrs=[], empty_override=True)

            formula_herbs = [self.herbs[x] for x in composition]
            self.add_formula(pref_name=formulas_df.loc[i, 'Pinyin.Name'].strip(),
                             synonyms=[formulas_df.loc[i, 'Chinese.Name']],
                             herbs=formula_herbs)

    def add_ingredient(self, **kwargs):
        if self.ingrs is None:
            self.ingrs = dict()

        new_ingr = Ingredient(**kwargs)
        if not new_ingr.cid in self.ingrs:
            self.ingrs.update({new_ingr.cid: new_ingr})

    def add_herb(self, **kwargs):
        if self.herbs is None:
            self.herbs = dict()

        new_herb = Herb(**kwargs)
        old_herb = self.herbs.get(new_herb.pref_name)
        if not old_herb is None:
            if_same = new_herb.is_same(old_herb)
            if if_same:
                return

            same_name = new_herb.pref_name
            all_dupes = [self.herbs[x] for x in self.herbs if x.split('~')[0] == same_name] + [new_herb]
            new_names = [same_name + f"~{x + 1}" for x in range(len(all_dupes))]
            for i, duped in enumerate(all_dupes):
                duped.pref_name = new_names[i]
            self.herbs.pop(same_name)
            self.herbs.update({x.pref_name: x for x in all_dupes})
        else:
            self.herbs.update({new_herb.pref_name: new_herb})

        for cpd in new_herb.ingrs:
            cpd_herbs = [x.pref_name for x in cpd.herbs]
            if not new_herb.pref_name in cpd_herbs:
                cpd.herbs.append(new_herb)

    def add_formula(self, **kwargs):

        if self.formulas is None:
            self.formulas = dict()

        new_formula = Formula(**kwargs)
        old_formula = self.formulas.get(new_formula.pref_name)
        if not old_formula is None:
            is_same = new_formula.is_same(old_formula)
            if is_same:
                return
            same_name = new_formula.pref_name
            all_dupes = [self.formulas[x] for x in self.formulas if x.split('~')[0] == same_name] + [new_formula]
            new_names = [same_name + f"~{x + 1}" for x in range(len(all_dupes))]
            for i, duped in enumerate(all_dupes):
                duped.pref_name = new_names[i]
            self.formulas.pop(same_name)
            self.formulas.update({x.pref_name: x for x in all_dupes})
        else:
            self.formulas.update({new_formula.pref_name: new_formula})

        for herb in new_formula.herbs:
            herb_formulas = [x.pref_name for x in herb.formulas]
            if not new_formula.pref_name in herb_formulas:
                herb.formulas.append(new_formula)

    def link_ingredients_n_formulas(self):
        for h in self.herbs.values():
            for i in h.ingrs:
                fla_names = set(x.pref_name for x in i.formulas)
                i.formulas += [x for x in h.formulas if not x.pref_name in fla_names]
            for f in h.formulas:
                ingr_cids = set(x.cid for x in f.ingrs)
                f.ingrs += [x for x in h.ingrs if not x.cid in ingr_cids]

    def serialize(self, add_links = True):
        out_dict = dict(
            ingrs={cid: ingr.serialize(links=add_links) for cid, ingr in self.ingrs.items()},
            herbs={name: herb.serialize(links=add_links) for name, herb in self.herbs.items()},
            formulas={name: formula.serialize(links=add_links) for name, formula in self.formulas.items()}
        )
        return (out_dict)

    def save_to_flat_json(self, p_out: str):
        ser_db = db.serialize()
        flat_db = dict()
        for ent_type in ser_db:
            for i, obj in ser_db[ent_type].items():
                flat_db[f"{ent_type}:{i}"] = obj
        with open(p_out, "w") as f:
            f.write(json.dumps(flat_db))

    def save_to_json(self, p_out: str):
        with open(p_out, "w") as f:
            json.dump(self.serialize(), f)

    @classmethod
    def load(cls, ser_dict: dict):
        db = cls(p_batman="")

        # make sure to create all entities before you link them together
        db.ingrs = {int(cid): Ingredient.load(db, ingr, skip_links=True) for cid, ingr in
                    ser_dict['ingrs'].items()}
        db.herbs = {name: Herb.load(db, herb, skip_links=True) for name, herb in ser_dict['herbs'].items()}
        db.formulas = {name: Formula.load(db, formula, skip_links=True) for name, formula in
                       ser_dict['formulas'].items()}

        # now set the links
        for i in db.ingrs.values():
            # NB: somehow gotta make it work w/out relying on str-int conversion
            i._set_links(db, ser_dict['ingrs'][str(i.cid)]['links'])
        for h in db.herbs.values():
            h._set_links(db, ser_dict['herbs'][h.pref_name]['links'])
        for f in db.formulas.values():
            f._set_links(db, ser_dict['formulas'][f.pref_name]['links'])
        return (db)

    @classmethod
    def read_from_json(cls, p_file: str):
        if p_file.lower().endswith(".json"):
            with open(p_file, "r") as f:
                json_db = json.load(f)
        elif p_file.lower().endswith(".gz"):
            import gzip
            with gzip.open(p_file, 'rt') as f:
                json_db = json.load(f)
        print("✅ JSON file has been read!\nLoading the DB into memory...")
        db = cls.load(json_db)
        print("✅ Database has been loaded successfully!")
        return (db)

    @classmethod
    def from_huggingface_raw(cls):

        print("\n⚠️  Warning: Loading the dataset from HuggingFace can take  5-10 minutes.")
        print("☕ Grab a coffee while waiting! The process has started...\n")
        print("We recommend loading the JSON file from the repo with from_huggingface() method.")

        from datasets import load_dataset

        dsets = dict()
        for entity_type, subset_name in cls.hf_subsets.items():
            dsets[entity_type] = load_dataset(cls.hf_repo, subset_name)

        known_tgs = {str(x['cid']): [y.split("(") for y in eval(x['targets_known'])] for x in dsets['ingrs']['train']}
        known_tgs = {x:{'symbols':[z[0] for z in y], "entrez_ids":[int(z[1].strip(")")) for z in y]} for x,y in known_tgs.items()}
        pred_tgs = {str(x['cid']): [y.split("(") for y in eval(x['targets_pred'])] for x in dsets['ingrs']['train']}
        pred_tgs = {x:{'symbols':[z[0] for z in y], "entrez_ids":[int(z[1].strip(")")) for z in y]} for x,y in pred_tgs.items()}

        json_db = dict()
        json_db['ingrs'] = {str(x['cid']): {'init': dict(cid=int(x['cid']),
                                                           targets_known=known_tgs[str(x['cid'])],
                                                           targets_pred=pred_tgs[str(x['cid'])],
                                                           pref_name=x['pref_name'],
                                                           synonyms=eval(x['synonyms']),
                                                           desc=x['description']
                                                           ),

                                                  'links': dict(
                                                      herbs=eval(x['herbs']),
                                                      formulas=eval(x['formulas'])
                                                  )
                                                  }
                                  for x in dsets['ingrs']['train']}

        json_db['herbs'] = {x['pref_name']: {'init': dict(pref_name=x['pref_name'],
                                                          synonyms=eval(x['synonyms']),
                                                          desc=x['description']),
                                             'links': dict(ingrs=eval(x['ingredients']),
                                                           formulas=eval(x['formulas']))} for x in
                            dsets['herbs']['train']}

        json_db['formulas'] = {x['pref_name']: {'init': dict(pref_name=x['pref_name'],
                                                             synonyms=eval(x['synonyms']),
                                                             desc=x['description']),
                                                'links': dict(ingrs=eval(x['ingredients']),
                                                              herbs=eval(x['herbs']))} for x in
                               dsets['formulas']['train']}

        db = cls.load(json_db)
        return (db)

    @classmethod
    def from_huggingface(cls):
        """
        Downloads and reads the compressed JSON file from HuggingFace repository.
        Uses the batman2/data repository and BATMAN_DB_v3.json.gz file.
        """
        from huggingface_hub import hf_hub_download

        print("⚠️ Loading this dataset usually takes ~3-5 minutes.")
        print("⏳ Downloading compressed database file from HuggingFace...")

        try:
            file_path = hf_hub_download(
                repo_id=cls.hf_repo,
                filename="data/BATMAN_DB_v3.json.gz",
                repo_type="dataset"
            )

            print("📂 Database file downloaded, now decompressing and loading...")

            return(cls.read_from_json(file_path))


        except Exception as e:
            print(f"❌ Error loading database: {str(e)}")
            raise


    def drop_isolated(self, how='any'):
        match how:
            case 'any':
                self.herbs = {x: y for x, y in self.herbs.items() if (y.ingrs and y.formulas)}
                self.formulas = {x: y for x, y in self.formulas.items() if (y.ingrs and y.herbs)}
                self.ingrs = {x: y for x, y in self.ingrs.items() if (y.formulas and y.herbs)}
            case 'all':
                self.herbs = {x: y for x, y in self.herbs.items() if (y.ingrs or y.formulas)}
                self.formulas = {x: y for x, y in self.formulas.items() if (y.ingrs or y.herbs)}
                self.ingrs = {x: y for x, y in self.ingrs.items() if (y.formulas or y.herbs)}
            case _:
                raise ValueError(f'Unknown how parameter: {how}. Known parameters are "any" and "all"')


    def create_formula_from_herbs(self,
                                  herbs: list[str]) -> 'Formula':
        present_herbs = set(x for x in herbs if x in self.herbs)
        if not present_herbs:
            return

        present_ingrs = list(
                            set(
                                sum([list([z.cid for z in self.herbs[x].ingrs]) for x in present_herbs], [])
                                )
                            )

        custom_flas = [x for x in self.formulas if x.startswith('Custom Formula ~')]
        if not custom_flas:
            custom_no = 1
        else:
            custom_no = max([int(x.rsplit("~",1)[-1]) for x in custom_flas])+1

        new_fla = Formula(pref_name = f"Custom Formula ~{custom_no}",
                          herbs = [self.herbs[x] for x in present_herbs])

        new_fla._set_links(self,
                           {'herbs':present_herbs,
                                  'ingrs':present_ingrs,
                                  'formulas':[]})
        return(new_fla)


    def pick_N_formulas_by_cids(self,
                              cids: list[int],
                              N_herbs: int = 4,
                              N_top: int = 1,
                              blacklist = None):
        present_cids = [x for x in cids if x in self.ingrs]
        if not present_cids:
            return
        raise NotImplementedError

    def select_top_items(self,
                         items: list,
                         data_map: str,
                         item_type: str,
                         min_items: int = 2,
                         N_top: int = 1,
                         blacklist: Optional[set] = None) -> dict[str, int]:
                             
        blacklist = blacklist or set()
        blacklist = set(blacklist)

        match item_type:
            case "herbs":
                lookup_attr = "pref_name"
            case 'ingrs':
                lookup_attr = "cid"
            case _:
                raise ValueError()

        present_check_set = getattr(self, item_type)

        present_items = set(x for x in items if x in present_check_set)
        if not present_items:
            return (dict())

        # only lookup entities that have at least 1 queries entity linked
        primary_entities = [present_check_set[x] for x in present_items]
        linked_entities = set(sum([x._get_link_dict()[data_map] for x in primary_entities],[]))
        linked_entities = {x:getattr(self,data_map)[x] for x in linked_entities}

        item_counts = {
            x: len(set(getattr(z, lookup_attr) for z in y.__dict__[item_type] ) & present_items)
            for x, y in linked_entities.items() if not x in blacklist
        }

        # Filter, descending sort, and select the top items
        item_counts = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        item_counts = [(x, y) for x, y in item_counts if y >= min_items and not x in blacklist]

        best_items = {x:y for x, y in item_counts[:N_top]}

        if not best_items:
            return (dict())

        return (best_items)

    def select_N_formulas_by_cids(self,
                                 cids: list[int],
                                 min_cids: int = 2,
                                 N_top: int = 1,
                                 blacklist: Optional[set] = None):
        return self.select_top_items(
            items=cids,
            item_type='ingrs',
            data_map='formulas',
            min_items=min_cids,
            N_top=N_top,
            blacklist = blacklist
        )

    def select_N_herbs_by_cids(self,
                               cids: list[int],
                               min_cids: int = 2,
                               N_top: int = 4,
                               blacklist: Optional[set] = None):
        return self.select_top_items(
            items=cids,
            item_type='ingrs',
            data_map='herbs',
            min_items=min_cids,
            N_top=N_top,
            blacklist = blacklist
        )

    def select_N_formulas_by_herbs(self,
                                   herbs: list[str],
                                   min_herbs: int = 2,
                                   N_top: int = 1,
                                   blacklist: Optional[set] = None):
        return self.select_top_items(
            items=herbs,
            item_type='herbs',
            data_map='formulas',
            min_items=min_herbs,
            N_top=N_top,
            blacklist = blacklist
        )

    def order_flas_synergistically(self,
                                  cids:list[int],
                                  flas:list[str]):
        present_flas = [x for x in flas if x in self.formulas]
        present_cids = [x for x in cids if x in self.ingrs]
        if not present_flas or not present_cids:
            return

        cpd_counts = {x:sum([self.count_cpd_occurrences(y, x) for y in cids]) for x in present_flas}
        ord_flas = sorted(cpd_counts.items(), key=lambda x: x[1], reverse=True)
        return( [x for x,y in ord_flas] )

    def count_cpd_occurrences(self, cid:int, fla: str)->int:
        if not cid in self.ingrs or not fla in self.formulas:
            return

        these_herbs = self.formulas[fla].herbs
        cid_count = [1 if cid in x._get_link_dict()['ingrs'] else 0 for x in these_herbs]
        return(sum(cid_count))

    def lookup_identical_herb_formula(self, herbs: list[str]) -> str:
        L = len(herbs)
        candidates = self.lookup_similar_herb_formula(herbs)
        for name in candidates:
            if len(self.formulas[name].get_herbs()) == L:
                return(name)
        return('')

    def lookup_similar_herb_formulas(self, herbs: list[str],
                                     buffer: int = 0) -> list[str]:
        L = len(herbs)
        max_hit, candidates = self.select_N_formulas_by_herbs(herbs=herbs,
                                                              min_herbs=L-buffer,
                                                              N_top=1000)

        candidates = sorted(candidates, key = lambda x: len(self.formulas[x].get_herbs()), reverse=False)
        return(candidates)

    def find_enriched_cpds(self, glist: Iterable[str],
                           tg_type: str = 'known', tg_names: str = "symbols",
                           thr: float = 0.001, bg_genes: int = 25332,
                           cpd_subset: Optional[set] = None,
                           blacklist: Optional[set] = None) -> dict[int,dict[str,Any]]:

        if not tg_type in ('known', 'predicted', 'both'):
            raise ValueError(f"Unknown type of target '{tg_type}'.\nAccepted types: known, predicted")

        if not tg_names in ('entrez_ids', 'symbols'):
            raise ValueError(f"Unknown target name category '{tg_names}'.\nAccepted types: symbols, entrez_ids")

        glist = set(glist)
        enrich_stats = dict()

        if cpd_subset is None:
            cpd_subset = set(self.ingrs.keys())
        else:
            cpd_subset = set([x for x in cpd_subset])
            cpd_subset = cpd_subset.intersection(set(self.ingrs.keys()))

        blacklist = blacklist or set()
        cpd_subset -= blacklist

        total_compars = len(cpd_subset)

        for cid, cpd_obj in self.ingrs.items():

            if not cid in cpd_subset:
                continue

            if tg_type != 'both':
                tg_set = set(self.ingrs[cid].targets[tg_type][tg_names])
            else:
                tg_set = set(self.ingrs[cid].targets['known'][tg_names]) | set(self.ingrs[cid].targets['predicted'][tg_names])


            # cols: in targets, not in targets
            # rows: in P3 output, not in P3 output
            hits = glist&tg_set
            cont_table = (a := np.array([[len(hits), len(tg_set - hits)],
                                         [len(hits - tg_set), 0]])) + \
                         np.array([[0, 0],
                                   [0, bg_genes - a.sum().sum()]])

            F_exact = fisher_exact(table=cont_table, alternative='greater')
            # ['compound', 'tissue', 'domain', 'p_value', 'target_type']
            enrich_stats[cid] = {"name":cpd_obj.pref_name,
                                 "Pv":F_exact[1],
                                 "corrPv":min(1, F_exact[1]*total_compars),
                                 "targets_hit":len(hits),
                                 "total_targets":len(tg_set)}
        
        sign_findings = {x:y for x,y in enrich_stats.items() if y['corrPv']<thr}
        
        sorted_items = sorted(
                sign_findings.items(),
                key=lambda x: (
                    x[1]['corrPv'],  # Lower p-value first
                    -x[1]['targets_hit'],  # More targets hit
                    -x[1]['targets_hit']/x[1]['total_targets']  # Higher percentage of relevant targets
                )
            )
        
        return dict(sorted_items)

    def select_herbs_for_targets(self,
                                glist: list|set,
                                tg_type: str = 'known',
                                tg_names: str = "symbols",
                                blacklist: Optional[set]= None)->dict[str, int]:
        blacklist = blacklist or set()
        blacklist = set(blacklist)

        glist = set(glist)
        # check compounds:
        #   - count N times a target is mentioned
        h_counts = dict()
        for h, h_obj in self.herbs.items():
            if h in blacklist:
                continue
            herb_count = 0
            for c_obj in h_obj.ingrs:
                if tg_type != 'both':
                    tg_set = set(c_obj.targets[tg_type][tg_names])
                else:
                    tg_set = set(c_obj.targets['known'][tg_names]) | set(c_obj.targets['predicted'][tg_names])

                herb_count += len(tg_set&glist)
            h_counts[h] = herb_count

        h_counts = dict(sorted(h_counts.items(), key=lambda x: x[1], reverse=True))
        return(h_counts)

    def get_greedy_formula(self, cids, max_herb=1000, blacklist = None):
        if blacklist is None:
            blacklist = [""]
        cids = set(cids) & set(self.ingrs.keys())
        selected_herbs = set()
        while cids:
            selected_herb = self.select_N_herbs_by_cids(cids, min_cids=1, N_top=100)[1]
            selected_herb = [x for x in selected_herb if not x in blacklist]
            if selected_herb:
                selected_herb = selected_herb[:1]
                selected_herbs |= set(selected_herb)
            else:
                break
            cids -= set([x.cid for x in self.herbs[selected_herb[0]].ingrs])
            if len(selected_herbs) == max_herb:
                break
        new_fla = self.create_formula_from_herbs(list(selected_herbs))
        return(new_fla)

def main():
    pass


if __name__ == '__main__':
    main()