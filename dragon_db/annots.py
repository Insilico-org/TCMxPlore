import pickle
from typing import Any
from itertools import permutations
import json

class TCMAnnotation:

    entity: str = 'superclass'

    preferred_name: str = ''
    synonyms: list[str] = []
    links: dict[str, list] = dict()

    def __init__(self, **kwargs):

        self.preferred_name = ''
        self.synonyms = []
        self.links = {}

        for k, v in kwargs.items():
            if k in (self.__dict__|self.__class__.__dict__):
                if isinstance(v, list):
                    v = [x for x in v if x and not isinstance(x, bool)]
                self.__dict__[k] = v

        self._empty_override = True

    def serialize(self, links=True)->dict:
        base_params = {x:y for x,y in self.__dict__.items() if not x.startswith('_') and x!= "links"}
        link_params = dict()
        if links:
            link_params = self._get_link_dict()
        out_dict = {"init":base_params, "links":link_params}
        return out_dict

    @classmethod
    def load(cls,
             db: 'TCMAnnotationDB',
             ser_dict: dict[str, Any],
             skip_links: bool=True):

        if "init" in ser_dict:
            init_args = ser_dict['init']
        else:
            init_args = ser_dict

        init_args.update({"_empty_override":skip_links})

        new_entity = cls(**init_args)
        if not skip_links:
            new_entity._set_links(db, ser_dict['links'])
        return (new_entity)

    def _get_link_dict(self)->dict:
        return {x:[z.preferred_name for z in y] for x,y in self.links.items()}

    def _set_links(self,
                   db: 'TCMAnnotationDB',
                   links: dict[str, list[str]]):
        for ent_type in links:
            self.links[ent_type] = [db.__dict__[ent_type].get(x) for x in links[ent_type]]
            self.links[ent_type] = [x for x in self.links[ent_type] if x is not None]

            for other in self.links[ent_type]:
                if not self.entity in other.links:
                    other.links[self.entity] = []
                other.links[self.entity].append(self)

    def is_isolated(self):
        if self.links:
            if any([x for x in self.links.values()]):
                return(False)
        return(True)

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.preferred_name}>"

class HerbAnnotation(TCMAnnotation):

    entity = "herbs"

    category: list[str] = []
    properties: dict[str,dict[str,list[str]]] = {
                                                  'taste':{'primary':[],
                                                           'secondary':[]},
                                                  'temperature':{'primary':[],
                                                                'secondary':[]},
                                                  'meridians':{'primary':[],
                                                               'secondary':[]}
                                                  }
    dosage: list[dict[str,str]] = [{'form':"", 'min':"", "max":''}]
    actions: list[dict[str,str]] = [{'action':'', 'indication':''}]

    indications: list[str] = []
    contraindications: list[str] = []
    interactions: list[str] = []
    incompatibility: list[str] = []

    notes: list[str] = []

class ConditionAnnotation(TCMAnnotation):

    entity = "conditions"

    description: list[str] = []

    symptoms: list[dict[str, str|list[str]]] = [{'name': '',
                     'clinical_manifestations':[''],
                     'treatment_principle':[''],
                     'herb_formulas' : [''],
                     'points':['']
                   }]
    herb_formulas: list[str] = []
    points: list[str] = []

class FormulaAnnotation(TCMAnnotation):

    entity = "formulas"


    composition: dict[str,dict[str,Any]] = {"":
                                                {'latin':'',
                                                 'actions':[],
                                                 'dosage':''
                                                 }
                                            }
    treats: list[str] = []
    syndromes : list[str] = []
    actions: list[str] = []

    contraindications: list[str] = []

    notes: list[str] = []

    def get_herbs(self)->list[str]:
        return(list(set([x.preferred_name for x in self.links['herbs']])))

    def set_herbs(self, db, herbs: list[str]):
        self.links['herbs'] = [db.herbs[x] for x in herbs]
class TCMAnnotationDB:
    hf_repo: str = "f-galkin/DragonTCM"

    def __init__(self, p_data=None, filetype=None):
        self.storage = p_data
        match filetype:
            case 'pckl':
                with open(self.storage, "rb") as f:
                    self.raw_data = pickle.load(f)
            case 'json':
                with open(self.storage, "rb") as f:
                    self.raw_data = json.load(f)
            case _:
                if not p_data is None:
                    raise ValueError(f"Format {filetype} not supported. Use json or pckl")

        self.herbs = dict()
        self.conditions = dict()
        self.formulas = dict()

        self.stragglers = {"formulas":set(), "conditions":set(), "herbs":set()}
        self.not_found = {"formulas":set(), "conditions":set(), "herbs":set()}
        self.implicit = {"formulas":set(), "conditions":set(), "herbs":set()}

        self.word_map = {"formulas":dict(), "conditions":dict(), "herbs":dict()}
        self.word_collisions = {"formulas":[], "conditions":[], "herbs":[]}

    @classmethod
    def from_huggingface(cls):
        """
        Create a TCMAnnotationDB instance from HuggingFace dataset.

        Args:
            dataset_name (str): HuggingFace dataset name

        Returns:
            TCMAnnotationDB: New database instance
        """
        from datasets import load_dataset

        # Initialize empty database
        db = cls()
        db.raw_data = {"herbs": {}, "conditions": {}, "formulas": {}}

        # Load all configurations
        configs = ["herbs", "conditions", "formulas"]
        for config in configs:
            dataset = load_dataset(cls.hf_repo, name=config, trust_remote_code=True)

            # Process each entry
            for item in dataset["train"]:
                entry_dict = {
                    "init": {
                        "preferred_name": item["name"],
                        "synonyms": json.loads(item["synonyms"]),
                    }
                }

                # Add config-specific fields
                if config == "herbs":
                    entry_dict["init"].update({
                        "category": json.loads(item["category"]),
                        "properties": json.loads(item["properties"]),
                        "actions": json.loads(item["actions"]),
                        "contraindications": json.loads(item["contraindications"]),
                        "interactions": json.loads(item["interactions"]),
                        "incompatibility": json.loads(item["incompatibility"]),
                        "notes": json.loads(item["notes"]),
                        "dosage": json.loads(item["dosage"]),
                        "indications": json.loads(item["indications"])
                    })
                elif config == "conditions":
                    entry_dict["init"].update({
                        "symptoms": json.loads(item["symptoms"]),
                        "description": str(item["description"]),
                        "herb_formulas": json.loads(item["herb_formulas"]),
                        "points": json.loads(item["points"])
                    })
                elif config == "formulas":
                    entry_dict["init"].update({
                        "actions": json.loads(item["actions"]),
                        "syndromes": json.loads(item["syndromes"]),
                        "treats": json.loads(item["treats"]),
                        "contraindications": json.loads(item["contraindications"]),
                        "notes": json.loads(item["notes"]),
                        "composition": json.loads(item["composition"])
                    })

                db.raw_data[config][item["name"].upper()] = entry_dict

        # Load relations
        relations = load_dataset(cls.hf_repo, name="relations", trust_remote_code=True)

        # Process relations
        for relation in relations["train"]:
            source = relation["source"].upper()
            target = relation["target"].upper()
            source_type = relation["source_type"] + "s"  # add 's' to match db keys
            target_type = relation["target_type"] + "s"

            # Add links to raw_data
            if source in db.raw_data[source_type]:
                if "links" not in db.raw_data[source_type][source]:
                    db.raw_data[source_type][source]["links"] = {}
                if target_type not in db.raw_data[source_type][source]["links"]:
                    db.raw_data[source_type][source]["links"][target_type] = []
                db.raw_data[source_type][source]["links"][target_type].append(target)

        # Initialize database objects
        for dtype in ["herbs", "conditions", "formulas"]:
            for info in db.raw_data[dtype].values():
                db._add_entity(dtype=dtype, **info)
            print(f"Added {len(db.__dict__[dtype])} {dtype}")

        # Set links between entities
        for dtype in db.raw_data:
            for k, obj in db.__dict__[dtype].items():
                if "links" in db.raw_data[dtype][k]:
                    obj._set_links(db, db.raw_data[dtype][k]["links"])

        db.identify_stragglers()

        return db

    @classmethod
    def make_new_db(cls, p_data):
        new_db = cls(p_data)

        for d in ('herbs', 'conditions', 'formulas'):
            for info in new_db.raw_data[d].values():
                new_db._add_entity(dtype = d, **info)
            print(f"Added {len(new_db.__dict__[d])} {d}")
        for d in ('herbs', 'conditions', 'formulas'):
            new_db.add_implicit_entities(dtype = d)

        new_db.cross_link_entities()

        new_db.identify_stragglers()
        # new_db.make_word_map()

        return(new_db)

    @classmethod
    def load(cls, p_file, filetype):
        new_db = cls(p_file, filetype)

        for dtype in new_db.raw_data:
            for info in new_db.raw_data[dtype].values():
                new_db._add_entity(dtype=dtype, **info)

        for dtype in new_db.raw_data:
            for k,obj in new_db.__dict__[dtype].items():
                obj._set_links(new_db, new_db.raw_data[dtype][k]['links'])

        new_db.identify_stragglers()

        return(new_db)

    def parse_entities(self, dtype = None):

        match dtype:
            case 'herbs':
                ent_cls = HerbAnnotation
            case 'conditions':
                ent_cls = ConditionAnnotation
            case 'formulas':
                ent_cls = FormulaAnnotation
            case 'ingrs':
                raise NotImplementedError()
            case _:
                raise ValueError("mode must be specified as one of: formulas, herbs, conditions")

        entity_dict = {x.upper(): ent_cls.load(self, ser_dict=y) for x, y in self.raw_data[dtype].items()}
        self.__dict__[dtype] = entity_dict

    def add_implicit_entities(self, dtype: str):
        implicit_mentions = self.find_missing_entries(dtype=dtype)
        for name in implicit_mentions:
            self._add_entity(dtype=dtype, preferred_name = name)
            self.implicit[dtype].add(name)

    def find_missing_entries(self, dtype: str, verbose = True):
        misses = set()
        match dtype:
            case 'formulas':
                # find FORMULAS that are mentioned in CONDITIONS but lack their own entry
                for k, cond_obj in self.conditions.items():
                    if cond_obj.herb_formulas:
                        cond_flas = [y.upper() for x,y in cond_obj.herb_formulas]
                        misses |= set([x for x in cond_flas if not x in self.formulas])

            case 'conditions':
                # find CONDITIONS that are mentioned in FORMULAS but lack their own entry
                for k, fla_obj in self.formulas.items():
                    if fla_obj.treats:
                        fla_indics = [x.upper() for x in fla_obj.treats]
                        misses |= set([x for x in fla_indics if not x in self.conditions])
                # find CONDITIONS that are mentioned in HERBS but lack their own entry
                for k, herb_obj in self.herbs.items():
                    if herb_obj.indications:
                        herb_indics = [x.upper() for x in herb_obj.indications]
                        misses |= set([x for x in herb_indics if not x in self.conditions])
            case 'herbs':
                # find HERBS that are mentioned in FORMULAS but lack their own entry
                for k, fla_obj in self.formulas.items():
                    if fla_obj.composition:
                        fla_compons = [x.upper() for x in fla_obj.composition]
                        misses |= set([x for x in fla_compons if not x in self.herbs])
        if verbose:
            print(f"Detected {len(misses)} implicitly mentioned {dtype}")
        return(misses)

    def cross_link_entities(self):
        # Conditions[herb_formulas + symptoms-herb_formulas] → Formulas
        self._link_conds_n_flas()
        # Formulas[composition] → Herbs
        self._link_herbs_n_flas()
        # + Lookup Herbs[actions-indication] → Conditions
        self._link_herbs_n_conds()

        # (Conditions ←→ Formulas) → Herbs
        # DO NOT EXTRAPOLATE LINKS FOR NOW
        pass

        self._remove_duped_links()

    def _link_conds_n_flas(self):
        for k, cond_obj in self.conditions.items():
            cond_flas = set([y.upper() for x,y in cond_obj.herb_formulas])
            not_in_db = set([x for x in cond_flas if not x in self.formulas])
            self.not_found['formulas'] |= set(not_in_db)
            cond_flas -= not_in_db

            if not 'formulas' in cond_obj.links:
                cond_obj.links['formulas'] = []
            cond_obj.links['formulas'] += [self.formulas[x] for x in cond_flas]

            for f in cond_obj.links['formulas']:
                if not 'conditions' in f.links:
                    f.links['conditions'] = []
                f.links['conditions'].append(cond_obj)

        for k, fla_obj in self.formulas.items():
            fla_indics = set([x.upper() for x in fla_obj.treats])
            not_in_db = set([x for x in fla_indics if not x in self.conditions])
            self.not_found['conditions'] |= set(not_in_db)
            fla_indics -= not_in_db

            if not 'conditions' in fla_obj.links:
                fla_obj.links['conditions'] = []
            fla_obj.links['conditions'] += [self.conditions[x] for x in fla_indics]

            for c in fla_obj.links['conditions']:
                if not 'formulas' in c.links:
                    c.links['formulas'] = []
                c.links['formulas'].append(fla_obj)

    def _link_herbs_n_flas(self):
        for k, fla_obj in self.formulas.items():
            fla_herbs = set([x.upper() for x, y in fla_obj.composition.items()])
            not_in_db = set([x for x in fla_herbs if not x in self.herbs])
            fla_herbs -= not_in_db

            if not 'herbs' in fla_obj.links:
                fla_obj.links['herbs'] = []
            fla_obj.links['herbs'] += [self.herbs[x] for x in fla_herbs]

            for h in fla_obj.links['herbs']:
                if not 'formulas' in h.links:
                    h.links['formulas'] = []
                h.links['formulas'].append(fla_obj)

    def _link_herbs_n_conds(self):
        for k, herb_obj in self.herbs.items():
            herb_conds = set([x.upper() for x in herb_obj.indications])
            not_in_db = set([x for x in herb_conds if not x in self.conditions])
            herb_conds -= not_in_db

            if not 'conditions' in herb_obj.links:
                herb_obj.links['conditions'] = []
            herb_obj.links['conditions'] += [self.conditions[x] for x in herb_conds]

            for c in herb_obj.links['conditions']:
                if not 'herbs' in c.links:
                    c.links['herbs'] = []
                c.links['herbs'].append(herb_obj)

    def _remove_duped_links(self):
        link_pairs = permutations(('herbs', 'conditions', 'formulas'), 2)
        for ent_type, other in link_pairs:
            for k, obj in self.__dict__[ent_type].items():
                if not other in obj.links:
                    continue
                other_ids = [x.preferred_name for x in obj.links[other]]
                if len(obj.links[other]) == len(set(other_ids)):
                    continue
                first_occur = dict()
                for i,link in enumerate(obj.links[other]):
                    if link.preferred_name not in first_occur:
                        first_occur[link.preferred_name] = i
                obj.links[other] = [obj.links[other][x] for x in first_occur.values()]

    def _add_entity(self, dtype, **kwargs):
        match dtype:
            case 'herbs':
                ent_cls = HerbAnnotation
            case 'conditions':
                ent_cls = ConditionAnnotation
            case 'formulas':
                ent_cls = FormulaAnnotation
            case _:
                raise ValueError("mode must be specified as one of: formulas, herbs, conditions")

        ent_obj = ent_cls.load(self, ser_dict=kwargs)
        ent_obj.preferred_name = ent_obj.preferred_name.upper()

        self.__dict__[dtype].update({ent_obj.preferred_name: ent_obj})

    def report_n_links(self):
        msg = ''
        link_pairs = permutations(('herbs', 'conditions', 'formulas'), 2)
        for ent_type, other in link_pairs:
            n_links = [x.links.get(other) for x in self.__dict__[ent_type].values()]
            n_links = [len(x) for x in n_links if not x is None]
            msg += f"Links from {ent_type} to {other}: {sum(n_links)} ({sum(n_links)/len(n_links):.2f} per item)\n"
        return(msg)

    def serialize(self, add_links=True):
        out_dict = dict(
            conditions={x: y.serialize(links=add_links) for x,y in self.conditions.items()},
            herbs={x: y.serialize(links=add_links) for x,y in self.herbs.items()},
            formulas={x: y.serialize(links=add_links) for x,y in self.formulas.items()}
        )
        return (out_dict)

    def save_to_json(self, p_out):
        with open(p_out, "w") as f:
            json.dump(self.serialize(), f, indent = 4)

    def identify_stragglers(self):
        for dtype in self.stragglers:
            isolated_ents = [x for x in self.__dict__[dtype].values() if x.is_isolated()]
            self.stragglers[dtype] |= set(isolated_ents)

    def select_top_items(self,
                         items: list,
                         data_map: str,
                         item_type: str,
                         min_items: int = 2,
                         N_top: int = 1):

        lookup_attr = 'preferred_name'
        present_check_set = getattr(self, item_type)

        present_items = set(x for x in items if x in present_check_set)
        if not present_items:
            return (0, [])

        # only lookup entities that have at least 1 queries entity linked
        primary_entities = [present_check_set[x] for x in present_items]
        linked_entities = set(sum([x._get_link_dict()[data_map] for x in primary_entities],[]))
        linked_entities = {x:getattr(self,data_map)[x] for x in linked_entities}

        item_counts = {
            x: len(set([getattr(z, lookup_attr) for z in y.links[item_type] ]) & present_items)
            for x, y in linked_entities.items()
        }

        # Filter, sort, and select the top items
        item_counts = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        item_counts = [(x, y) for x, y in item_counts if y >= min_items]

        best_items = [x for x, y in item_counts[:N_top]]
        if not best_items:
            return (0, [])

        n_max = item_counts[0][1]
        return (n_max, best_items)

    def select_N_formulas_by_herbs(self,
                                   herbs: list[str],
                                   min_herbs: int = 2,
                                   N_top: int = 1):
        return self.select_top_items(
            items=herbs,
            item_type='herbs',
            data_map='formulas',
            min_items=min_herbs,
            N_top=N_top
        )

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
                                                              N_top=100)

        candidates = sorted(candidates, key = lambda x: len(self.formulas[x].get_herbs()), reverse=False)
        return(candidates)


def main():
    pass

if __name__ == "__main__":
    main()
