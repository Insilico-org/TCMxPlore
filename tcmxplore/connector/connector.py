# connector/connector.py
import json
from thefuzz import process, fuzz
from copy import deepcopy as dcp
from suffix_tree import Tree as SuffixTree
from functools import reduce

from batman_db import batman
from dragon_db import annots

from functools import wraps
import time

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def split_query(query):
    sub_q = [x.lower() for x in query.split(" ")]
    for i, w in enumerate(sub_q):
        sub_q[i] = reduce(lambda x,y: x.replace(y,""), ".-~/,'()", w)
    return(sub_q)

def match_N_w_score(q, word_map, N, scorer = None):
    if scorer is None:
        scorer = fuzz.token_set_ratio
    q = " ".join(split_query(q))
    word_map = [" ".join(split_query(x)) for x in word_map]
    hits = process.extract(q, word_map, limit=N, scorer=scorer)
    if hits:
        hits = sorted(hits, key=lambda x: x[1], reverse=True)
        return(hits)
    return(None)

def get_all_keys(d, keys=set()):

    for k,v in d.items():
        keys.add(k)
        if isinstance(v, dict):
            get_all_keys(v, keys)
    return keys
class DBConnector:

    def __init__(self):
        self.dbs = dict()
        self.p_dbs = dict()
        self.db_configs=dict()

        self.word_maps = dict()
        self.trees = dict()

        self.cross_mappings = dict()
        self.equivalents = dict()

    def save_to_file(self, p_out):
        out_dict = {x:getattr(self,x) for x in ("p_dbs", "db_configs")}
        out_dict["cross_mappings"] = {x: {str(z): a for z, a in y.items()} for x,y in self.cross_mappings.items()}
        out_dict["equivalents"] = {x: {str(z): a for z, a in y.items()} for x,y in self.equivalents.items()}
        out_dict["word_maps"] = {x:{z:{b:[getattr(d, "cid") if z == 'ingrs' else getattr(d, self.db_configs[x]['primary']) for d in c] for b,c in a.items()}for z,a in y.items()} for x,y in self.word_maps.items()}
        out_dict["db_types"] = {x:str(type(y)) for x,y in self.dbs.items()}
        with open(p_out, 'w') as f:
            json.dump(out_dict, f)

    @classmethod
    def load(self, p_in, verbose = True):

        new_con = DBConnector()

        if p_in.lower().endswith(".json"):
            with open(p_in, 'r') as f:
                in_dict = json.load(f)
        elif p_in.lower().endswith(".gz"):
            import gzip
            with gzip.open(p_in, 'rt') as f:
                in_dict = json.load(f)

        for attr in ("p_dbs", "db_configs", "equivalents"):
            setattr(new_con, attr, in_dict[attr])

        if in_dict["cross_mappings"]:
            if verbose:
                print("Adding cross mappings")
            new_con.cross_mappings = {x: {eval(z): a for z, a in y.items()} for x, y in
                                      in_dict['cross_mappings'].items()}
            if verbose:
                print("Done adding cross mappings")

        if in_dict['equivalents']:
            if verbose:
                print("Adding equivalents")
            new_con.equivalents = {x: {eval(z): a for z, a in y.items()} for x, y in
                                   in_dict['equivalents'].items()}
            if verbose:
                print("Done adding equivalents")

        for db_name, p_db in in_dict["p_dbs"].items():
            if verbose:
                print(f"Loading DB: {db_name}")
            db_type = in_dict['db_types'][db_name]
            if "batman.TCMDB" in db_type:
                this_db = batman.TCMDB.read_from_json(p_db)
            elif "annots.TCMAnnotationDB" in db_type:
                this_db = annots.TCMAnnotationDB.load(p_db, filetype='json')
            else:
                raise TypeError(f"Unknown type of TCM DB: {db_type}")
            new_con.dbs[db_name] = this_db
            if verbose:
                print(f"Uploaded DB: {db_name}")

        if in_dict["word_maps"]:
            if verbose:
                print("Adding word maps")
            new_con.word_maps = {x:
                                   {z:
                                      {b:
                                         [getattr(new_con.dbs[x], z)[d] for d in c] for b,c in a.items()
                                      }
                                   for z,a in y.items()}
                                 for x, y in in_dict["word_maps"].items()}
            if verbose:
                print("Done adding word maps")

        return(new_con)

    def add_db(self, db, name, p_db,
               primary_field, secondary_field,
               parse_fields = None,
               verbose = True,
               do_maps = True,
               do_trees = False):

        if parse_fields is None:
            parse_fields = ['formulas', 'herbs']
        if not name in self.dbs:
            self.dbs.update({name:db})
        if not name in self.p_dbs:
            self.p_dbs[name] = p_db
        if not name in self.db_configs:
            self.db_configs[name] = {'primary':primary_field,
                                     'secondary':secondary_field}

        if do_maps:
            if verbose:
                print(f"Adding word map for {name}")
            self.word_maps[name] = {x:dict() for x in parse_fields}
            self.make_word_map(name)

        if do_trees:
            if verbose:
                print(f"Adding suffix tree for {name}")
            self.trees[name] = {x: dict() for x in parse_fields}
            self.make_trees(name)

    def make_word_map(self, db_id):
        this_db = self.dbs[db_id]
        this_map = self.word_maps[db_id]
        name_field = self.db_configs[db_id]['primary']
        syn_field = self.db_configs[db_id]['secondary']

        for dtype in this_map:
            for k, obj in getattr(this_db, dtype).items():
                # collect all ptrs first
                all_ptrs = [getattr(obj, name_field)] + getattr(obj, syn_field)
                all_ptrs =  [" ".join(split_query(x)) for x in all_ptrs]
                all_ptrs = list(set(all_ptrs))
                for p in all_ptrs:
                    if not p in this_map:
                        this_map[dtype][p] = []
                    this_map[dtype][p].append(obj)

    def make_trees(self, db_id):
        # mask_strings = ["\'links\'", "\'init\'", "\'preferred_name\'", "\'synonyms\'",
        #                 "\'entity\'", "\'points\'", "\'symptoms\'", "\'description\'",
        #                 "\'herb_formulas\'", "\'formulas\'", "\'herbs\'", "\'conditions\'",
        #                 "\'actions\'", "\'composition\'", "\'notes\'", "\'syndromes\'", "\'treats\'",
        #                 "\'latin\'", "\'dosage\'", "\'main\'", "\'additional\'", "\'category\'",
        #                 "\'incompatibility\'", "\'form\'", "\'min\'", "\'max\'", "\'desc\'",
        #                 "\'entrez_ids\'", "\'symbols\'", "\'targets_pred\'", "\'pref_name\'",
        #                 "\'targets_known\'", "\'ingredient\'", "\'cid\'", "\'\',",
        #                 "[]","{:","}, ", "{}", " :", "::", ",", ",:"]
        # mask_strings = [(x, "") for x in mask_strings]
        parse_fields = ['formulas', 'herbs', 'conditions', 'ingrs']
        parse_fields = [x for x in parse_fields if x in self.dbs[db_id].__dict__]
        self.trees[db_id] = {x: dict() for x in parse_fields}
        ser_db = self.dbs[db_id].serialize(add_links=False)
        for tag in self.trees[db_id]:
            # t=SuffixTree({x:reduce(lambda x,y: x.replace(*y), mask_strings, str(y).lower())
            #                 for x,y in ser_db[tag].items()})
            t=SuffixTree({x:str(y).lower() for x,y in ser_db[tag].items()})
            self.trees[db_id][tag] = t

    def tree_find(self, db_id,
                  query, dtype):

        sub_q = split_query(query)
        hits = [set([y for y,z in self.trees[db_id][dtype].find_all(x)]) for x in sub_q]
        hits = set.intersection(*hits)

        if hits:
            return([getattr(self.dbs[db_id], dtype)[x] for x in hits])
        return(None)

    def tree_find_total(self, query):
        all_hits = {}
        for db_id in self.dbs:
            all_hits[db_id] = dict()
            for dtype in self.word_maps[db_id]:
                all_hits[db_id][dtype] = self.tree_find(db_id, query, dtype)
        return(all_hits[db_id][dtype])


    def match_lucky(self, db_id,
                    query, dtype,
                    thr = 50):
        this_map = self.word_maps[db_id][dtype]

        search_res = match_N_w_score(query, this_map, N=1)
        if search_res is None:
            return (None)
        best_hit, score = search_res[0]
        if score < thr:
            return(None)
        return(this_map[best_hit])

    def match_N(self, db_id,
               query, dtype,
               N=5, thr = 50):
        this_map = self.word_maps[db_id][dtype]
        name_field = self.db_configs[db_id]['primary']

        hit_scores = match_N_w_score(query, this_map,  N=2*N)
        if hit_scores is None:
            return(None)
        hit_words = [x for x,y in hit_scores if y>thr]
        if hit_words:
            hit_objs = [this_map[x] for x in hit_words]
            hit_objs = sum(hit_objs, [])
            deduped = {getattr(x, name_field):x for x in hit_objs}
            return(list(deduped.values()))
        return(None)

    def match_total(self, query, N = 5, thr = 75):
        all_hits = {}
        for db_id in self.dbs:
            all_hits[db_id] = dict()
            for dtype in self.word_maps[db_id]:
                all_hits[db_id][dtype] = self.match_N(db_id, query, dtype, N, thr=thr)
        return(all_hits)

    def link_herbs(self, db_id1, db_id2,
                   thr=80, accept_on=96,
                   verbose = False):
        db1, db2 = self.dbs[db_id1], self.dbs[db_id2]
        # herbs1_map = self.word_maps[db_id1]['herbs']
        herbs2_map = self.word_maps[db_id2]['herbs']
        name_field1 = self.db_configs[db_id1]['primary']
        syn_field1 = self.db_configs[db_id1]['secondary']
        name_field2 = self.db_configs[db_id2]['primary']
        # syn_field2 = self.db_configs[db_id2]['secondary']

        # check perfect matches in herb IDs first
        common_names = set(db1.herbs.keys()).intersection(set(db2.herbs.keys()))
        if verbose:
            print(f"Found {len(common_names)} identically named herbs")
        cross_mappings = {x:x for x in common_names}

        # inspect matched in names-synonyms for the rest
        for h, h_obj in db1.herbs.items():
            if h in cross_mappings:
                continue
            all_terms1 = [h] + getattr(h_obj, syn_field1)
            all_terms1 = list(set(all_terms1))

            match_scores = dict()
            for w in all_terms1:
                match_res = match_N_w_score(w, herbs2_map, N=1)
                if match_res is None:
                    continue
                match, curr_score = match_res[0]
                if curr_score < thr:
                    continue

                for match_obj in herbs2_map[match]:
                    match_ID = getattr(match_obj, name_field2)
                    if not match_ID in match_scores:
                        match_scores[match_ID] = 0
                match_scores[match_ID] = max(match_scores[match_ID], curr_score)
                if curr_score > accept_on:
                    break

            if match_scores:
                best_match, best_score = max(match_scores.items(), key=lambda x: x[1])
                cross_mappings[h] = best_match
                if verbose:
                    print(f"Mapped herb {h} → {best_match}")

        if not "herbs" in self.cross_mappings:
            self.cross_mappings['herbs'] = dict()

        self.cross_mappings['herbs'].update({(db_id1, db_id2):cross_mappings})

    def get_equivalent_herbs(self, db_id1, db_id2):
        eqvt_herbs = {(db_id1,db_id2): dict(),
                      (db_id2,db_id1): dict()}
        lesser_key = min(self.cross_mappings['herbs'].keys(), key = lambda x:len(eqvt_herbs[x]))
        bigger_key = lesser_key[::-1]
        lesser_dict, bigger_dict = self.cross_mappings['herbs'][lesser_key], self.cross_mappings['herbs'][bigger_key]
        for k,v in lesser_dict.items():
            if k == bigger_dict[v]:
                eqvt_herbs[lesser_key][k] = v
                eqvt_herbs[bigger_key][v] = k

        if not 'herbs' in self.equivalents:
            self.equivalents['herbs'] = dict()
        self.equivalents['herbs'].update(eqvt_herbs)

    def fill_empty_formulas(self, filled_db_id, source_db_id, verbose = True):
        filled_db, source_db = self.dbs[filled_db_id], self.dbs[source_db_id]
        same_name_flas = set(filled_db.formulas.keys()) & set(source_db.formulas.keys())
        N_filled = 0
        for f_name in same_name_flas:
            rec_fla, don_fla = filled_db.formulas[f_name], source_db.formulas[f_name]
            if rec_fla.get_herbs():
                    continue
            donor_herbs = don_fla.get_herbs()
            translated_herbs = [self.cross_mappings['herbs'][(source_db_id, filled_db_id)].get(x) for x in donor_herbs]
            translated_herbs = [x for x in translated_herbs if not x is None]
            rec_fla.set_herbs(filled_db, translated_herbs)
            N_filled += 1
        if verbose:
            print(f"Filled {N_filled} empty formulas")


    def get_equivalent_flas(self, db_id1, db_id2,
                            leniency = 0):

        eqv_flas ={(db_id1, db_id2):{},
                   (db_id2, db_id1):{}}

        lesser_db = min(self.dbs, key = lambda x:len(self.dbs[x].formulas))
        bigger_db = db_id1*(lesser_db==db_id2) + db_id2*(lesser_db==db_id1)

        for f_name, f_obj in self.dbs[lesser_db].formulas.items():
            these_herbs = f_obj.get_herbs()
            other_herbs = [self.cross_mappings['herbs'][(lesser_db, bigger_db)].get(x) for x in these_herbs]
            if any([x is None for x in other_herbs]):
                continue
            candidates = self.dbs[bigger_db].lookup_similar_herb_formulas(other_herbs, buffer=leniency)
            if candidates:
                if not f_name in eqv_flas[(lesser_db, bigger_db)]:
                    eqv_flas[(lesser_db, bigger_db)][f_name] = []
                eqv_flas[(lesser_db, bigger_db)][f_name].append(candidates[0])
                if not candidates[0] in eqv_flas[(bigger_db, lesser_db)]:
                    eqv_flas[(bigger_db, lesser_db)][candidates[0]] = []
                eqv_flas[(bigger_db, lesser_db)][candidates[0]].append(f_name)

        if not 'formulas' in self.equivalents:
            self.equivalents['formulas'] = dict()
        self.equivalents['formulas'].update(eqv_flas)

    def create_linked_herbs(self):
        raise NotImplementedError()
    
    def create_linked_flas(self):
        pass

class BatmanDragonConnector(DBConnector):

    hf_repo: str = "f-galkin/DragonTCM"
    hf_file: str = "4Nov2024_connector.json.gz"

    @classmethod
    @timer
    def from_huggingface(cls):

        from huggingface_hub import hf_hub_download

        print("📂 Accessing HuggingFace repository...")

        try:
            file_path = hf_hub_download(
                repo_id=cls.hf_repo,
                filename=cls.hf_file,
                repo_type="dataset"
            )

            print("✅ Connector for BATMAN-TCM2 and DragonTCM has been loaded")
            print("Start processing individual databases:")

            return(cls.load(file_path))
        except Exception as e:
            print(f"❌ Error loading connector: {str(e)}")
            raise

    @classmethod
    def load(self, p_in, verbose = True):

        new_con = DBConnector()

        if p_in.lower().endswith(".json"):
            with open(p_in, 'r') as f:
                in_dict = json.load(f)
        elif p_in.lower().endswith(".gz"):
            import gzip
            with gzip.open(p_in, 'rt') as f:
                in_dict = json.load(f)

        for attr in ("p_dbs", "db_configs", "equivalents"):
            setattr(new_con, attr, in_dict[attr])

        if in_dict["cross_mappings"]:
            if verbose:
                print("Adding cross mappings")
            new_con.cross_mappings = {x: {eval(z): a for z, a in y.items()} for x, y in
                                      in_dict['cross_mappings'].items()}
            if verbose:
                print("Done adding cross mappings")

        if in_dict['equivalents']:
            if verbose:
                print("Adding equivalents")
            new_con.equivalents = {x: {eval(z): a for z, a in y.items()} for x, y in
                                   in_dict['equivalents'].items()}
            if verbose:
                print("Done adding equivalents")

        for db_name, p_db in in_dict["p_dbs"].items():
            if verbose:
                print(f"Loading DB: {db_name}")
            db_type = in_dict['db_types'][db_name]
            if "batman.TCMDB" in db_type:
                this_db = batman.TCMDB.from_huggingface()
            elif "annots.TCMAnnotationDB" in db_type:
                this_db = annots.TCMAnnotationDB.from_huggingface()
            else:
                raise TypeError(f"Unknown type of TCM DB: {db_type}")
            new_con.dbs[db_name] = this_db
            if verbose:
                print(f"Uploaded DB: {db_name}")

        if in_dict["word_maps"]:
            if verbose:
                print("Adding word maps")
            new_con.word_maps = {x:
                                   {z:
                                      {b:
                                         [getattr(new_con.dbs[x], z)[d] for d in c] for b,c in a.items()
                                      }
                                   for z,a in y.items()}
                                 for x, y in in_dict["word_maps"].items()}
            if verbose:
                print("Done adding word maps")

        return(new_con)

def main():
    pass

if __name__ == '__main__':
    main()

