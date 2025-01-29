# # agents/tcm_crew.py
import requests
import json
from time import sleep
from typing import Dict, List, Union, Optional, Any
import numpy as np
from enum import Enum

from dataclasses import dataclass, asdict
from ..connector.connector import BatmanDragonConnector, DBConnector

class TCMCategory(Enum):
    CONDITIONS = 'conditions'
    HERBS = 'herbs'
    FORMULAS = 'formulas'
    INGREDIENTS = 'ingrs'

class TCMDatabase(Enum):
    AMERICAN_DRAGON = 'American Dragon'
    BATMAN = 'BATMAN'

@dataclass
class TCMQueryResult:
    """
    Standardized result object for TCM queries
    
    Attributes:
        query (str): Original search term
        database (str): Database that was queried (American Dragon or BATMAN)
        category (str): Type of entity searched (herbs, formulas, etc.)
        results (List[Dict]): Found entries and their data
        status (str): Query status ('success', 'error', etc.)
        error (Optional[str]): Error message if any
        match_type (Optional[str]): Type of match found ('exact', 'fuzzy', 'tree')
    """
    query: str
    database: str
    category: str
    results: List[Dict]
    status: str = 'success'
    error: Optional[str] = None
    match_type: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert result object to dictionary format."""
        return asdict(self)
        
        
class TCMTools:
    """Singleton class to manage TCM database connections and queries"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not TCMTools._initialized:
            self._connector = None
            TCMTools._initialized = True
            
    def initialize(self,
                   from_huggingface: bool = True,
                   p_connector = None):
        """Initialize the database connector"""
#         from connector import BatmanDragonConnector
        from connector.connector import BatmanDragonConnector
        
        if from_huggingface:
            self._connector = BatmanDragonConnector.from_huggingface()
        else:
            self._connector = DBConnector.load(p_connector)
        TCMTools._initialized = True
        
    def _check_initialization(self):
        """Verify connector is initialized"""
        if not self._connector:
            raise RuntimeError("TCM Tools not initialized. Call initialize() first.")
    
    def add_trees(self):
        for db_name in self._connector.dbs:
            self.add_tree(db_name)
    
    def add_tree(self, db_name):
        self._connector.make_trees(db_name)
    

# DB QUERY TOOLS

def query_tcmdb(query: str, db_id: str, category: str, n_hits: int = 1) -> dict:
    """
    Core function to query Traditional Chinese Medicine databases for various entities.
    
    Args:
        query (str): Search term to look up (e.g., herb name, formula name, condition)
        db_id (str): Database identifier, either:
            - 'American Dragon': Traditional TCM knowledge database
            - 'BATMAN': Molecular/compound database with target information
        category (str): Type of entity to search for:
            - 'conditions': Medical conditions and symptoms (only available in American Dragon)
            - 'herbs': Medicinal herbs and materials
            - 'formulas': Traditional herbal formulas
            - 'ingrs': Chemical compounds and ingredients (only available in BATMAN)
        n_hits (int): Maximum number of matches to return (default 5)
    
    Returns:
        List[Dict] if matches found, where each dict contains:
            - init (Dict): Core entity information
                - preferred_name (str): Primary name of entity
                - synonyms (List[str]): Alternative names
                - Additional fields specific to entity type
            - links (Dict): Related entities by category
        
        str if no matches found:
            Error message in format "{query} — Not found in {db_id}:{category}"
    """

    tcm = TCMTools()
    tcm._check_initialization()
    
    try:
        this_db = tcm._connector.dbs[db_id]
        this_subset = getattr(this_db, category)
        
        # Exact match
        if query in this_subset:
            perfect_hit = this_subset.get(query)
            return TCMQueryResult(
                query=query,
                database=db_id,
                category=category,
                results=[perfect_hit.serialize()],
                match_type='exact'
            ).to_dict()
        
        # Fuzzy match
        matches = tcm._connector.match_N(db_id, query, category, N=n_hits, thr=60)
        if matches:
            return TCMQueryResult(
                query=query,
                database=db_id,
                category=category,
                results=[x.serialize() for x in matches],
                match_type='fuzzy'
            ).to_dict()
            
        # Tree search
        matches = tcm._connector.tree_find(db_id, query, category)
        if matches:
            return TCMQueryResult(
                query=query,
                database=db_id,
                category=category,
                results=[x.serialize() for x in matches],
                match_type='tree'
            ).to_dict()
            
        # No matches
        return TCMQueryResult(
            query=query,
            database=db_id,
            category=category,
            results=[],
            status='not_found',
            error=f'No matches found for {query} in {db_id}:{category}'
        ).to_dict()
        
    except Exception as e:
        return TCMQueryResult(
            query=query,
            database=db_id,
            category=category,
            results=[],
            status='error',
            error=str(e)
        ).to_dict()


def find_TCM_condition(query: str) -> dict[str,Any]:
    """
    Look up information about a TCM condition, disease, or symptom pattern.
    
    Args:
        query (str): Name of condition or symptom to search for
        
    Returns:
        List[Dict] if matches found, each dict containing:
            - init:
                - preferred_name (str): Primary name of condition
                - synonyms (List[str]): Alternative names
                - description (List[str]): Detailed condition description
                - symptoms (List[Dict]): Clinical manifestations and treatments
                - herb_formulas (List[str]): Recommended formulas
                - points (List[str]): Related acupuncture points
            - links:
                - herbs (List[str]): Associated medicinal herbs
                - formulas (List[str]): Associated formulas
        
        str: Error message if no matches found
    
    Example:
        >> results = find_TCM_condition("diabetes")
        >> if isinstance(results, list):
        >>     for result in results:
        >>         print(f"Treatment: {result['init']['herb_formulas']}")
        >> else:
        >>     print(results)  # Error message
    """
    return query_tcmdb(query, 'American Dragon', 'conditions')

def find_TCM_herb(query: str, db_id: str = 'American Dragon') -> dict[str,Any]:
    """
    Look up traditional information about a TCM medicinal herb or material.
    
    Args:
        query (str): Name of herb or medicinal material to search for
        db_id (str): Use "American Dragon" (default) for a TCM-focused annotation and use "BATMAN" for molecular targets in the response
    Returns: 
        # For American Dragon
        List[Dict] if matches found, each dict containing: 
            - init:
                - preferred_name (str): Primary name of herb
                - synonyms (List[str]): Alternative names
                - category (List[str]): Herb classifications
                - properties (Dict): Traditional attributes
                    - taste (Dict): Primary and secondary tastes
                    - temperature (Dict): Primary and secondary properties
                    - meridians (Dict): Primary and secondary meridians
                - actions (List[Dict]): Actions and indications
                - contraindications (List[str]): Usage warnings
                - dosage (List[Dict]): Dosage guidelines
            - links:
                - formulas (List[str]): Formulas containing this herb
                - conditions (List[str]): Treatable conditions
        
        # For BATMAN
        
    
    Example:
        >> results = find_TCM_herb("ginseng")
        >> if isinstance(results, list):
        >>     for result in results:
        >>         print(f"Properties: {result['init']['properties']}")
        >>         print(f"Used in: {len(result['links']['formulas'])} formulas")
        >> else:
        >>     print(results)  # Error message
    """
    return query_tcmdb(query, db_id, 'herbs')

def find_TCM_herb_molecular(query: str) -> dict[str,Any]:
    """
    Look up modern molecular and biochemical information about a TCM herb.
    
    Args:
        query (str): Name of herb to search for
        
    Returns:
        List[Dict] if matches found, each dict containing:
            - init:
                - preferred_name (str): Primary name of herb
                - synonyms (List[str]): Alternative names
            - links:
                - ingrs (List[int]): PubChem CIDs of active compounds
                - formulas (List[str]): Formulas containing this herb
        
        str: Error message if no matches found
        
    Note:
        Results come from BATMAN-TCM2 database which focuses on molecular 
        mechanisms and compound-target interactions rather than traditional usage.
    
    Example:
        >> results = find_TCM_herb_molecular("ginseng")
        >> if isinstance(results, list):
        >>     for result in results:
        >>         print(f"Contains {len(result['links']['ingrs'])} compounds")
        >> else:
        >>     print(results)  # Error message
    """
    return query_tcmdb(query, 'BATMAN', 'herbs')

def find_TCM_formula(query: str) -> dict[str,Any]:
    """
    Look up traditional information about a TCM herbal formula.
    
    Args:
        query (str): Name of formula to search for
        
    Returns:
        List[Dict] if matches found, each dict containing:
            - init:
                - preferred_name (str): Primary name of formula
                - synonyms (List[str]): Alternative names
                - composition (Dict[str, Dict]): Herbs and their properties
                - treats (List[str]): Target conditions
                - syndromes (List[str]): TCM syndromes addressed
                - actions (List[str]): Therapeutic actions
                - contraindications (List[str]): Usage warnings
                - notes (List[str]): Additional information
            - links:
                - herbs (List[str]): Component herbs
                - conditions (List[str]): Treatable conditions
        
        str: Error message if no matches found
    
    Example:
        >> results = find_TCM_formula("liu wei di huang wan")
        >> if isinstance(results, list):
        >>     for result in results:
        >>         print(f"Contains herbs: {result['links']['herbs']}")
        >> else:
        >>     print(results)  # Error message
    """
    return query_tcmdb(query, 'American Dragon', 'formulas')

def find_TCM_formula_molecular(query: str) -> dict[str,Any]:
    """
    Look up molecular and biochemical information about a TCM formula.
    
    Args:
        query (str): Name of formula to search for
        
    Returns:
        List[Dict] if matches found, each dict containing:
            - init:
                - preferred_name (str): Primary name of formula
                - synonyms (List[str]): Alternative names
            - links:
                - herbs (List[str]): Component herbs (max 1000)
                - ingrs (List[int]): PubChem CIDs of compounds (max 1000)
                - formulas (List[str]): Related formulas (max 1000)
        
        str: Error message if no matches found
        
    Note:
        Results are limited to 1000 related entities per category (ingrs/herbs/formulas)
        to manage response size. Results come from BATMAN-TCM2 database focusing on
        molecular mechanisms rather than traditional usage.
    
    Example:
        >> results = find_TCM_formula_molecular("liu wei di huang wan")
        >> if isinstance(results, list):
        >>     for result in results:
        >>         print(f"Active compounds: {len(result['links']['ingrs'])}")
        >> else:
        >>     print(results)  # Error message
    """
    result = query_tcmdb(query, 'BATMAN', 'formulas')
    
    if result['results']:
        # Truncate large result sets
        truncated_results = []
        for hit in result['results']:
            truncated_hit = hit.copy()
            for link_type in ('ingrs', 'herbs', 'formulas'):
                if link_type in hit['links']:
                    truncated_hit['links'][link_type] = hit['links'][link_type][:1000]
            truncated_results.append(truncated_hit)
        result['results'] = truncated_results
        
    return result

def find_TCM_compound(query: str, max_links: int = 5000) -> dict[str,Any]:
    """
    Look up information about a compound's molecular targets in TCM context.
    
    Args:
        query (str): A name or an identifier of compound to search for
        
    Returns:
        List[Dict] if matches found, each dict containing:
            - init:
                - cid (int): PubChem Compound ID
                - pref_name (str): Primary compound name
                - synonyms (List[str]): Alternative names
                - targets_known (Dict): Experimentally validated targets
                    - symbols (List[str]): Gene symbols
                    - entrez_ids (List[int]): NCBI gene IDs
                - targets_pred (Dict): Predicted protein targets
                    - symbols (List[str]): Gene symbols
                    - entrez_ids (List[int]): NCBI gene IDs
            - links:
                - herbs (List[str]): Source herbs (max 1000)
                - formulas (List[str]): Related formulas (max 1000)
        
        str: Error message if no matches found
    
    Note:
        Results are limited to 1000 related entities per category to manage size.
        Targets are divided into 'known' (experimentally validated) and 
        'predicted' (computational predictions).
    
    Example:
        >> results = find_TCM_compound("quercetin")
        >> print(results)
        [{'query': 2244,
         'database': 'BATMAN',
         'category': 'ingrs',
         'results': [{'init': {
            'cid': 2244,
            'targets_known': {'symbols': ['PTGS1','PRKAG1','CCND1'],
                              'entrez_ids' : [5562, 5563, 4846]},
            'targets_pred' : {'symbols': ['AKR1C4', 'AKR1C2', 'RPS6KA6'],
                              'entrez_ids': [1109, 1646, 27330, 6195]},
             'pref_name': 'aspirin',
             'desc': '',
             'synonyms': ['2-acetyloxybenzoic acid'],
             'entity': 'ingredient'
                               },
                      'links': {
             'ingrs': [],
             'herbs': ['FENG FANG'],
             'formulas': ['BA XIAN GAO~4', 'BAI DU SAN~9']
                                }
                     }],
         'status': 'success',
         'error': None,
         'match_type': 'exact'}]
    """

    result = query_tcmdb(query, 'BATMAN', 'ingrs')
    
    if result['results']:
        # Truncate large result sets
        truncated_results = []
        for hit in result['results']:
            truncated_hit = hit.copy()
            for link_type in ('ingrs', 'herbs', 'formulas'):
                if link_type in hit['links']:
                    truncated_hit['links'][link_type] = hit['links'][link_type][:max_links]
            truncated_results.append(truncated_hit)
        result['results'] = truncated_results

    return result


# CHEMISTRY TOOLS

def get_pubchem_cid(compound_name: str, max_retries: int = 3) -> Optional[str]:
    """
    Get PubChem CID for a compound name.
    
    Args:
        compound_name (str): Name of the compound to look up
        max_retries (int): Maximum number of retry attempts on rate limit
        
    Returns:
        Optional[str]: PubChem CID if found, None otherwise
        
    Example:
        >> get_pubchem_cid("aspirin")
        '2244'
        >> get_pubchem_cid("not_a_real_compound")
        None
    """
    
    CID_ENDPOINT = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/TXT"
    
    # Format query
    query = compound_name.strip().lower().replace("-", " ")
    if query.endswith("hcl") and not query.endswith(" hcl"):
        query = query.removesuffix("hcl") + " hcl"
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(CID_ENDPOINT.format(query))
            
            if response.status_code == 200:
                cids = response.text.strip().split("\n")
                return cids[0] if cids else None
                
            elif response.status_code == 404:
                return None
                
            elif response.status_code == 503:  # Rate limited
                retry_count += 1
                sleep(1)
                continue
                
            else:
                return None
                
        except requests.RequestException:
            retry_count += 1
            sleep(1)
            continue
            
    return None

def get_chembl_from_cid(pubchem_cid: int|str, max_retries: int = 3) -> Optional[str]:
    """
    Get ChEMBL ID for a PubChem CID.
    
    Args:
        pubchem_cid (str|int): PubChem CID to look up
        max_retries (int): Maximum number of retry attempts on rate limit
        
    Returns:
        Optional[str]: ChEMBL ID if found, None otherwise
        
    Example:
        >> get_chembl_from_cid("2244")  # Aspirin
        'CHEMBL25'
        >> get_chembl_from_cid("invalid")
        None
    """
    
    if not pubchem_cid.isnumeric():
        raise ValueError(f"The provided CID is not numeric: {pubchem_cid}")
        
    SYNONYM_ENDPOINT = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/synonyms/TXT"
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(SYNONYM_ENDPOINT.format(pubchem_cid))
            
            if response.status_code == 200:
                synonyms = response.text.strip().split("\n")
                chembl_ids = [x for x in synonyms if x.startswith("CHEMBL")]
                return chembl_ids[0] if chembl_ids else None
                
            elif response.status_code == 404:
                return None
                
            elif response.status_code == 503:  # Rate limited
                retry_count += 1
                sleep(1)
                continue
                
            else:
                return None
                
        except requests.RequestException:
            retry_count += 1
            sleep(1)
            continue
            
    return None

def get_description_from_cid(pubchem_cid: int|str, max_retries: int = 3)->Optional[list[str]]:
    
    """
    Get a brief description for a compound specified as a numeric PubChem CID.
    Might contain several descriptions from different sources.
    
    Args:
        pubchem_cid (str|int): PubChem CID to look up
        max_retries (int): Maximum number of retry attempts on rate limit
        
    Returns:
        Optional[str]: A JSON foramtted response containing compound descriptions if found, None otherwise
        
    Example:
        >> get_chembl_from_cid("1983")  # Acetaminophen
        ['Acetaminophen', 'Paracetamol is a member of the class of phenols that is 4-aminophenol in which one of the hydrogens attached to the amino group has been replaced by an acetyl group...']

        >> get_chembl_from_cid("invalid")
        None
    """
    pubchem_cid = str(pubchem_cid)
    if not pubchem_cid.isnumeric():
        raise ValueError(f"The provided CID is not numeric: {pubchem_cid}")
    
    DESC_ENDPOINT = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/description/JSON"
        
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(DESC_ENDPOINT.format(pubchem_cid))
            
            if response.status_code == 200:
                descs = json.loads(response.text)
                if "Fault" in descs:
                    return None
                # first entry is just a title+CID
                descs = [descs[0]['Title']]
                if not descs:
                    return None
                descs += [x['Description'] for x in descs['InformationList']['Information'][1:] if str(x['CID']) == pubchem_cid]
                return(descs)
                
            elif response.status_code == 404:
                return None
                
            elif response.status_code == 503:  # Rate limited
                retry_count += 1
                sleep(1)
                continue
                
            else:
                return None
                
        except Exception as e:
            retry_count += 1
            sleep(1)
            continue
            
    return None
 
def get_description_from_cid_batch(pubchem_cid_batch: list[int | str], max_retries: int = 3, batch_size: int = 100) -> \
Optional[dict[int, list[str]]]:
    """
    Get brief descriptions for a batch of compounds specified as numeric PubChem CIDs.
    Processes CIDs in batches to avoid overloading the API.

    Args:
        pubchem_cid_batch (list[int|str]): List of PubChem CIDs to look up
        max_retries (int): Maximum number of retry attempts on rate limit
        batch_size (int): Maximum number of CIDs to request in a single API call

    Returns:
        Optional[dict[int,list[str]]]: Dictionary mapping CIDs to their descriptions if found, None on failure

    Example:
        >> get_description_from_cid_batch([1983, 2244])  # Acetaminophen, Aspirin
        {1983: ['Acetaminophen', 'Paracetamol is a member of the class of phenols...'],
         2244: ['Aspirin', 'Acetylsalicylic acid is a member of the class of benzoic acids...']}

        >> get_description_from_cid_batch(['invalid'])
        None
    """
    # Validate input
    try:
        cids = [str(cid) for cid in pubchem_cid_batch]
        if not all(cid.isnumeric() for cid in cids):
            invalid_cids = [cid for cid in cids if not cid.isnumeric()]
            raise ValueError(f"Non-numeric CIDs found: {invalid_cids}")
    except (TypeError, ValueError) as e:
        print(f"Input validation failed: {e}")
        return None

    DESC_ENDPOINT = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/description/JSON"
    all_descriptions = {}

    # Process CIDs in batches
    for i in range(0, len(cids), batch_size):
        batch = cids[i:i + batch_size]
        cid_string = ",".join(batch)
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.get(DESC_ENDPOINT.format(cid_string))
                if response.status_code == 200:
                    data = json.loads(response.text)
                    if "Fault" in data:
                        print(f"API fault for batch starting at index {i}")
                        break

                    # Process the response
                    info_list = data.get('InformationList', {}).get('Information', [])

                    # Group descriptions by CID
                    for info in info_list:
                        cid = str(info.get('CID'))
                        if cid not in all_descriptions:
                            all_descriptions[cid] = []

                        # Add title if it's the first entry
                        if 'Title' in info:
                            all_descriptions[cid].append(info['Title'])
                        # Add description if present
                        if 'Description' in info:
                            all_descriptions[cid].append(info['Description'])

                    break  # Successfully processed this batch

                elif response.status_code == 404:
                    print(f"No descriptions found for batch starting at index {i}")
                    break

                elif response.status_code == 503:  # Rate limited
                    retry_count += 1
                    sleep(1)
                    continue

                else:
                    print(f"Unexpected status code: {response.status_code} for batch starting at index {i}")
                    break

            except Exception as e:
                print(f"Error during API request for batch starting at index {i}: {str(e)}")
                retry_count += 1
                sleep(1)
                continue

            if retry_count >= max_retries:
                print(f"Max retries ({max_retries}) exceeded for batch starting at index {i}")

    # Return None if no descriptions were found
    if not all_descriptions:
        return None

    # Convert string CIDs back to integers in result
    return {int(k): v for k, v in all_descriptions.items()}

def get_compound_targets(pubchem_cid: int|str, tg_type: str = "both") -> list[str]:
    '''
    Get the gene symbols of protein targets affected by a compound.
    
    Args:
        pubchem_cid (str|int): PubChem CID to look up in the BATMAN-TCM DB
        tg_type (str): What type of protein targets should be considered when looking for fitting compounds: 'known' (lower nuber of hits), 'predicted', or 'both' (higher number of compunds) Default: 'both'
        
    Returns:
        list[str]: A list with the gene symbols of a compound's protein targets affected at the selected level of evidence
        
    Example:
        >> get_compound_targets(9064, tg_type = 'known')
        ['KLF7','DHFR','FASN','PPARG']
        >> get_compound_targets('INVALID', tg_type = 'known')
        []
        
    '''
    if not str(pubchem_cid).isnumeric():
        raise ValueError(f"Provided Pubchem CID is not numeric: {pubchem_cid}")
    pubchem_cid = int(pubchem_cid)
    
    tcm = TCMTools()
    tcm._check_initialization()
    this_db = tcm._connector.dbs['BATMAN']
    
    result = this_db.ingrs.get(pubchem_cid, [])
    if not result:
        return(result)
    match tg_type:
        case "both":
            return(result.targets['predicted']['symbols'] + result.targets['known']['symbols'])
        case "known":
            return(result.targets['known']['symbols'])
        case 'predicted': 
            return(result.targets['predicted']['symbols'])
        case _:
            raise ValueError()
    

#HERB PICKING TOOLS

def pick_cpds_by_targets(gene_list: list,
                         tg_type: str = "both",
                         thr: float=0.01,
                         **kwargs # blacklist + cpd_subset
                         ) -> dict[int,dict[str,Any]]:
    """
    Find the compounds that interact with the provided genes, according to the BATMAN-TCM database.
    
    Args:
        gene_list (list[str]): A list of gene symbols that should to be affected by the selected compounds
        tg_type (str): What type of protein targets should be considered when looking for fitting compounds: 'known' (lower nuber of hits), 'predicted', or 'both' (higher number of compunds) Default: 'both'
        thr (float): P-value threshold (post multiple comparison) for determining if an overlap between the provided gene list and a compound's protein targets is significant. Default: 0.01
        
    Returns:
        dict[int,dict[str,Any]]: A dictionary with all compound significantly interacting with the provided genes. Keys are CID references. Values are a dict with a common name ('name') of a compound and overlap statistics ('Pv' and 'corrPv' for significance, 'targets_hit' for genes that overlapped, 'total_targets' for all targets a compounds affects)
        
    Example:
        >> pick_cpds_by_targets(['EGLN1', 'AKT3', 'ERK'...])
        {2244: {'name' : "aspirin",
                 'Pv' 0.004,
                 'corrPv': 0.0087,
                 'targets_hit':10,
                 'total_targets':230}
          }
        >> get_chembl_from_cid("invalid")
        {-1: {'name' : "NA",
              'Pv' : 1.,
              'corrPv' : 1.,
              'targets_hit' : 0,
              'total_targets' : 0}
          }
    """
    
    tcm = TCMTools()
    tcm._check_initialization()

    blank_result = {-1: dict(name = "No compounds found",
                             Pv = 1.,
                             corrPv= 1.,
                             targets_hit=0,
                             total_targets=0)
                      }
    try:
        this_db = tcm._connector.dbs['BATMAN']
      
        picked_cpds = this_db.find_enriched_cpds(
                              gene_list, # genes to be targeted by natural compounds
                              tg_type=tg_type, # consider both known and predicted molecular targets
                              thr=thr, 
                              **kwargs # blacklist + cpd_subset
                                                )
    except Exception as e:
        print(e)
        picked_cpds = blank_result
    if not picked_cpds:
        picked_cpds = blank_result
                      
    return picked_cpds
    

def pick_herbs_by_targets(gene_list: set,
                          tg_type: str = 'known',
                          top_herbs: float = .01,
                          blacklist: Optional[set] = None) -> dict[str, int]:
                              
    '''
    Find the herbs that interact with the provided gene list. The strength of an interraction is based on the total number of occurrences for all genes gene among all ingredients composing a herb. Since herbs can contain many ingredients interacting with the same gene, the total hot count can potentially be larger than the number of genes.
    
    Args:
        gene_list (list[str]): A list of gene symbols that should to be affected by the selected compounds
        tg_type (str): What type of protein targets should be considered when looking for fitting herbs: 'known' (lower nuber of hits), 'predicted', or 'both' (higher number of hits). Default: 'known'
        top_herbs (float): The number of target counts (expressed as a qunatile) to be used as a cutoff to present the top fitting herbs. Use 1. to get the counts for all herbs stored in the database. Default:0.01
        blacklist (Optional[set[str]]): The herbs that need to be excluded from the output
    Returns:
        dict[str, int]: A dictionary with herb primary names as keys and the total accumulated number of hit tartgets as values.
    Example:
        >> pick_herbs_by_targets(['EGLN1', 'AKT3', 'ERK'...])
        {'MA HUANG': 97, 'GUO GO': 95, ...}
    '''
    tcm = TCMTools()
    tcm._check_initialization()
    
    blank_result = {"No herbs found":0}
    
    try:
        result = tcm._connector.dbs['BATMAN'].select_herbs_for_targets(glist = gene_list,
                                                                       tg_type=tg_type,
                                                                       blacklist = blacklist)
        # truncate the results
        q_cutoff = np.quantile([x for x in result.values()], top_herbs)
        result = {x:y for x,y in result.items() if y>q_cutoff}
        
    except Exception as e:
        result = blank_result
    
    if not result:
        result = blank_result
    
    return(result)
    
  
def pick_herbs_by_cids(cids: list[int],
                       min_cids: int = 2,
                       N_top: int = 10,
                       blacklist: Optional[set] = None) -> dict[str, int]:
    '''
    Find the herbs that contain the largest subset of a provided set of compounds.
    
    Args:
        cids (list[int]): 
        blacklist (Optional[set[int]]): The herbs that need to be excluded from the output
    Returns:
        dict[str, int]: Keys are the primary herb names and the values are the number of compounds included in their composition from the provided list of compounds.
    Example:
        >> pick_herbs_by_cids([2244, 1234, 567], min_cids = 2, N_top=1)
        {"LU DONG":3}
    '''
    
    tcm = TCMTools()
    tcm._check_initialization()
    
    blank_result = {"No herbs found":0}
    
    try:
        result = tcm._connector.dbs['BATMAN'].select_N_herbs_by_cids(cids = cids,
                                                                     min_cids = min_cids,
                                                                     N_top=N_top,
                                                                     blacklist = blacklist)
    except Exception as e:
        result = blank_result
    
    if not result:
        result = blank_result
    
    return(result)


# PROPERTY SCORING
class TCMPropertyScorer:
    """Class for normalizing and scoring TCM herb properties"""

    # Standard property mappings
    STANDARD_TASTES = {
        'Bitter': {'Bitter', 'Slightly Bitter'},
        'Sweet': {'Sweet', 'Slightly Sweet', 'Slightly sweet'},
        'Acrid': {'Acrid', 'Slightly Acrid', 'Pungent', 'Slightly Pungent', 'acrid'},
        'Salty': {'Salty', 'Slightly Salty'},
        'Sour': {'Sour', 'Slightly Sour'},
        'Astringent': {'Astringent', 'Slightly Astringent'},
        'Aromatic': {'Aromatic', 'Fragrant'},
        'Bland': {'Bland', 'Mild'},
        'Unknown': {'No Information', 'Serrt', 'Harsh', 'Hot'}
    }

    STANDARD_TEMPERATURES = {
        'Hot': {'Hot', 'Very Hot', 'Extremely Hot', 'Very Warm', 'Warm (dry-fried)'},
        'Warm': {'Warm', 'Slightly Warm', 'Slightly warm', 'Lukewarm', 'Leukewarm'},
        'Neutral': {'Neutral', 'Neutral (raw)'},
        'Cool': {'Cool', 'Slightly Cool', 'Mildly Cool'},
        'Cold': {'Cold', 'Very Cold', 'Extremely Cold', 'Slightly Cold', 'Mildly Cold'},
        'Unknown': {
            'No Information', 'Depending on the preparation', 'Aromatic',
            'Slippery', 'Salty', 'Slightly Bitter', 'Non-toxic', 'Toxic',
            'Very Toxic', 'Extremely Toxic', 'Slightly Toxic', 'Slightly toxic',
            'Mildly Toxic', 'Non toxic'
        }
    }

    STANDARD_MERIDIANS = {
        # 12 Regular Meridians
        'Lung': {'Lung', 'Lungs'},
        'Large Intestine': {'Large Intestine'},
        'Stomach': {'Stomach'},
        'Spleen': {'Spleen'},
        'Heart': {'Heart'},
        'Small Intestine': {'Small Intestine', 'Small Intestines'},
        'Bladder': {'Bladder', 'Urinary Bladder'},
        'Kidney': {'Kidney', 'Kidneys'},
        'Pericardium': {'Pericardium'},
        'Triple Burner': {'Triple Burner', 'San Jiao'},
        'Gallbladder': {'Gallbladder', 'Gall Bladder'},
        'Liver': {'Liver'},

        # 8 Extraordinary Meridians
        'Du': {'Du', 'DU'},
        'Ren': {'Ren'},
        'Chong': {'Chong'},
        'Dai': {'Dai'},
        'Yang Qiao': {'Yang Qiao'},
        'Yin Qiao': {'Yin Qiao'},
        'Yang Wei': {'Yang Wei'},
        'Yin Wei': {'Yin Wei'},

        # Special cases
        'All Meridians': {
            'All twelve channels', 'All twelve Channels', 'All 12 Meridians',
            'All 12 Channels', 'All 5 Zang'
        },
        'Unknown': {'No Information', '?', 'Ling'}
    }

    def __init__(self):
        """Initialize TCM Property Scorer"""
        self._tcm = None

    @property
    def tcm(self):
        """Lazy loading of TCM tools"""
        if self._tcm is None:
            self._tcm = TCMTools()
            self._tcm._check_initialization()
        return self._tcm

    def normalize_property(self, value: str, standard_map: Dict[str, set[str]]) -> str:
        """Map variant property values to standard form"""
        for standard, variants in standard_map.items():
            if value in variants:
                return standard
        return 'Unknown'

    def normalize_herb_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize herb properties to standard forms"""
        if not properties:
            return {
                'taste': {'primary': ['Unknown'], 'secondary': []},
                'temperature': {'primary': ['Unknown'], 'secondary': []},
                'meridians': {'primary': ['Unknown'], 'secondary': []}
            }

        normalized = {}

        # Normalize tastes
        if 'taste' in properties:
            normalized['taste'] = {
                'primary': [self.normalize_property(t, self.STANDARD_TASTES)
                            for t in properties['taste'].get('primary', [])],
                'secondary': [self.normalize_property(t, self.STANDARD_TASTES)
                              for t in properties['taste'].get('secondary', [])]
            }
        else:
            normalized['taste'] = {'primary': ['Unknown'], 'secondary': []}

        # Normalize temperature
        if 'temperature' in properties:
            normalized['temperature'] = {
                'primary': [self.normalize_property(t, self.STANDARD_TEMPERATURES)
                            for t in properties['temperature'].get('primary', [])],
                'secondary': [self.normalize_property(t, self.STANDARD_TEMPERATURES)
                              for t in properties['temperature'].get('secondary', [])]
            }
        else:
            normalized['temperature'] = {'primary': ['Unknown'], 'secondary': []}

        # Normalize meridians
        if 'meridians' in properties:
            normalized['meridians'] = {
                'primary': [self.normalize_property(m, self.STANDARD_MERIDIANS)
                            for m in properties['meridians'].get('primary', [])],
                'secondary': [self.normalize_property(m, self.STANDARD_MERIDIANS)
                              for m in properties['meridians'].get('secondary', [])]
            }
        else:
            normalized['meridians'] = {'primary': ['Unknown'], 'secondary': []}

        return normalized

    def score_herb_properties(self,
                              herb: "HerbAnnotation",
                              target_weights: Dict[str, Dict[str, float]],
                              verbose: bool = False) -> float:
        """
        Score a herb based on how well its properties match target weights.
        Unknown or missing properties contribute their 'Unknown' weight to the score.
        """
        scores = []

        if verbose:
            print(f"\nScoring herb: {herb.preferred_name}")
            print("\nNormalized properties:")
            print(json.dumps(self.normalize_herb_properties(herb.properties), indent=2))

        # Normalize herb properties
        normalized = self.normalize_herb_properties(herb.properties)

        # Score meridians
        if target_weights.get('meridian_weights'):
            meridians = normalized['meridians']['primary'] + normalized['meridians']['secondary']
            # If no meridians specified, count as one Unknown
            meridians = meridians if meridians else ['Unknown']
            meridian_score = sum(target_weights['meridian_weights'].get(m, 0) for m in meridians) / len(meridians)
            scores.append(meridian_score)

            if verbose:
                print("\nMeridian scoring:")
                print(f"Meridians: {meridians}")
                print(f"Score: {meridian_score:.3f}")

        # Score tastes
        if target_weights.get('taste_weights'):
            tastes = normalized['taste']['primary'] + normalized['taste']['secondary']
            # If no tastes specified, count as one Unknown
            tastes = tastes if tastes else ['Unknown']
            taste_score = sum(target_weights['taste_weights'].get(t, 0) for t in tastes) / len(tastes)
            scores.append(taste_score)

            if verbose:
                print("\nTaste scoring:")
                print(f"Tastes: {tastes}")
                print(f"Score: {taste_score:.3f}")

        # Score temperature
        if target_weights.get('temperature_weights'):
            # Use primary temperature or Unknown if missing
            temp = normalized['temperature']['primary'][0] if normalized['temperature']['primary'] else 'Unknown'
            temp_score = target_weights['temperature_weights'].get(temp, 0)
            scores.append(temp_score)

            if verbose:
                print("\nTemperature scoring:")
                print(f"Temperature: {temp}")
                print(f"Score: {temp_score:.3f}")

        final_score = sum(scores) / len(scores) if scores else 0.0

        if verbose:
            print("\nScore breakdown:")
            print(f"Individual scores: {[f'{s:.3f}' for s in scores]}")
            print(f"Final score (average): {final_score:.3f}")

        return round(final_score, 3)
    def find_best_shi_herbs(self,
                            target_weights: Dict[str, Dict[str, float]],
                            blacklist: Optional[set[str]] = None,
                            N_top: int = 10,
                            min_score: float = 0.3) -> List[tuple[str, float]]:
        """Find the top N Shi herbs based on property weights"""
        this_db = self.tcm._connector.dbs['American Dragon']
        blacklist = blacklist or set()
        scored_herbs = []

        for herb_name, herb_obj in this_db.herbs.items():
            if herb_name in blacklist:
                continue

            score = self.score_herb_properties(herb_obj, target_weights)

            if score >= min_score:
                scored_herbs.append((herb_name, score))

        scored_herbs.sort(key=lambda x: x[1], reverse=True)

        N_return = min(N_top, len(scored_herbs))
        if N_return > 0:
            return scored_herbs[:N_return]
        else:
            return [("No suitable herbs found", 0.0)]

def get_herbal_compounds(herb_list: list[str]) -> set[int]:
    '''
    Extract all compounds contained in a list of herbs
    Args:
        herb_list:

    Returns:

    '''

    tcm = TCMTools()
    tcm._check_initialization()

    batman_db = tcm._connector.dbs['BATMAN']
    present_herbs = [x for x in herb_list if x in batman_db.herbs]
    if not present_herbs:
        return dict()

    all_cids = set.union(*[{x.cid for x in batman_db.herbs[y].ingrs} for y in present_herbs])
    return(all_cids)

def get_herbal_targets(herb_list: list[str],
                       top_N: int=100,
                       tg_type: str = 'both',
                       gene_names: str = "symbols") -> dict[str | int, int]:
    '''
    Count the occurrences of molecular targets across a list of herbs using BATMAN-TCM database.

    Args:
        herb_list: List of herb names to analyze
        top_N: Maximum number of top targets to return, ordered by frequency
        tg_type: Type of targets to count ('both', 'known', or 'predicted')
        gene_names: Type of gene identifiers to use ('symbols' or 'entrez_ids')

    Returns:
        Dictionary mapping top N targets to their occurrence counts, sorted by count in descending order

    Example:
        >> targets = get_herbal_targets(['DANG SHEN', 'GUI ZHI'], top_N=5)
        >> print(targets)
        {'AKT1': 12, 'MAPK1': 10, 'TNF': 8, 'IL6': 7, 'VEGFA': 6}
    '''

    # Input validation
    if not isinstance(herb_list, list) or not herb_list:
        return {}

    if not isinstance(top_N, int) or top_N < 1:
        raise ValueError("top_N must be a positive integer")

    if tg_type not in ('both', 'known', 'predicted'):
        raise ValueError("tg_type must be one of: 'both', 'known', 'predicted'")

    if gene_names not in ('symbols', 'entrez_ids'):
        raise ValueError("gene_names must be one of: 'symbols', 'entrez_ids'")

    tcm = TCMTools()
    tcm._check_initialization()
    batman_db = tcm._connector.dbs['BATMAN']

    present_herbs = [x for x in herb_list if x in batman_db.herbs]
    if not present_herbs:
        return dict()

    tg_counts = dict()
    try:
        for herb_name in present_herbs:
            herb_obj = batman_db.herbs[herb_name]
            herb_counts = herb_obj.get_target_counts(tg_type, gene_names)
            for target, count in herb_counts.items():
                tg_counts[target] = tg_counts.get(target, 0) + count

        sorted_targets = dict(
            sorted(
                tg_counts.items(),
                key=lambda x: (-x[1], x[0])  # Sort by count desc, then name asc
            )[:top_N]
        )

        return(sorted_targets)

    except Exception as e:
        print(f"Error processing herbal targets: {str(e)}")
        return dict()


def count_targets_in_compounds(cids: list[int],
                               gene_list: list[str],
                               tg_type: str = "both",
                               gene_names: str = "symbols",
                               N_top: Optional[int] = 100) -> dict[str, int]:
    '''
    Count how many genes from a provided list each compound affects.

    Args:
        cids: List of PubChem compound IDs to analyze
        gene_list: List of gene identifiers to check against
        tg_type: Type of targets to count ('both', 'known', or 'predicted')
        gene_names: Type of gene identifiers to use ('symbols' or 'entrez_ids')
        N_top: Optional number of top compounds to return, sorted by target count

    Returns:
        Dictionary mapping compound IDs to their target count within the gene list,
        optionally limited to top N compounds by count. For tg_type='both', counts
        unique targets across both known and predicted sets.

    Example:
        >> genes = ['AKT1', 'MAPK1', 'TNF']
        >> count_targets_in_compounds([2244, 5280343], genes, N_top=1)
        {2244: 2}
    '''
    if tg_type not in ('both', 'known', 'predicted'):
        raise ValueError("tg_type must be one of: 'both', 'known', 'predicted'")

    if gene_names not in ('symbols', 'entrez_ids'):
        raise ValueError("gene_names must be one of: 'symbols', 'entrez_ids'")

    if N_top is not None and not (isinstance(N_top, int) and N_top > 0):
        raise ValueError("N_top must be a positive integer")

    tcm = TCMTools()
    tcm._check_initialization()
    batman_db = tcm._connector.dbs['BATMAN']
    gene_set = set(gene_list)

    if tg_type in ('known', 'predicted'):
        target_count_dict = {
            cid: len(set(batman_db.ingrs[cid].targets[tg_type][gene_names]) & gene_set)
            for cid in cids if cid in batman_db.ingrs
        }
    else:  # tg_type == "both"
        target_count_dict = {
            cid: len(
                (set(batman_db.ingrs[cid].targets['known'][gene_names]) |
                 set(batman_db.ingrs[cid].targets['predicted'][gene_names])) & gene_set
            )
            for cid in cids if cid in batman_db.ingrs
        }

    # Sort by count (descending) and limit to top N if specified
    sorted_dict = dict(
        sorted(
            target_count_dict.items(),
            key=lambda x: (-x[1], x[0])  # Sort by count desc, then CID asc
        )[:N_top if N_top is not None else len(target_count_dict)]
    )

    return sorted_dict