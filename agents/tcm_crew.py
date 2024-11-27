import requests
from time import sleep
from typing import Dict, List, Union, Optional
from enum import Enum

from dataclasses import dataclass, asdict
from connector.connector import BatmanDragonConnector, DBConnector

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
            self._connector.make_trees(db_name)



def query_tcmdb(query: str, db_id: str, category: str, n_hits: int = 1) -> Union[List[Dict], str]:
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



def find_TCM_condition(query: str) -> Union[List[Dict], str]:
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
        >>> results = find_TCM_condition("diabetes")
        >>> if isinstance(results, list):
        >>>     for result in results:
        >>>         print(f"Treatment: {result['init']['herb_formulas']}")
        >>> else:
        >>>     print(results)  # Error message
    """
    return query_tcmdb(query, 'American Dragon', 'conditions')

def find_TCM_herb(query: str) -> Union[List[Dict], str]:
    """
    Look up traditional information about a TCM medicinal herb or material.
    
    Args:
        query (str): Name of herb or medicinal material to search for
        
    Returns:
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
        
        str: Error message if no matches found
    
    Example:
        >>> results = find_TCM_herb("ginseng")
        >>> if isinstance(results, list):
        >>>     for result in results:
        >>>         print(f"Properties: {result['init']['properties']}")
        >>>         print(f"Used in: {len(result['links']['formulas'])} formulas")
        >>> else:
        >>>     print(results)  # Error message
    """
    return query_tcmdb(query, 'American Dragon', 'herbs')

def find_TCM_herb_molecular(query: str) -> Union[List[Dict], str]:
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
        >>> results = find_TCM_herb_molecular("ginseng")
        >>> if isinstance(results, list):
        >>>     for result in results:
        >>>         print(f"Contains {len(result['links']['ingrs'])} compounds")
        >>> else:
        >>>     print(results)  # Error message
    """
    return query_tcmdb(query, 'BATMAN', 'herbs')

def find_TCM_formula(query: str) -> Union[List[Dict], str]:
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
        >>> results = find_TCM_formula("liu wei di huang wan")
        >>> if isinstance(results, list):
        >>>     for result in results:
        >>>         print(f"Contains herbs: {result['links']['herbs']}")
        >>> else:
        >>>     print(results)  # Error message
    """
    return query_tcmdb(query, 'American Dragon', 'formulas')

def find_TCM_formula_molecular(query: str) -> Union[List[Dict], str]:
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
        >>> results = find_TCM_formula_molecular("liu wei di huang wan")
        >>> if isinstance(results, list):
        >>>     for result in results:
        >>>         print(f"Active compounds: {len(result['links']['ingrs'])}")
        >>> else:
        >>>     print(results)  # Error message
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

def find_TCM_compound(query: str) -> Union[List[Dict], str]:
    """
    Look up information about a compound's molecular targets in TCM context.
    
    Args:
        query (str): Name or identifier of compound to search for
        
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
        >>> results = find_TCM_compound("quercetin")
        >>> if isinstance(results, list):
        >>>     for result in results:
        >>>         print(f"Known targets: {result['init']['targets_known']}")
        >>> else:
        >>>     print(results)  # Error message
    """
    result = query_tcmdb(query, 'BATMAN', 'ingrs')
    
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



def get_pubchem_cid(compound_name: str, max_retries: int = 3) -> Optional[str]:
    """
    Get PubChem CID for a compound name.
    
    Args:
        compound_name (str): Name of the compound to look up
        max_retries (int): Maximum number of retry attempts on rate limit
        
    Returns:
        Optional[str]: PubChem CID if found, None otherwise
        
    Example:
        >>> get_pubchem_cid("aspirin")
        '2244'
        >>> get_pubchem_cid("not_a_real_compound")
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

def get_chembl_from_cid(pubchem_cid: str, max_retries: int = 3) -> Optional[str]:
    """
    Get ChEMBL ID for a PubChem CID.
    
    Args:
        pubchem_cid (str): PubChem CID to look up
        max_retries (int): Maximum number of retry attempts on rate limit
        
    Returns:
        Optional[str]: ChEMBL ID if found, None otherwise
        
    Example:
        >>> get_chembl_from_cid("2244")  # Aspirin
        'CHEMBL25'
        >>> get_chembl_from_cid("invalid")
        None
    """
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
