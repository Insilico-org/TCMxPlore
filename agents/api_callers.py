from typing import Optional, Union
import json
import requests

from time import sleep
from functools import lru_cache
from threading import Lock

from typing import Optional, List, Dict, Any, Optional, ClassVar
from chembl_webresource_client.new_client import new_client

from dataclasses import dataclass, field

class EnrichrAnalysis:

    legend: tuple[str] = ('Rank', 'Term name', 'P-value', 'Odds ratio', 'Combined score',
              'Overlapping genes', 'Adjusted p-value', 'Old p-value', 'Old adjusted p-value')

    def __init__(self, glist:list[str], caller: callable):
        self.glist=glist
        self.glist_id = None
        self.res = None

        self.caller = caller(self)

    def analyze(self):
        self.caller()

class EnrichrCaller:

    url: str = 'https://maayanlab.cloud/Enrichr/'
    db: str = "KEGG_2015"

    def __init__(self, container: "EnrichrAnalysis"):
        self.cont = container

    def add_list(self, desc: str="NA"):
        q = self.url+'addlist'
        payload = dict(list=(None,"\n".join(self.cont.glist)),
                       description=(None, desc))

        response = requests.post(q, files=payload)
        if not response.ok:
            raise Exception('Error analyzing gene list')
        self.cont.glist_id = json.loads(response.text)["userlistId"]


    def enrich(self):
        q = self.url + f'enrich?userlistId={self.cont.glist_id}&backgroundType={self.db}'
        response = requests.get(q)
        if not response.ok:
            raise Exception('Error fetching enrichment results')

        self.cont.res = json.loads(response.text)

    def __call__(self, *args, **kwargs):
        '''
        Send a list of genes to Enrichr to get information about enriched pathways
        >> glist = ["ENTPD5","FBXO3","TRAP1", "KDR", "SEPHS2", "EXOSC4", "RILP","HOXA7"]
        >> enran=EnrichrAnalysis(glist, EnrichrCaller)
        >> enran.analyze()
        >> print(enran.res)
        Args:
            *args:
            **kwargs:

        Returns:

        '''
        self.add_list()
        self.enrich()
        print("Done with Enrichr analysis")


@dataclass
class PubChemCache:
    synonyms: Dict[int, List[str]] = field(default_factory=dict)
    descriptions: Dict[int, List[str]] = field(default_factory=dict)
    chembl_ids: Dict[int, str] = field(default_factory=dict)
    names: Dict[int, str] = field(default_factory=dict)
    cids: Dict[str, int] = field(default_factory=dict)


class PubChemAPI:
    """Thread-safe singleton PubChem API manager with caching"""

    _instance: ClassVar[Optional['PubChemAPI']] = None
    _lock: ClassVar[Lock] = Lock()
    MAX_SYNONYMS = 50

    ENDPOINTS = {
        'cids': "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON",
        'synonyms': "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/synonyms/JSON",
        'description': "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/description/JSON"
    }
    URL_MAX_LENGTH = 2000

    def __new__(cls, *args, **kwargs) -> 'PubChemAPI':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize only once due to singleton pattern"""
        if not hasattr(self, 'initialized'):
            self.max_retries = max_retries
            self.retry_delay = retry_delay
            self.cache = PubChemCache()
            self.initialized = True

    def _calculate_batch_size(self, items: list[str], base_url: str) -> int:
        """Calculate maximum batch size based on URL length limit"""
        url_overhead = len(base_url.format(""))
        item_overhead = 1
        max_items = (self.URL_MAX_LENGTH - url_overhead) // (max(len(str(x)) for x in items) + item_overhead)
        return min(max_items, 100)

    @lru_cache(maxsize=1000)
    def _make_request(self, endpoint: str, query: str, parser: callable):
        """Cached request handler"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                response = requests.get(endpoint.format(query))
                if response.status_code == 200:
                    return parser(response)
                elif response.status_code == 404:
                    return None
                elif response.status_code == 503:
                    retry_count += 1
                    sleep(self.retry_delay)
                    continue
                else:
                    return None
            except requests.RequestException:
                retry_count += 1
                sleep(self.retry_delay)
                continue
        return None

    def _make_batched_request(self,
                              endpoint: str,
                              items: list[str],
                              parser: callable,
                              join_char: str = ",") -> Optional[dict]:
        """Make batched requests with cache filtering"""
        batch_size = self._calculate_batch_size(items, endpoint)
        results = {}

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_str = join_char.join(batch)

            batch_result = self._make_request(endpoint, batch_str, parser)
            if batch_result:
                results.update(batch_result)

        return results if results else None

    def _format_compound_name(self, name: str) -> str:
        """Format compound name for API query"""
        query = name.strip().lower().replace("-", " ")
        if query.endswith("hcl") and not query.endswith(" hcl"):
            query = query.removesuffix("hcl") + " hcl"
        return query

    def _validate_cid(self, cid: Union[int, str]) -> str:
        """Validate and format CID"""
        cid = str(cid)
        if not cid.isnumeric():
            raise ValueError(f"Invalid CID format: {cid}")
        return cid

    def get_cids_batch(self, names: list[str]) -> Optional[dict[str, str]]:
        """Get PubChem CIDs for multiple compound names"""
        formatted_names = [self._format_compound_name(name) for name in names]
        results = {}

        # First check cache
        uncached_names = []
        for name in formatted_names:
            if name in self.cache.cids:
                results[name] = str(self.cache.cids[name])
            else:
                uncached_names.append(name)

        # Query API only for uncached names
        for name in uncached_names:
            def parse_cids(response: requests.Response) -> Optional[str]:
                data = json.loads(response.text)
                cids = data.get('IdentifierList', {}).get('CID', [])
                if cids:
                    cid = int(cids[0])
                    # Cache name->CID mapping
                    with self._lock:
                        self.cache.cids[name] = cid
                    return str(cid)
                return None

            cid = self._make_request(self.ENDPOINTS['cids'], name, parse_cids)
            if cid:
                results[name] = cid

        return results if results else None

    def get_synonyms_batch(self, cids: list[Union[int, str]]) -> Optional[dict[int, list[str]]]:
        """Get synonyms with caching and CHEMBL ID extraction"""
        validated_cids = [int(self._validate_cid(cid)) for cid in cids]
        uncached_cids = [cid for cid in validated_cids if cid not in self.cache.synonyms]

        if uncached_cids:
            def parse_synonyms(response: requests.Response) -> dict[int, list[str]]:
                data = json.loads(response.text)
                results = {}
                for info in data.get('InformationList', {}).get('Information', []):
                    cid = info.get('CID')
                    if not cid:
                        continue

                    all_synonyms = info.get('Synonym', [])
                    if all_synonyms:
                        # First cache CHEMBL ID from full list if present
                        with self._lock:
                            chembl_ids = [x for x in all_synonyms if x.startswith("CHEMBL")]
                            if chembl_ids:
                                self.cache.chembl_ids[int(cid)] = chembl_ids[0]

                        # Then truncate synonyms list and cache primary name
                        truncated_synonyms = all_synonyms[:self.MAX_SYNONYMS]
                        results[int(cid)] = truncated_synonyms
                        with self._lock:
                            self.cache.names[int(cid)] = truncated_synonyms[0]
                return results

            new_results = self._make_batched_request(
                self.ENDPOINTS['synonyms'],
                [str(cid) for cid in uncached_cids],
                parse_synonyms
            )

            if new_results:
                with self._lock:
                    self.cache.synonyms.update(new_results)

        return {cid: self.cache.synonyms[cid]
                for cid in validated_cids
                if cid in self.cache.synonyms}

    def get_descriptions_batch(self, cids: list[Union[int, str]]) -> Optional[dict[int, list[str]]]:
        """Get descriptions with caching"""
        validated_cids = [int(self._validate_cid(cid)) for cid in cids]
        uncached_cids = [cid for cid in validated_cids if cid not in self.cache.descriptions]

        if uncached_cids:
            def parse_descriptions(response: requests.Response) -> dict[int, list[str]]:
                data = json.loads(response.text)
                results = {}
                for info in data.get('InformationList', {}).get('Information', []):
                    cid = info.get('CID')
                    if not cid:
                        continue

                    descriptions = []
                    if 'Title' in info:
                        descriptions.append(info['Title'])
                    if 'Description' in info:
                        descriptions.append(info['Description'])

                    if descriptions:
                        results[int(cid)] = descriptions
                return results

            new_results = self._make_batched_request(
                self.ENDPOINTS['description'],
                [str(cid) for cid in uncached_cids],
                parse_descriptions
            )

            if new_results:
                with self._lock:
                    self.cache.descriptions.update(new_results)

        return {cid: self.cache.descriptions[cid]
                for cid in validated_cids
                if cid in self.cache.descriptions}

    def get_chembl_ids_batch(self, cids: list[Union[int, str]]) -> Optional[dict[int, str]]:
        """Get CHEMBL IDs (cached during synonym retrieval)"""
        validated_cids = [int(self._validate_cid(cid)) for cid in cids]
        uncached_cids = [cid for cid in validated_cids if cid not in self.cache.chembl_ids]

        if uncached_cids:
            self.get_synonyms_batch(uncached_cids)

        return {cid: self.cache.chembl_ids[cid]
                for cid in validated_cids
                if cid in self.cache.chembl_ids}

    def get_names_batch(self, cids: list[Union[int, str]]) -> Optional[dict[int, str]]:
        """Get primary names (cached during synonym retrieval)"""
        validated_cids = [int(self._validate_cid(cid)) for cid in cids]
        uncached_cids = [cid for cid in validated_cids if cid not in self.cache.names]

        if uncached_cids:
            self.get_synonyms_batch(uncached_cids)

        return {cid: self.cache.names[cid]
                for cid in validated_cids
                if cid in self.cache.names}
@dataclass
class CHEMBL_Annotation:
    chembl_id: str
    parent_id: Optional[str] = None
    activities: List[Dict[str, Any]] = None
    safety_warnings: List[Dict[str, Any]] = None
    indications: List[Dict[str, Any]] = None

    description: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'chembl_id': self.chembl_id,
            'parent_id': self.parent_id,
            'activities': self.activities or [],
            'safety_warnings': self.safety_warnings or [],
            'indications': self.indications or []
        }

    def add_desc(self, name: str, desc: str):
        self.name = name
        self.description = desc

# can't make use of list queries with __in due to pagination limits(

class ChemblBulkAPI:
    def __init__(self):
        self.molecule_client = new_client.molecule_form
        self.activity_client = new_client.activity
        self.safety_client = new_client.drug_warning
        self.indication_client = new_client.drug_indication

    def get_parent_molecules(self, chembl_ids: List[str]) -> Dict[str, str]:
        results = {}
        for chembl_id in chembl_ids:
            response = self.molecule_client.filter(
                molecule_chembl_id=chembl_id
            ).only(['molecule_chembl_id', 'parent_chembl_id'])

            for item in response[0:1]:
                mol_id = item.get('molecule_chembl_id')
                parent_id = item.get('parent_chembl_id')
                if mol_id and parent_id:
                    results[mol_id] = parent_id
        return results

    def get_activities(self, parent_ids: List[str], limit: int = 10) -> Dict[str, List[Dict]]:
        results = {}
        for parent_id in parent_ids:
            response = self.activity_client.filter(
                parent_molecule_chembl_id=parent_id
            ).only(['parent_molecule_chembl_id', 'assay_description'])

            results[parent_id] = [item for item in response[0:limit]]
        return results

    def get_safety_warnings(self, parent_ids: List[str], limit: int = 10) -> Dict[str, List[Dict]]:
        results = {}
        for parent_id in parent_ids:
            response = self.safety_client.filter(
                parent_molecule_chembl_id=parent_id
            ).only(['parent_molecule_chembl_id', 'warning_class'])

            results[parent_id] = [item for item in response[0:limit]]
        return results

    def get_indications(self, parent_ids: List[str], limit: int = 10) -> Dict[str, List[Dict]]:
        results = {}
        for parent_id in parent_ids:
            response = self.indication_client.filter(
                parent_molecule_chembl_id=parent_id
            ).only(['parent_molecule_chembl_id', 'efo_term', 'max_phase_for_ind'])

            results[parent_id] = [item for item in response[0:limit]]
        return results

    def create_annotations(self, chembl_ids: List[str], limit: int = 10) -> List[CHEMBL_Annotation]:
        annotations = []
        parent_mappings = self.get_parent_molecules(chembl_ids)
        parent_ids = list(set(parent_mappings.values()))

        activities = self.get_activities(parent_ids, limit)
        warnings = self.get_safety_warnings(parent_ids, limit)
        indications = self.get_indications(parent_ids, limit)

        for chembl_id in chembl_ids:
            parent_id = parent_mappings.get(chembl_id)
            annotation = CHEMBL_Annotation(
                chembl_id=chembl_id,
                parent_id=parent_id,
                activities=activities.get(parent_id, []),
                safety_warnings=warnings.get(parent_id, []),
                indications=indications.get(parent_id, [])
            )
            annotations.append(annotation)

        return annotations

def main():

    processor = ChemblBulkAPI()
    compounds = ["CHEMBL25", "CHEMBL192", "CHEMBL140"]
    res = processor.create_annotations(compounds)

    pubchem = PubChemAPI()
    cids = pubchem.get_cids_batch(['aspirin', 'paracetamol', 'ibuprofen', 'curcumin'])
    syns = pubchem.get_synonyms_batch([2244])
    pubchem.get_synonyms_batch([2244])
    pubchem.get_names_batch([2244])