# Inputs: patient anamnesis, gene lists to be targeted
# 1. Make a class to handle the overall workflow
# 2. Make an agent to find Jun and Chen herbs -- target-based
# 3. Make an agent that finds Zuo herbs -- toxicity-based
# 4. Make and agent that finds Shi herbs -- meridian-based

# Jun -- provide the principal therapeutic effect
# Chen -- Support the medical efficacy of the Jun medicine
# Zuo -- Treat associated symptoms or reduce toxicity of the other medicines;
# Shi -- direct other medicines to the diseased organ or contribute to the harmony of all herbs in the formula

# Extra considerations:
# - add blacklist of compounds
# - use tree search to find herb candidates
# - inspect all herbs added at each step for compatibility with patient and toxicity

############
from dotenv import load_dotenv
load_dotenv()

import json
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Tuple, Set, Any, Iterable, Union, OrderedDict
from json_repair import repair_json
from collections import defaultdict
import requests

from just_agents.base_agent import BaseAgent
from agents.tcm_crew import (TCMTools, find_TCM_herb, find_TCM_condition,
                             pick_cpds_by_targets, pick_herbs_by_targets,
                             pick_herbs_by_cids, get_description_from_cid,
                             get_compound_targets, get_description_from_cid_batch,
                             find_TCM_compound, TCMPropertyScorer, get_herbal_targets)
from agents.api_callers import  EnrichrAnalysis, EnrichrCaller
from dragon_db.annots import HerbAnnotation, TCMAnnotation
from abc import ABC, abstractmethod

GPT4TURBO = {
    "model": "gpt-4-turbo-preview",
    "temperature": 0.4,  # Lower temperature for more focused outputs
    "tools": []
}


@dataclass
class PatientProfile:
    """Stores patient-specific information and preferences"""
    age: int = field()
    sex: str = field()

    conditions: List[str] = field(default_factory=list, repr=False)
    allergies: List[str] = field(default_factory=list, repr=False)
    medications: List[str] = field(default_factory=list, repr=False)
    risk_factors: List[str] = field(default_factory=list, repr=False)

    goals: List[str] = field(default_factory=list, repr=False)

    all_conditions: List[str] = field(init=False, repr=False)

    def __post_init__(self):
        """Combine all relevant patient conditions for analysis"""
        self.all_conditions = (
            self.conditions +
            self.risk_factors +
            [f"Taking medication: {med}" for med in self.medications]
        )

    @classmethod
    def from_dict(cls, profile_data: Dict) -> 'PatientProfile':
        """Create PatientProfile from a dictionary"""
        return cls(
            age=profile_data.get('age', -1),
            sex=profile_data.get('sex', 'male'),

            conditions=profile_data.get('conditions', []),
            allergies=profile_data.get('allergies', []),
            medications=profile_data.get('medications', []),
            risk_factors=profile_data.get('risk_factors', []),

            goals=profile_data.get('goals', [])
        )

    def serialize(self, exclude: Set[str] = None) -> Dict[str, Any]:
        """
        Convert PatientProfile to a dictionary for serialization
        Args:
            exclude: Optional set of field names to exclude from serialization
        """
        exclude = exclude or set()
        serializable_fields = [
            field.name for field in fields(self)
            if field.init and field.name not in exclude
        ]

        return {
            field: getattr(self, field)
            for field in serializable_fields
        }

    def to_json(self) -> str:
        """Convert PatientProfile to JSON string"""
        return json.dumps(self.serialize())

# Safety check
class ContraindicationAgent(BaseAgent):
    """Agent specialized in analyzing herb contraindications against patient conditions"""

    def __init__(self, llm, completion_max_tries=3):

        super().__init__(llm_options=llm,
                         tools=[find_TCM_herb],
                         completion_max_tries=completion_max_tries)
        self.base_prompt = """You are a TCM expert specialized in analyzing herb safety and contraindications.
Your task is to determine if a herb is safe for a patient by analyzing the herb's contraindications
against the patient's conditions and risk factors."""

    def evaluate_safety(self,
                        herb_name: str,
                        patient_obj: 'PatientProfile') -> Tuple[bool, str]:
        """
        Evaluate if herb is safe for patient by analyzing contraindications

        Returns:
            Tuple of (is_safe: bool, reason: str)
        """
        herb_dict = find_TCM_herb(herb_name)['results'][0]
        herb_safety = dict()
        for i in ('contraindications', 'indication', 'interactions', 'incompatibility', 'notes'):
            if i in herb_dict:
                herb_safety[i] = herb_dict.pop(i)
        if not herb_safety:
            herb_safety['safety_category'] = 'Unknown'

        ser_patient = patient_obj.serialize(exclude={'risk_factors',
                                                     'medications',
                                                     'allergies',
                                                     'conditions'})
        patient_conditions = "\n".join(patient_obj.all_conditions)

        analysis_prompt = f"""
        Analyze if this herb is safe for the patient.
        {json.dumps(ser_patient, indent=2)}
        Their medical history reads:
        {json.dumps(patient_conditions, indent=2)}

        Herb description:
        {json.dumps(herb_dict, indent=2)}
        
        Herb notes on intended use and safety profile:
        {json.dumps(herb_safety, indent=2)}

        Respond in JSON format:
        {{
            "is_safe": true/false,
            "reason": "Brief explanation of decision"
        }}
        """

        joined_prompt = self.base_prompt + analysis_prompt
        print(f"Submitting a prompt with {len(joined_prompt)} characters")
        response = self.query(joined_prompt)
        try:
            result = json.loads(repair_json(response))
            return result["is_safe"], result["reason"]
        except (json.JSONDecodeError, KeyError):
            return False, "Failed to analyze herb safety"

## Herb selection
class BaseTCMAgent(BaseAgent):
    """Base class for TCM herb selection agents"""

    base_prompt: str = "Your are a helpful assistant"

    def __init__(self,
                 llm,
                 tools=None,
                 completion_max_tries=3,
                 max_compounds=50,
                 max_herbs=10,
                 max_desc_chars=2000):
        if tools is None:
            tools = [find_TCM_herb]
        super().__init__(
            llm_options=llm,
            tools=tools,
            completion_max_tries=completion_max_tries
        )

        self.blacklist = {
            "compounds": set(),
            "herbs": set()
        }

        self.max_compounds = max_compounds
        self.max_herbs = max_herbs
        self.max_desc_chars = max_desc_chars

        self.initialize()

    def initialize(self):
        self.instruct(self.base_prompt)

    def update_blacklist(self, terms: set, dtype: str):
        """Add herbs or compounds to blacklist"""
        self.blacklist[dtype] |= set(terms)

    def enriched_compounds(self,
                           gene_list: Iterable[str],
                           thr: float=0.0001,
                           **kwargs # blacklist + cpd_subset
                          ) -> Dict[int, Any]:
        """Find compounds targeting specified genes"""
        result = pick_cpds_by_targets(
            gene_list,
            tg_type="both",
            thr=thr,
            **kwargs
        )
        return {x: y for x, y in result.items()
                if x not in self.blacklist['compounds']}

    def prep_compound_summaries(self, cpd_descs: Dict[int, Dict[str, Any]]) -> str:
        """Format compound information for prompt"""
        summaries = []
        total_compounds = len(cpd_descs)
        previous_genes = None

        for cid, info in list(cpd_descs.items())[:self.max_compounds]:
            target_ratio = f"{info['Among them, proteins from the provided list']}/{info['Total protein targets']}"
            description = "\n".join(info.get('DB descriptions', ['Not available']))
            desc_str = f"({description})" if description else ""

            current_genes = list(info['Specific targets hit'])
            gene_str = f"{';'.join(current_genes)}"

            if previous_genes is not None and previous_genes == current_genes:
                if len("same as above") < len(gene_str):
                    gene_str = "same as above"

            summary = f"CID:{cid}{desc_str}: {target_ratio} targets [{gene_str}]"
            summaries.append(summary)
            previous_genes = current_genes

        cpd_descs_formatted = ' | '.join(summaries)
        if total_compounds > self.max_compounds:
            remaining = total_compounds - self.max_compounds
            cpd_descs_formatted += f"\n... and {remaining} other compounds with similar target profiles"

        return cpd_descs_formatted

    def get_herb_compounds_info(self,
                                gene_list: List[str],
                                jun_cpds: Dict,
                                chem_descriptions: bool = False) -> Dict:
        """Get detailed compound information for herbs"""
        if chem_descriptions:
            all_descs = get_description_from_cid_batch(list(jun_cpds.keys()))
        else:
            all_descs = {
                x: [find_TCM_compound(x)['results'][0]['init']['pref_name']]
                for x in jun_cpds
            }

        cpd_descs = {}
        if -1 not in jun_cpds:
            for cpd in jun_cpds:
                hit_tgs = set(get_compound_targets(cpd)) & set(gene_list)
                cpd_descs[cpd] = {
                    "DB descriptions": all_descs[cpd],
                    "Total protein targets": len(get_compound_targets(cpd)),
                    "Among them, proteins from the provided list": len(hit_tgs),
                    "Specific targets hit": hit_tgs
                }
        return cpd_descs

    def choose_another_herb(self, herb_name: str, reason: str):
        response = self.query(f"Herb {herb_name} is shown to be unsafe for the patient, due to "
                              f"the following considerations: \n {reason}\n"
                              f"\nSuggest an alternative herb among the preselected options "
                              f"using the same JSON output format")
        return(self.parse_response(response))

    @abstractmethod
    def prep_analysis_prompt(self, **kwargs) -> str:
        """Prepare analysis prompt - must be implemented by child classes"""
        pass

    @staticmethod
    def parse_response(response: str) -> Tuple[str, dict, str]:
        """Parse agent response into structured format"""
        try:
            result = json.loads(repair_json(response))
            result['herb_context'] = find_TCM_herb(result['herb_name'])
            return result["herb_name"], result['herb_context'], result["reason"]
        except (json.JSONDecodeError, KeyError):
            return "NA", "NA", "Failed to analyze herb selection"

    @staticmethod
    def calc_tcm_similarity(herb1_props: Dict, herb2_props: Dict) -> float:
        """Calculate similarity score between two herbs' TCM properties"""
        score = 0
        max_score = 0

        # Compare tastes
        if 'taste' in herb1_props and 'taste' in herb2_props:
            common_tastes = set(herb1_props['taste']) & set(herb2_props['taste'])
            score += len(common_tastes)
            max_score += max(len(herb1_props['taste']), len(herb2_props['taste']))

        # Compare temperatures
        if 'temperature' in herb1_props and 'temperature' in herb2_props:
            if herb1_props['temperature'] == herb2_props['temperature']:
                score += 1
            max_score += 1

        # Compare meridians
        if 'meridians' in herb1_props and 'meridians' in herb2_props:
            common_meridians = set(herb1_props['meridians']) & set(herb2_props['meridians'])
            score += len(common_meridians)
            max_score += max(len(herb1_props['meridians']), len(herb2_props['meridians']))

        return score / max_score if max_score > 0 else 0

    @staticmethod
    def herb_common_compound_counts(cpds: Iterable,
                                    blacklist: Optional[set] = None) -> dict[str, int]:
        """Find herbs that share compounds with Jun herb"""
        common_cpds = {}
        cpds = set(cpds)
        blacklist = blacklist or set()
        blacklist = set(blacklist)

        for cid in cpds:
            compound_info = find_TCM_compound(cid)
            if compound_info['results']:
                # NB: only 5000 herbs will be used due to link truncation in `find_TCM_compound`
                cpd_herbs = compound_info['results'][0]['links'].get('herbs', [])
                for herb in cpd_herbs:
                    if herb in blacklist:
                        continue
                    common_cpds[herb] = common_cpds.get(herb, 0) + 1
        return common_cpds

class JunHerbAgent(BaseTCMAgent):
    """Agent specialized in selecting Jun (Emperor) herbs"""

    base_prompt: str = """You are a TCM practitioner specialized in selecting Jun (Monarch) herbs.
    Your task is to analyze herbs and select those that best fulfill the Jun (Monarch) role while maintaining
    formula harmony and avoiding any contraindications. Pay special attention to patient safety and 
    medical history when evaluating herbs.
    
    The Monarch is the ingredient that directly treats the principal syndrome, which is manifested by the main symptoms.
    The Monarch can be one ingredient, but it can also be a herbal combination. Generally speaking, the Monarch has a 
    relatively large dosage within the formula as a whole, and it enters the meridians where the pathological 
    changes are manifest.
    
    As a Jun herb selector, focus on:
    1. Direct targeting of main symptoms and conditions
    2. Strong affinity for specified target genes
    3. Mostly safe primary therapeutic effects
    4. Appropriate dosage range for principal herbs
    """

    def find_jun_herb(self,
                      gene_list: List[str],
                      patient_obj: 'PatientProfile',
                      chem_descriptions: bool = False,
                      blacklist_cpds: Optional[set] = None,
                      blacklist_herbs: Optional[set] = None) -> Tuple[str, dict, str]:
        """Find best Jun herb based on gene targets and patient profile"""

        # Get compounds targeting genes
        blacklist_cpds = blacklist_cpds or set()
        blacklist_cpds = set(blacklist_cpds)
        blacklist_herbs = blacklist_herbs or set()
        blacklist_herbs = set(blacklist_herbs)

        best_cpds = self.enriched_compounds(gene_list, blacklist=blacklist_cpds)

        # Get herb candidates based on compounds
        best_herbs1 = pick_herbs_by_cids(best_cpds, N_top=10, blacklist = blacklist_herbs)
        if best_herbs1 is None:
            best_herbs1 = {"NA": "No herbs found"}

        # Get herb candidates based on protein targets
        best_herbs2 = self.get_protein_herb_candidates(gene_list, blacklist_herbs = blacklist_herbs)
        if best_herbs2 is None:
            best_herbs2 = {"NA": "No herbs found"}

        # Get compound descriptions
        cpd_descs = self.get_herb_compounds_info(
            gene_list, best_cpds, chem_descriptions
        )

        # Prepare herb stats and entries
        herb_stats = {
            'Herbs selected based on their ingredients':
                [('Herb name', 'N compounds ')] + list(best_herbs1.items()),
            "Herbs selected based on protein targets":
                [('Herb name', 'N targets affected')] + list(best_herbs2.items())
        }

        herb_entries = {
            x: find_TCM_herb(x)['results'][0]
            for x in best_herbs1 | best_herbs2
        }
        print(f"Considering these herbs for Jun:\n{list(herb_entries.keys())}")

        # Generate and process analysis
        analysis_prompt = self.prep_analysis_prompt(
            patient_obj=patient_obj,
            gene_list=gene_list,
            cpd_descs=cpd_descs,
            herb_stats=herb_stats,
            herb_entries=herb_entries
        )

        print(f"Submitting a prompt with {len(analysis_prompt)} characters")
        response = self.query(analysis_prompt)

        return self.parse_response(response)

    def get_protein_herb_candidates(self,
                                    genes: list[str],
                                    N_top: int = 10,
                                    blacklist_herbs: Optional[set] = None) -> dict[str, int]:
        """Get herbs based on protein target overlap"""
        blacklist_herbs = blacklist_herbs or set()
        blacklist_herbs = set(blacklist_herbs)

        pre_result = pick_herbs_by_targets(genes,
                                           tg_type="both",
                                           top_herbs=.2,
                                           blacklist = blacklist_herbs)
        result = {}
        for i, k in enumerate(pre_result):
            if i == N_top:
                break
            result[k] = pre_result[k]
        return result

    def prep_analysis_prompt(self, **kwargs) -> str:
        """Prepare analysis prompt for Jun herb selection"""
        patient_obj = kwargs['patient_obj']
        gene_list = kwargs['gene_list']
        cpd_descs = kwargs['cpd_descs']
        herb_stats = kwargs['herb_stats']
        herb_entries = kwargs['herb_entries']

        ser_patient = patient_obj.serialize(exclude={
            'risk_factors', 'medications', 'allergies', 'conditions'
        })
        patient_conditions = ";".join(patient_obj.all_conditions)
        cpd_descs_formatted = self.prep_compound_summaries(cpd_descs)

        herb_entries = {
            # x: y['init'] | {'condition_links': y['links']['conditions']}
            x: y['init'] for x, y in herb_entries.items()
        }

        return f"""
        Given the following pre-selected herbs and their associated data, select the most appropriate 
        Jun (Emperor) herb that will serve as the primary therapeutic agent in the formula. 

        Patient Profile:
        {json.dumps(ser_patient, indent=2)}
        Patient's medical history reads:
        {json.dumps(patient_conditions, indent=2)}

        Analysis Requirements:
        1. Review each herb's:
           - Target proteins and their relevance to the patient's health status
           - Known compounds contained in herbs and their therapeutic effects
           - Traditional TCM properties and organ meridians
           - Safety profile and potential interactions

        2. Consider:
           - Primary therapeutic alignment with patient's condition
           - Strength and directness of effect on target symptoms
           - Safety and contraindication profile
           - Traditional usage as a Jun herb in classical formulas

        3. Evaluate herb candidates based on:
           - Number of relevant protein targets hit
           - Number of beneficial compounds present
           - Historical evidence as a Jun herb
           - Safety record in clinical practice

        Please analyze the provided herb data and select ONE herb that best fulfills the Jun 
        role in a TCM formula to be completed later. 
        Explain your reasoning clearly but concisely.

        Please focus particularly on how the selected herb directly benefits the patient
        from the viewpoint of its molecular targets, associated biological pathways, 
        and TCM-termed actions.

        See below the information collected on herb candidates.

        - Original genetic targets to be affected by the herb: 
        {';'.join(gene_list)}

        - Compounds affecting the genetic targets (in order of ascending P-value):
        {cpd_descs_formatted}

        - Results of herb database screening:
        {herb_stats}

        - Reference information on candidate herbs
        {herb_entries}
        
        Respond in JSON format:
        {{
            "herb_name": "primary name of the herb",
            "reason": "Brief explanation of decision"
        }}
        """

class ChenHerbAgent(BaseTCMAgent):
    """Agent specialized in selecting Chen (Minister) herbs"""

    base_prompt: str = """You are a TCM practitioner specialized in selecting Chen (Minister) herbs.
        Your task is to analyze herbs and select those that best complement the Jun herb
        while maintaining formula harmony.
        
        
        • The Minister is the ingredient that has the function of accentuating and enhancing the effect of the Jun (Emperor) 
            ingredient to treat the principal syndrome.
        • The Minister serves as the main ingredient acting directly against a coexisting syndrome.
        
        Focus on selecting herbs that:
        1. Target genes and pathways not covered by the Jun herb
        2. Support and extend the therapeutic action of the Jun herb
        4. Have compatible safety profiles with both the Jun herb and patient conditions
        5. Satisfy TCM pairing principles with Jun and the patient (taste, actions, temperature)        

        """

    def find_chen_herb(self,
                       gene_list: List[str],
                       patient_obj: 'PatientProfile',
                       jun_herb_name: str,
                       blacklist_cpds: Optional[set] = None,
                       blacklist_herbs: Optional[set] = None,
                       chem_descriptions: bool = False) -> tuple[str, dict, str]:
        """Find best Chen herb to complement Jun herb"""

        blacklist_cpds = blacklist_cpds or set()
        blacklist_cpds = set(blacklist_cpds)
        blacklist_herbs = blacklist_herbs or set()
        blacklist_herbs = set(blacklist_herbs) | {jun_herb_name}

        # Get Jun herb information
        jun_herb_batman = find_TCM_herb(jun_herb_name, db_id='BATMAN')
        jun_herb_dragon = find_TCM_herb(jun_herb_name)
        jun_cpds = jun_herb_batman['results'][0]['links']['ingrs']

        enriched_cpds = self.enriched_compounds(gene_list, blacklist = blacklist_cpds)
        # only keep relevant compounds
        jun_cpds = {cid: stats for cid, stats in enriched_cpds.items() if cid in jun_cpds}

        # Find herbs sharing compounds with Jun herb
        common_cpds = self.herb_common_compound_counts(jun_cpds, blacklist = blacklist_herbs)

        best_herbs = dict(sorted(
            common_cpds.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_herbs])

        # Get compound information
        cpd_descs = self.get_herb_compounds_info(
            gene_list, jun_cpds, chem_descriptions
        )

        # Process herb entries
        herb_entries = self.process_herb_entries(
            best_herbs, jun_herb_dragon['results'][0]['init'].get('properties', {})
        )

        # Generate and process analysis
        analysis_prompt = self.prep_analysis_prompt(
            patient_obj=patient_obj,
            gene_list=gene_list,
            cpd_descs=cpd_descs,
            herb_stats={"(Herb name, Compounds shared with Jun)": list(best_herbs.items())},
            herb_entries=herb_entries,
            jun_herb_obj=jun_herb_dragon
        )

        print(f"Submitting a prompt with {len(analysis_prompt)} characters")
        response = self.query(analysis_prompt)
        return self.parse_response(response)

    def process_herb_entries(self, best_herbs: Dict[str, int], jun_properties: Dict) -> Dict:
        """Process herb entries with TCM properties and molecular data"""
        herb_entries = {}
        for herb_name in best_herbs:
            if herb_name != "NA":
                trad_info = find_TCM_herb(herb_name)
                if trad_info['results']:
                    herb_entries[herb_name] = trad_info['results'][0]

                    chen_properties = herb_entries[herb_name]['init'].get('properties', {})
                    similarity_score = self.calc_tcm_similarity(
                        jun_properties, chen_properties
                    )

                    mol_info = find_TCM_herb(herb_name, db_id='BATMAN')
                    if mol_info['results']:
                        herb_entries[herb_name]['analysis'] = {
                            'shared_compounds': best_herbs[herb_name],
                            'total_compounds': len(mol_info['results'][0]['links'].get('ingrs', [])),
                            'tcm_similarity': similarity_score
                        }
        return herb_entries

    def prep_analysis_prompt(self, **kwargs) -> str:
        """Prepare analysis prompt for Chen herb selection"""
        patient_obj = kwargs['patient_obj']
        gene_list = kwargs['gene_list']
        cpd_descs = kwargs['cpd_descs']
        herb_stats = kwargs['herb_stats']
        herb_entries = kwargs['herb_entries']
        jun_herb_obj = kwargs['jun_herb_obj']

        ser_patient = patient_obj.serialize(exclude={
            'risk_factors', 'medications', 'allergies', 'conditions'
        })
        patient_conditions = ";".join(patient_obj.all_conditions)
        cpd_descs_formatted = self.prep_compound_summaries(cpd_descs)

        herb_entries = {
            # x: y['init'] | {'condition_links': y['links']['conditions']}
            x: y['init'] for x, y in herb_entries.items()
        }

        return f"""
        Select the most appropriate Chen (Minister) herb to complement this Jun herb:
        {json.dumps(jun_herb_obj['results'][0]['init'], indent=2)}

        Patient Profile:
        {json.dumps(ser_patient, indent=2)}
        Patient conditions:
        {json.dumps(patient_conditions, indent=2)}

        Analysis Requirements:
        1. Review each candidate herb's:
           - Target proteins not covered by Jun herb
           - Known compounds and their effects
           - TCM properties relative to Jun herb
           - Safety profile and interactions

        2. Consider:
           - Complementarity with Jun herb's actions
           - Coverage of untargeted pathways
           - Traditional pairing principles
           - Overall formula harmony

        3. Evaluate candidates based on:
           - Number of unique protein targets
           - TCM property alignment
           - Historical pairing evidence 
           - Safety with Jun herb

        The Chen herb should extend and support the Jun herb's therapeutic action
        while maintaining formula harmony. Focus on TCM property compatibility
        and coverage of untargeted genes.

        Genes to be targeted by Jun and Chen:
        {';'.join(gene_list)}

        Compounds affecting target genes (in order of significance):
        {cpd_descs_formatted}

        Candidate herb statistics:
        {herb_stats}

        Detailed herb information:
        {herb_entries}

        Respond in JSON format:
        {{
            "herb_name": "primary name of selected herb",
            "reason": "Brief explanation emphasizing TCM harmony and pathway coverage"
        }}
        """

class ZuoHerbAgent(BaseTCMAgent):
    """Agent specialized in selecting Zuo (Assistant) herbs to address secondary symptoms"""
    base_prompt: str = """You are a TCM practitioner specialized in selecting Zuo (Assistant) herbs.
    
     The Assistant accentuates and enhances the therapeutic effect of the Jun (Emperor) or Chen (Minister) ingredients, 
     or directly treats secondary symptoms. The ingredient that has this function can be considered a helping Assistant.
    • The Assistant moderates or eliminates the toxicity or harsh properties of the Jun or Chen ingredients. 
        The ingredient that has this function can be considered a corrective assistant.
    • The Assistant has a function or a moving tendency which goes against the Emperor ingredient but which is helpful 
        in fulfilling the therapeutic effect and which may be used in complicated and serious conditions. The ingredient
         that has this function can be considered a strategic Assistant.
    
    Your task is to analyze herbs and select those that best fulfill the Zuo role by:
    1. Addressing secondary symptoms not well-managed by Jun (Emperor) and Chen (Minister) herbs
    2. Supporting and harmonizing with the primary therapeutic strategy
    3. Providing additional therapeutic effects without overshadowing Jun-Chen
    4. Ensuring compatibility with existing formula components
    
    """

    def __init__(self, **kwargs):
        super().__init__(tools=[find_TCM_herb, find_TCM_condition], **kwargs)

    def find_zuo_herb(self,
                      patient_obj: 'PatientProfile',
                      jun_herb_name: str,
                      chen_herb_name: str,
                      blacklist_herbs: Optional[set] = None) -> tuple[str, dict, str]:
        """Find best Zuo herb to address secondary symptoms"""
        blacklist_herbs = blacklist_herbs or set()
        blacklist_herbs = set(blacklist_herbs) | {jun_herb_name, chen_herb_name}

        # Get existing herb information using tool
        jun_info = find_TCM_herb(jun_herb_name)['results'][0]
        chen_info = find_TCM_herb(chen_herb_name)['results'][0]

        # Extract primary actions
        jun_actions = self._extract_herb_actions(jun_info)
        chen_actions = self._extract_herb_actions(chen_info)
        primary_actions = jun_actions | chen_actions

        # Identify secondary symptoms
        secondary_symptoms = self._find_secondary_symptoms(
            primary_actions,
            patient_obj.all_conditions
        )

        # Get candidate herbs for each symptom using tool
        candidate_herbs = set()
        for symptom in secondary_symptoms:
            symptom_info = find_TCM_condition(symptom)
            if symptom_info['results']:
                for result in symptom_info['results']:
                    if 'links' in result and 'herbs' in result['links']:
                        candidate_herbs.update(result['links']['herbs'])

        # Remove existing herbs
        candidate_herbs = candidate_herbs - blacklist_herbs

        # Process candidates
        herb_entries = {}
        for herb_name in list(candidate_herbs)[:self.max_herbs]:
            herb_info = find_TCM_herb(herb_name)
            if not herb_info['results']:
                continue

            herb_entry = herb_info['results'][0]

            # Analyze using internal methods
            coverage = self._analyze_herb_coverage(herb_entry, secondary_symptoms)
            compatibility = self._check_herb_compatibility(herb_entry, jun_info, chen_info)

            herb_entries[herb_name] = {
                'info': herb_entry,
                'analysis': {
                    'symptom_coverage': coverage,
                    'compatibility_issues': compatibility
                }
            }

        # Generate and process analysis
        analysis_prompt = self.prep_analysis_prompt(
            patient_obj=patient_obj,
            herb_entries=herb_entries,
            existing_herbs={
                'Jun': jun_info,
                'Chen': chen_info
            },
            secondary_symptoms=secondary_symptoms
        )

        print(f"Submitting a prompt with {len(analysis_prompt)} characters")
        response = self.query(analysis_prompt)

        return self.parse_response(response)

    @staticmethod
    def _extract_herb_actions(herb_info: dict) -> set:
        """Extract therapeutic actions and indications from herb info"""
        actions = set()

        # Extract from actions field
        if 'actions' in herb_info['init']:
            for action in herb_info['init']['actions']:
                if isinstance(action, str):
                    actions.add(action.lower())
                elif isinstance(action, dict) and 'action' in action:
                    actions.add(action['action'].lower())

        # Extract from treatment indications
        if 'treats' in herb_info['init']:
            actions.update(cond.lower() for cond in herb_info['init']['treats'])

        return actions

    @staticmethod
    def _find_secondary_symptoms(
        primary_actions: set,
        patient_conditions: list) -> list:
        """Identify patient symptoms not addressed by primary actions"""
        secondary = []

        for condition in patient_conditions:
            cond_lower = condition.lower()
            if not any(action in cond_lower or cond_lower in action
                       for action in primary_actions):
                secondary.append(condition)

        return secondary

    @staticmethod
    def _analyze_herb_coverage(
        herb_info: dict,
        symptoms: list) -> dict:
        """Analyze how well an herb addresses given symptoms"""
        coverage = {
            'symptoms_addressed': [],
            'coverage_score': 0
        }

        symptoms_lower = [s.lower() for s in symptoms]

        # Check actions
        if 'actions' in herb_info['init']:
            for action in herb_info['init']['actions']:
                action_text = (action['action'] if isinstance(action, dict)
                               else action).lower()
                for symptom in symptoms_lower:
                    if symptom in action_text or action_text in symptom:
                        coverage['symptoms_addressed'].append(symptom)

        # Check treatment indications
        if 'treats' in herb_info['init']:
            for condition in herb_info['init']['treats']:
                condition = condition.lower()
                for symptom in symptoms_lower:
                    if symptom in condition or condition in symptom:
                        coverage['symptoms_addressed'].append(symptom)

        coverage['symptoms_addressed'] = list(set(coverage['symptoms_addressed']))
        if symptoms:
            coverage['coverage_score'] = len(coverage['symptoms_addressed']) / len(symptoms)

        return coverage

    def _check_herb_compatibility(
        self,
        herb_info: dict,
        jun_info: dict,
        chen_info: dict) -> list:
        """Check for compatibility issues between herbs"""
        issues = []

        # Get primary actions
        primary_actions = self._extract_herb_actions(jun_info)
        primary_actions.update(self._extract_herb_actions(chen_info))

        # Check contraindications
        if 'contraindications' in herb_info['init']:
            for contra in herb_info['init']['contraindications']:
                if any(action in contra.lower() for action in primary_actions):
                    issues.append(f"Contraindication with primary action: {contra}")

        # Check traditional incompatibilities
        if 'incompatibility' in herb_info['init']:
            for incomp in herb_info['init']['incompatibility']:
                if incomp in (jun_info['init'].get('preferred_name', ''),
                              chen_info['init'].get('preferred_name', '')):
                    issues.append(f"Traditional incompatibility with {incomp}")

        return issues

    def prep_analysis_prompt(self, **kwargs) -> str:
        """Prepare analysis prompt for Zuo herb selection"""
        patient_obj = kwargs['patient_obj']
        herb_entries = kwargs['herb_entries']
        existing_herbs = kwargs['existing_herbs']
        secondary_symptoms = kwargs['secondary_symptoms']

        ser_patient = patient_obj.serialize(exclude={
            'risk_factors', 'medications', 'allergies', 'conditions'
        })

        return f"""
        Select the most appropriate Zuo (Assistant) herb to address secondary symptoms 
        while harmonizing with the existing formula:

        Jun (Emperor) herb:
        {json.dumps(existing_herbs['Jun']['init'], indent=2)}

        Chen (Minister) herb:
        {json.dumps(existing_herbs['Chen']['init'], indent=2)}

        Patient Profile:
        {json.dumps(ser_patient, indent=2)}

        Patient Conditions:
        {json.dumps(patient_obj.all_conditions, indent=2)}

        Secondary Symptoms Requiring Management:
        {json.dumps(secondary_symptoms, indent=2)}

        Analysis Requirements:
        1. Review each candidate herb's:
           - Coverage of secondary symptoms
           - Compatibility with Jun-Chen pair
           - Traditional assistant herb usage
           - Safety profile

        2. Consider:
           - Effectiveness for target symptoms
           - Harmonious interaction with formula
           - Support for primary therapeutic strategy
           - Known synergistic effects

        Candidate herb details:
        {json.dumps({name: {
            'properties': entry['info']['init'].get('properties', {}),
            'actions': entry['info']['init'].get('actions', []),
            'analysis': entry['analysis'],
            'contraindications': entry['info']['init'].get('contraindications', [])
        } for name, entry in herb_entries.items()}, indent=2)}

        Respond in JSON format:
        {{
            "herb_name": "selected herb name",
            "reason": "Brief explanation emphasizing symptom coverage and harmony"
        }}
        """

class ShiHerbAgent(BaseTCMAgent):

    base_prompt: str = """You are a TCM practitioner specialized in selecting Shi (Guide) herbs.
        Your task is to analyze the current formula's properties and determine the ideal TCM characteristics
        (meridians, tastes, temperatures) needed in a Shi herb to optimally direct the formula's therapeutic
        effects.
        
        The Guide serves to bring the rest of the formula into the meridian or region where the main pathological change 
        exists, allowing its actions to focus on this specific region. In most cases, the Jun (Emperor) carries out 
        this function as it must enter the place where the pathological change is located. The Guide harmonizes 
        and integrates the actions of the other ingredients in order to balance the action, 
        temperature, speed and direction of the formula.
        
        As a Shi herb selector, focus on:
        1. Analyzing existing formula components' properties
        2. Identifying target organs/meridians based on patient condition
        3. Determining complementary properties needed for guidance or attenuation of certain TCM effects
        4. Generating appropriate weights for TCM properties

        """

    step2_prompt: str = """
        
        Now, based on your analysis and scoring, select the herb among the preselected Shi candidates 
        that will improve the quality of the Jun-Chen-Zuo formula the most. Return the your answer as a JSON: 
        {
            "reason":"explanation of your choice",
            "herb_name": "The preferred name of the selected Shi herb"
        }
        
        You may explore individual herbs in the database to give more accurate answers
        
        """

    def initialize(self):
        self.instruct(self.base_prompt)
        self.scorer = TCMPropertyScorer()

    def calculate_property_weights(self,
                                   patient_obj: 'PatientProfile',
                                   formula_herbs: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate weights for TCM properties based on formula analysis"""

        # Format herbs data for prompt with normalized properties
        herb_details = {}
        for role, herb_info in formula_herbs.items():
            properties = herb_info['results'][0]['init'].get('properties', {})
            normalized_properties = self.scorer.normalize_herb_properties(properties)

            herb_details[role] = {
                "name": herb_info['results'][0]['init']['preferred_name'],
                "properties": normalized_properties,
                "actions": herb_info['results'][0]['init'].get('actions', [])
            }

        # Generate analysis prompt
        analysis_prompt = self.prep_analysis_prompt(
            patient_obj=patient_obj,
            herb_details=herb_details
        )

        print(f"Submitting a prompt with {len(analysis_prompt)} characters")
        response = self.query(analysis_prompt)

        try:
            weights = json.loads(repair_json(response))
            return {
                "meridian_weights": weights["meridian_weights"],
                "taste_weights": weights["taste_weights"],
                "temperature_weights": weights["temperature_weights"],
                "reason": weights.get("reasoning", "")
            }
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "meridian_weights": {"default": 1.0},
                "taste_weights": {"default": 1.0},
                "temperature_weights": {"default": 1.0},
                "reason": "Error parsing agent response"
            }

    def prep_analysis_prompt(self,
                             patient_obj: 'PatientProfile',
                             herb_details: Dict[str, Dict]) -> str:
        """Prepare prompt for property weight analysis"""

        standard_properties = {
            "Tastes": list(self.scorer.STANDARD_TASTES.keys()),
            "Temperatures": list(self.scorer.STANDARD_TEMPERATURES.keys()),
            "Meridians": list(self.scorer.STANDARD_MERIDIANS.keys())
        }

        return f"""
        
        The weights you provide will be used to score candidate herbs, so ensure they reflect
        the relative importance of each property in guiding the formula's effects.
        
        Analyze the current formula components and patient profile to determine optimal
        weights for TCM properties in selecting a Shi (Guide) herb.

        Patient Profile:
        {json.dumps(patient_obj.serialize(), indent=2)}

        Current Formula Herbs:
        {json.dumps(herb_details, indent=2)}

        Based on this information, determine appropriate weights (0.0-1.0) for TCM properties
        that will help select a Shi herb to:
        1. Guide the formula to the organs/meridians where the action of other herbs is needed the most 
        2. Balance the properties of other herbs, such as temperature and taste . 
        3. Address the patient's specific needs and the properties of their conditions.

        Use only the following standardized properties in your weights:
        {json.dumps(standard_properties, indent=2)}

        For each property type, assign weights between -1.0 and 1.0 where:
        - Positive values (0.1 to 1.0) indicate desirable properties
        - Negative values (-1.0 to -0.1) indicate undesirable properties that should be avoided
        - 0.0 indicates neutral properties that neither help nor hinder

        You must provide weights for ALL standard properties listed above, not just the desirable ones.

        Respond in JSON format and avoid additional comments outside of this JSON:
        {{  "reasoning": "Brief explanation of weight assignments, including why certain properties are contraindicated",
            "meridian_weights": {{
                "meridian_name": weight,  # e.g. "Liver": 0.8, "Triple Burner": -0.3
                ... # weights for ALL standard meridians
            }},
            "taste_weights": {{
                "taste_name": weight,  # e.g. "Sweet": 0.6, "Bitter": -0.4
                ... # weights for ALL standard tastes
            }},
            "temperature_weights": {{
                "temperature_name": weight,  # e.g. "Warm": 0.7, "Cold": -0.5
                ... # weights for ALL standard temperatures
            }}
        }}
        """

    def select_shi_candidates(self,
                              patient_obj: 'PatientProfile',
                              herb_names: dict[str,str],
                              blacklist_herbs: Optional[set]=None,
                              N: int=10,
                              min_score: float=0.4) -> dict[str, Any]:

        blacklist_herbs = blacklist_herbs or set()
        blacklist_herbs = set(blacklist_herbs) | set(herb_names.values())

        formula_herbs = {x:find_TCM_herb(y) for x,y in herb_names.items()} # herb status : herb details
        target_weights = self.calculate_property_weights(
                                            patient_obj=patient_obj,
                                            formula_herbs=formula_herbs
                                                        )

        candidates = self.scorer.find_best_shi_herbs(
                                                    target_weights=target_weights,
                                                    blacklist=blacklist_herbs,
                                                    N_top=N,
                                                    min_score=min_score
                                                )
        return   {"shi_candidates":candidates} | target_weights

    def select_shi_herb(self, *args, **kwargs ):
        candidates = self.select_shi_candidates(*args, **kwargs)

        response = self.query(self.step2_prompt + str(candidates))
        out = list(self.parse_response(response))
        out[2] = "Weights reason:\n"+ candidates['reason'] + "\nShi selection reason:\n" + out[2]
        return out

## Formula composition

class FormulaAnalysis:
    """Stores and analyzes a complete TCM formula design"""

    def __init__(self, patient: 'PatientProfile'):
        self.patient = patient
        self.herbs = {
            'jun': None,  # Will store HerbAnnotation objects
            'chen': None,
            'zuo': None,
            'shi': None
        }
        self.reasoning = {
            'jun': '',
            'chen': '',
            'zuo': '',
            'shi': ''
        }
        self.safety_checks = {
            'jun': ('', ''),  # (is_safe: bool, reason: str)
            'chen': ('', ''),
            'zuo': ('', ''),
            'shi': ('', '')
        }
        self.rejected_herbs = defaultdict(list)  # role -> [(herb_name, reason)]

    def add_herb(self,
                 role: str,
                 herb: 'HerbAnnotation',
                 reason: str,
                 safety: tuple[bool, str]):
        """Add a herb to the formula with its selection reasoning"""
        self.herbs[role] = herb
        self.reasoning[role] = reason
        self.safety_checks[role] = safety

    def add_rejection(self, role: str, herb_name: str, reason: str):
        """Record a rejected herb and the reason"""
        self.rejected_herbs[role].append((herb_name, reason))

    def get_formula_composition(self) -> Dict[str, Any]:
        """Get the complete formula analysis results"""
        return {
            'status': 'complete' if all(self.herbs.values()) else 'incomplete',
            'patient': self.patient.serialize(),
            'formula': {
                role: herb['preferred_name'] if herb else None
                for role, herb in self.herbs.items()
            },
            'reasoning': self.reasoning,
            'safety': {
                role: check[1] for role, check in self.safety_checks.items()
            },
            'rejected': dict(self.rejected_herbs)
        }


class FormulaDesignAgent:
    """
    Coordinates TCM formula design process through herb selection and safety validation.
    Acts as a builder for FormulaAnalysis objects.
    """

    def __init__(self, llm_config: dict):
        # Initialize sub-agents
        self.contraindication_agent = ContraindicationAgent(llm=llm_config)
        self.jun_agent = JunHerbAgent(llm=llm_config)
        self.chen_agent = ChenHerbAgent(llm=llm_config)
        self.zuo_agent = ZuoHerbAgent(llm=llm_config)
        self.shi_agent = ShiHerbAgent(llm=llm_config)

        # Define global blacklists
        self.global_blacklist = {
            'compounds': {
                5462328,  # Heroin
                175  # Acetate
            },
            'herbs': {"GOU SHEN", 'WU GONG','ZHI CAO WU',
                      'MENG CHONG', 'HA MA YOU', 'XIONG DAN',
                      'SHE XIANG', 'ZI HE CHE', 'DONG CHONG XIA CAO',
                      'LU RONG', 'JIU', 'REN NIAO'}
        }

        toxic_herbs = {'FU ZI', 'WU TOU', 'XI XIN', 'MA HUANG',
                       'YANG JIN HUA', 'LEI GONG TENG', 'WU GONG', 'QUAN XIE',
                       'BAI HUA SHE', 'MANG CHONG', 'ZHE CHONG', 'SHAN DOU GEN',
                       'BAN XIA', 'TIAN NAN XING', 'BAI FU ZI', 'WEI LING XIAN',
                       'XIAN MAO', 'WU ZHU YU', 'HUA JIAO', 'YUAN ZHI',
                       'KU LIAN PI', 'HE SHI', 'GUA DI', 'LI LU',
                       'CHANG SHAN', 'GAN SUI', 'DA JI', 'YUAN HUA',
                       'SHANG LU', 'QIAN NIU ZI', 'BA DOU', 'ZHU SHA'}
        self.global_blacklist['herbs'] |= toxic_herbs

        self.max_retries = 5

    def _get_herb_blacklist(self, analysis: FormulaAnalysis) -> Set[str]:
        """Combine global blacklist with patient-specific contraindications"""
        blacklist = self.global_blacklist['herbs'].copy()

        # Add herbs rejected for this patient
        for rejected in analysis.rejected_herbs.values():
            blacklist |= set([herb for herb, _ in rejected])

        # Could expand here to add patient-specific contraindications
        # based on their conditions, medications, etc.

        return blacklist

    def check_herb_safety(self,
                          herb_name: str,
                          analysis: FormulaAnalysis) -> tuple[bool, str]:
        """Check if an herb is safe for the current patient"""
        return self.contraindication_agent.evaluate_safety(
            herb_name=herb_name,
            patient_obj=analysis.patient
        )

    def select_jun_herb(self,
                        analysis: FormulaAnalysis,
                        gene_targets: List[str]) -> bool:
        """Select a safe Jun herb targeting specified genes"""

        herb_blacklist = self._get_herb_blacklist(analysis)
        name, context, reason = self.jun_agent.find_jun_herb(
            gene_list=gene_targets,
            patient_obj=analysis.patient,
            blacklist_herbs=herb_blacklist,
            blacklist_cpds=self.global_blacklist['compounds']
        )
        assert not name in herb_blacklist, f"Herb {name} found in blacklist!"
        if name == "NA":
            return False

        for _ in range(self.max_retries):

            herb_dict = context['results'][0]['init']
            is_safe, safety_reason = self.check_herb_safety(name, analysis)

            if is_safe:
                analysis.add_herb('jun', herb_dict, reason, (is_safe, safety_reason))
                print(f"Herb {name} added as Jun")
                return True

            analysis.add_rejection('jun', name, safety_reason)
            print(f"==> Herb {name} found not safe for patient")
            name, context, reason = self.jun_agent.choose_another_herb(name, safety_reason)

        return False

    def select_chen_herb(self,
                         analysis: FormulaAnalysis,
                         gene_targets: List[str]) -> bool:
        """Select a safe Chen herb complementing the Jun herb"""
        if not analysis.herbs['jun']:
            raise ValueError("Cannot select Chen herb before Jun herb")

        herb_blacklist = self._get_herb_blacklist(analysis)
        name, context, reason = self.chen_agent.find_chen_herb(
            gene_list=gene_targets,
            patient_obj=analysis.patient,
            jun_herb_name=analysis.herbs['jun']['preferred_name'],
            blacklist_herbs=herb_blacklist,
            blacklist_cpds=self.global_blacklist['compounds']
        )

        if name == "NA":
            return False

        for _ in range(self.max_retries):
            herb_dict = context['results'][0]['init']
            is_safe, safety_reason = self.check_herb_safety(name, analysis)

            if is_safe:
                analysis.add_herb('chen', herb_dict, reason, (is_safe, safety_reason))
                print(f"==> Herb {name} added as Chen")
                return True

            analysis.add_rejection('chen', name, safety_reason)
            name, context, reason = self.jun_agent.choose_another_herb(name, safety_reason)

        return False

    def select_zuo_herb(self,
                        analysis: FormulaAnalysis) -> bool:
        """Select a safe Zuo herb addressing secondary symptoms"""
        if not (analysis.herbs['jun'] and analysis.herbs['chen']):
            raise ValueError("Cannot select Zuo herb before Jun and Chen herbs")

        herb_blacklist = self._get_herb_blacklist(analysis)
        name, context, reason = self.zuo_agent.find_zuo_herb(
            patient_obj=analysis.patient,
            jun_herb_name=analysis.herbs['jun']['preferred_name'],
            chen_herb_name=analysis.herbs['chen']['preferred_name'],
            blacklist_herbs=herb_blacklist
        )

        if name == "NA":
            return False

        for _ in range(self.max_retries):

            herb_dict = context['results'][0]['init']
            is_safe, safety_reason = self.check_herb_safety(name, analysis)

            if is_safe:
                analysis.add_herb('zuo', herb_dict, reason, (is_safe, safety_reason))
                print(f"==> Herb {name} added as Zuo")
                return True

            analysis.add_rejection('zuo', name, safety_reason)
            name, context, reason = self.jun_agent.choose_another_herb(name, safety_reason)

        return False

    def select_shi_herb(self,
                        analysis: FormulaAnalysis) -> bool:
        """Select a safe Shi herb to guide the formula"""
        if not all(analysis.herbs[role] for role in ['jun', 'chen', 'zuo']):
            raise ValueError("Cannot select Shi herb before Jun, Chen, and Zuo herbs")

        herb_blacklist = self._get_herb_blacklist(analysis)
        name, context, reason = self.shi_agent.select_shi_herb(
            patient_obj=analysis.patient,
            herb_names={
                "Jun": analysis.herbs['jun']['preferred_name'],
                "Chen": analysis.herbs['chen']['preferred_name'],
                "Zuo": analysis.herbs['zuo']['preferred_name']
            },
            blacklist_herbs=herb_blacklist
        )

        if name == "NA":
            return False

        for _ in range(self.max_retries):

            herb_dict = context['results'][0]['init']
            is_safe, safety_reason = self.check_herb_safety(name, analysis)

            if is_safe:
                analysis.add_herb('shi', herb_dict, reason, (is_safe, safety_reason))
                print(f"==> Herb {name} added as Shi")
                return True

            analysis.add_rejection('shi', name, safety_reason)
            name, context, reason = self.jun_agent.choose_another_herb(name, safety_reason)

        return False

    def design_formula(self,
                       patient: 'PatientProfile',
                       gene_targets: list[str]) -> FormulaAnalysis:
        """
        Design a complete formula for a patient while maintaining safety checks

        Returns:
            FormulaAnalysis object containing the complete formula design
        """
        analysis = FormulaAnalysis(patient)

        # Select herbs in sequence
        try:
            if not self.select_jun_herb(analysis, gene_targets):
                return analysis

            if not self.select_chen_herb(analysis, gene_targets):
                return analysis

            if not self.select_zuo_herb(analysis):
                return analysis

            if not self.select_shi_herb(analysis):
                return analysis

        except Exception as e:
            print(f"Formula design error: {str(e)}")

        return analysis

## Formula presenter
# - Formula/dosages/regimen/preparation
# - Warnings
# - Herb actions and total actions: meridians, temperture, taste
class FormulaPresenter(BaseAgent):

    base_prompt: str = f"""
        You are a TCM practitioner analyzing a herbal formula.
         
        Consult with databases to gain more understanding of the components of this formula. Present a md-formatted text 
        description of the formula to be shown to the patient.
        
        In your description of the TCM formula, provide a breakdown with the following sections and sub-sections:
        1. Description
        1.1 Contents *[skip detailed explanations]*
            - Form *[decoction, pill, powder, or something else]*
            - Herb pinyin name (Chinese hyerogliphics, common name): recommended weight and form of component *[for all ingredients]*
        1.2 Instructions
            - Single dose / Times per day
            - Before/after food, preferred time of the day
            - Maximal daily dose
            - Course duration 
        2. Actions
        2.1 TCM actions *[actions of this formula explained in TCM terms]*
            - Meridians *[primary and secondary meridians this formula affects]*
            - Temperature and taste
            - Actions *[primary and secondary effects on yin/yang/elements]*
            - Indications *[indications and conditions that this formula may help with based on TCM annotation of the herbs]*
        2.2 Pharmacological actions
            - Compounds *[key active ingredients encountered in this formula]*
            - Protein targets *[key proteins the herbs and compounds in this formula interact with]*
            - Pathways *[key metabolic pathways reported to be affected in this formula]*
            - Geroprotective effects *[potential anti-aging effects this formula may have based on its active 
                ingredients and affected genes/pathways]*
        2.3 Details *[how this formula is expected to interact with the patient, address the TCM imbalances in their body 
        and affect the aging process in them]*
        3. Warnings
            - Potential side effects, contraindications, allergies, safety notes in the context of the patient;
            - Diet *[certain foods and drinks that may alter the effects of the formula, according to their temperatures 
                and tastes. if any foods need to be avoided to maximize this formula's effect]*
        4. Notes *[any additional details that do not fit in this presentation format]*
        
        Consider the patient's specific needs, how various herbs can affect any preexisting conditions, 
        based on the following information from their personal physician: 
        
        """


## Formula Analyzer
# - most affected proteins [histogram]
# - are any pathways enriched (ENRICHER KG)
# - MoA — TCM style
# - MoA — Western style []
# - geroprotection
class FormulaAnalyzer(BaseAgent):

    base_prompt: str = "Your are a helpful assistant"

    def __init__(self,
                 llm,
                 tools=None,
                 completion_max_tries=3,
                 max_compounds=50,
                 max_herbs=10,
                 max_desc_chars=2000):

        if tools is None:
            tools = [find_TCM_herb]

        super().__init__(
            llm_options=llm,
            tools=[find_TCM_herb],
            completion_max_tries=completion_max_tries
        )

        self.blacklist = {
            "compounds": set(),
            "herbs": set()
        }

        self.max_compounds = max_compounds
        self.max_herbs = max_herbs
        self.max_desc_chars = max_desc_chars

        self.initialize()

    def initialize(self):
        self.instruct(self.base_prompt)
        self.enricher = EnrichrCaller

    def describe_formula(self,
                         herb_names: list(str),
                         gene_list: list[str]):
        report = dict()

        # Count top targts hit by herbs
        report['target_counts'] = get_herbal_targets(herb_names)

        # Herb pathway enrichment
        enran_herb = EnrichrAnalysis(report['target_counts'], EnrichrCaller)
        enran_herb.analyze()
        report["pathways_herbs"] = {"headers": enran_herb.legend, "metrics": enran_herb.res}

        # Gene list pathway enrichment
        enran_gene_list = EnrichrAnalysis(gene_list, EnrichrCaller)
        enran_gene_list.analyze()
        report["pathways_herbs"] = {"headers": enran_gene_list.legend, "metrics": enran_gene_list.res}

        # Compounds affecting the signature?

        #

