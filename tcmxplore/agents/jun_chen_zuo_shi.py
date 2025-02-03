# agents/jun_chen_zuo_shi.py
from dotenv import load_dotenv
load_dotenv()

import json
from dataclasses import dataclass, field, fields,  asdict
from typing import List, Dict, Optional, Tuple, Set, Any, Iterable, Union, OrderedDict
from json_repair import repair_json
from collections import defaultdict, Counter
import requests
from time import sleep
import re

from just_agents.base_agent import BaseAgent
from ..agents.tcm_crew import (TCMTools, find_TCM_herb, find_TCM_condition,
                             pick_cpds_by_targets, pick_herbs_by_targets,
                             pick_herbs_by_cids, get_description_from_cid,
                             get_compound_targets, get_description_from_cid_batch,
                             find_TCM_compound, TCMPropertyScorer, get_herbal_targets,
                             get_herbal_compounds, count_targets_in_compounds)
from ..agents.api_callers import  EnrichrAnalysis, EnrichrCaller, ChemblBulkAPI, CHEMBL_Annotation, PubChemAPI
from ..dragon_db.annots import HerbAnnotation, TCMAnnotation
from abc import ABC, abstractmethod
import logging
import asyncio


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
    @staticmethod
    def _parse_list(text: Union[str, List[str]]) -> List[str]:
        """Convert delimited text or list to clean list
        
        Handles multiple delimiter types:
        - Commas (,)
        - Semicolons (;)
        - Newlines (\n)
        - Bullet points (•)
        - Dashes (-)
        """
        # If already a list, clean each item
        if isinstance(text, list):
            items = text
        else:
            if not text or str(text).lower() == 'none':
                return []
                
            # Replace common delimiters with commas
            text = str(text).replace('\n', ',').replace(';', ',').replace('•', ',')
            
            # Split by commas
            items = text.split(',')
        
        # Clean up each item
        cleaned_items = []
        for item in items:
            # Strip whitespace
            item = str(item).strip()
            
            # Remove common list markers
            item = re.sub(r'^\s*[-*•]\s*', '', item)  # Remove bullet points
            item = re.sub(r'^\s*\d+\.\s*', '', item)  # Remove numbered lists
            
            if item:  # Only add non-empty items
                cleaned_items.append(item)
        
        return cleaned_items

    @classmethod
    def from_dict(cls, profile_data: Dict) -> 'PatientProfile':
        """Create PatientProfile from a dictionary"""
        return cls(
            age=profile_data.get('age', -1),
            sex=profile_data.get('sex', 'male'),

            conditions=cls._parse_list(profile_data.get('conditions', [])),
            allergies=cls._parse_list(profile_data.get('allergies', [])),
            medications=cls._parse_list(profile_data.get('medications', [])),
            risk_factors=cls._parse_list(profile_data.get('risk_factors', [])),

            goals=cls._parse_list(profile_data.get('goals', []))
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

    def __init__(self, llm_options: dict, completion_max_tries=3):

        super().__init__(llm_options=llm_options,
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
        print(f"Submitting a safety analysis prompt with {len(joined_prompt)} characters")
        try:
            response = self.query(joined_prompt, timeout=60)
        except Exception as e:
            print(f"Error submitting safety analysis prompt: {e}")
            # Propagate the error with more context
            raise RuntimeError(f"Failed to analyze herb safety for {herb_name}: {str(e)}") from e

        try:
            result = json.loads(repair_json(response))
            return result["is_safe"], result["reason"]
        except (json.JSONDecodeError, KeyError) as e:
            # Propagate parsing errors as well
            raise RuntimeError(f"Failed to parse safety analysis response for {herb_name}: {str(e)}") from e


## Herb selection
class BaseTCMAgent(BaseAgent):
    """Base class for TCM herb selection agents"""

    base_prompt: str = "Your are a helpful assistant"

    def __init__(self,
                 llm_options,
                 tools=None,
                 completion_max_tries=3,
                 max_compounds=50,
                 max_herbs=10,
                 max_desc_chars=2000):
        if tools is None:
            tools = [find_TCM_herb]
        super().__init__(
            llm_options=llm_options,
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
                           thr: float=0.01,
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
                              f"\nSuggest a safe alternative herb among the preselected options "
                              f"using the same JSON output format")

        parsed_resp = self.parse_response(response)
        return(parsed_resp)

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

        best_cpds = self.enriched_compounds(gene_list, blacklist=blacklist_cpds, thr=0.01)

        # Get herb candidates based on compounds
        best_herbs1 = pick_herbs_by_cids(best_cpds, N_top=10, blacklist = blacklist_herbs)
        if 'No herbs found' in best_herbs1:
            best_herbs1.pop('No herbs found')

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

        blacklist_herbs = set(blacklist_herbs) | {find_TCM_herb(jun_herb_name).get('preferred_name', "")}

        # Get Jun herb information
        jun_herb_batman = find_TCM_herb(jun_herb_name, db_id='BATMAN')
        jun_herb_dragon = find_TCM_herb(jun_herb_name)
        jun_cpd_list = jun_herb_batman['results'][0]['links']['ingrs']
        jun_cpd_set = set(jun_cpd_list)

        enriched_cpds = self.enriched_compounds(gene_list, blacklist = blacklist_cpds)
        # only keep relevant compounds
        jun_cpds = {cid: stats for cid, stats in enriched_cpds.items() if cid in jun_cpd_set}

        # Find herbs sharing compounds with Jun herb
        common_cpds = self.herb_common_compound_counts(jun_cpds, blacklist = blacklist_herbs)

        best_herbs = dict(sorted(
            common_cpds.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_herbs])        
        
        # If no herbs found through compound sharing, try direct target-based approach
        if not best_herbs:
            print("No herbs found through compound sharing, trying target-based approach...")
            
            # Get Jun herb's target genes
            jun_targets = set(jun_herb_dragon.get('targets', {}).get('symbols', []))
            
            # Find herbs targeting similar genes
            target_herbs = pick_herbs_by_targets(
                gene_list=set(gene_list) | jun_targets,  # Include both disease genes and Jun herb targets
                tg_type='both',  # Consider both known and predicted targets
                top_herbs=0.05,  # Be more lenient with the cutoff
                blacklist=blacklist_herbs
            )
            
            if 'No herbs found' in target_herbs:
                target_herbs.pop('No herbs found')
                
            best_herbs = dict(sorted(
                target_herbs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.max_herbs])
            
            if not best_herbs:
                print("No herbs found through target-based approach either.")
                return "", {}, "No suitable Chen herbs found through compound sharing or target analysis."

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
        # [!]: somehow this line generated a formula wiht ×3 DNAG GUI, not sure how exactly but just in case
        # adding an extra herb look up
        # blacklist_herbs = set(blacklist_herbs) | {jun_herb_name, chen_herb_name}
        blacklist_herbs = set(blacklist_herbs) | {find_TCM_herb(x).get('preferred_name', "") for x in (jun_herb_name, chen_herb_name)}

        jun_found = find_TCM_herb(jun_herb_name)
        chen_found = find_TCM_herb(chen_herb_name)

        # Get existing herb information using tool
        jun_info = jun_found['results'][0]
        chen_info = chen_found['results'][0]
        
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

            # [!]: Need a better way to assign scores since exact matches are very rare
            # print(f" ===> Analyzed {herb_name} - Coverage score: {coverage['coverage_score']}, Compatibility issues: {len(compatibility)}")

            herb_entries[herb_name] = {
                'info': herb_entry,
                'analysis': {
                    'symptom_coverage': coverage,
                    'compatibility_issues': compatibility
                }
            }
        
        if not herb_entries:
            print("No valid herb entries after analysis")

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
                if isinstance(action, dict) and 'action' in action:
                    actions.add(action['action'].lower())
                if isinstance(action, dict) and 'indication' in action:
                    actions.add(action['indication'].lower())

        # Extract from treatment indications
        if 'indications' in herb_info['init']:
            actions.update(cond.lower() for cond in herb_info['init']['indications'])

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
        # [!]: somehow this line generated a formula wiht ×3 DNAG GUI, not sure how exactly but just in case
        # adding an extra herb look up
        # blacklist_herbs = set(blacklist_herbs) | set(herb_names.values())
        blacklist_herbs = set(blacklist_herbs) | {find_TCM_herb(x).get('preferred_name', "") for x in herb_names.values()}

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

    def __init__(self, 
                 llm_options: dict):
        # Initialize sub-agents
        
        self.contraindication_agent = ContraindicationAgent(llm_options=llm_options)
        self.jun_agent = JunHerbAgent(llm_options=llm_options)
        self.chen_agent = ChenHerbAgent(llm_options=llm_options)
        self.zuo_agent = ZuoHerbAgent(llm_options=llm_options)
        self.shi_agent = ShiHerbAgent(llm_options=llm_options)

        # Define global blacklists
        self.global_blacklist = {
            'compounds': {
                            5462328,  # Heroin
                            175,  # Acetate
                            241, # Benzene
                            702, # Ethanol
                            8857, # Ethyl acetate
                            243, # Benzoic acid
                            7501, # Styrene
                            931, # Naphatalene
                            44630435, # NA
                            887, # Methanol
                            177, # Acetaldehyde
                            284, # Formic acid
                        },
            'herbs': {"GOU SHEN", 'WU GONG','ZHI CAO WU',
                      'MENG CHONG', 'HA MA YOU', 'XIONG DAN',
                      'SHE XIANG', 'ZI HE CHE', 'DONG CHONG XIA CAO',
                      'LU RONG', 'JIU', 'REN NIAO', "HOU ZAO"}
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
        self.current_analysis = None

        

    def _get_herb_blacklist(self) -> Set[str]:
        """Combine global blacklist with patient-specific contraindications"""
        blacklist = self.global_blacklist['herbs'].copy()

        # Add herbs rejected for this patient
        for rejected in self.current_analysis.rejected_herbs.values():
            blacklist |= set([herb for herb, _ in rejected])

        # Could expand here to add patient-specific contraindications
        # based on their conditions, medications, etc.

        return blacklist

    def check_herb_safety(self, herb_name: str) -> tuple[bool, str]:
        """Check if an herb is safe for the current patient"""
        try:
            is_safe = self.contraindication_agent.evaluate_safety(
                herb_name=herb_name,
                patient_obj=self.current_analysis.patient
            )
            return is_safe

        except Exception as e:
            # Log the error and propagate it
            print(f"Error checking safety for {herb_name}: {e}")
            raise
        
    def select_jun_herb(self,
                        gene_targets: List[str]) -> bool:
        """Select a safe Jun herb targeting specified genes"""

        herb_blacklist = self._get_herb_blacklist()
        name, context, reason = self.jun_agent.find_jun_herb(
            gene_list=gene_targets,
            patient_obj=self.current_analysis.patient,
            blacklist_herbs=herb_blacklist,
            blacklist_cpds=self.global_blacklist['compounds']
        )

        assert not name in herb_blacklist, f"Herb {name} found in blacklist!"
        if name == "NA":
            return False

        for _ in range(self.max_retries):
            
            herb_dict = context['results'][0]['init']
            is_safe, safety_reason = self.check_herb_safety(name)

            if is_safe:
                self.current_analysis.add_herb('jun', herb_dict, reason, (is_safe, safety_reason))
                print(f"==> Herb {name} added as Jun")
                return True

            self.current_analysis.add_rejection('jun', name, safety_reason)
            print(f"Herb {name} found not safe for patient")
            name, context, reason = self.jun_agent.choose_another_herb(name, safety_reason)

        return False

    def select_chen_herb(self,
                         gene_targets: List[str]) -> bool:
        """Select a safe Chen herb complementing the Jun herb"""
        if not self.current_analysis.herbs['jun']:
            raise ValueError("Cannot select Chen herb before Jun herb")

        herb_blacklist = self._get_herb_blacklist()
        
        name, context, reason = self.chen_agent.find_chen_herb(
            gene_list=gene_targets,
            patient_obj=self.current_analysis.patient,
            jun_herb_name=self.current_analysis.herbs['jun']['preferred_name'],
            blacklist_herbs=herb_blacklist,
            blacklist_cpds=self.global_blacklist['compounds']
        )
        print(f"====> {name}")
        if name == "NA" or context == "NA" or name == "Not applicable":
            print(f"Failed to find suitable Chen herb: {reason}")
            return False

        for _ in range(self.max_retries):
            try:
                herb_dict = context['results'][0]['init']
                is_safe, safety_reason = self.check_herb_safety(name)

                if is_safe:
                    self.current_analysis.add_herb('chen', herb_dict, reason, (is_safe, safety_reason))
                    print(f"==> Herb {name} added as Chen")
                    return True

                self.current_analysis.add_rejection('chen', name, safety_reason)
                name, context, reason = self.chen_agent.choose_another_herb(name, safety_reason)  # Fixed: was using jun_agent
                
                if name == "NA" or context == "NA" or name == "Not applicable":
                    print(f"Failed to find alternative Chen herb: {reason}")
                    return False
                    
            except (KeyError, IndexError) as e:
                print(f"Error processing herb {name}: {str(e)}")
                return False

        return False

    def select_zuo_herb(self) -> bool:
        """Select a safe Zuo herb addressing secondary symptoms"""
        if not (self.current_analysis.herbs['jun'] and self.current_analysis.herbs['chen']):
            raise ValueError("Cannot select Zuo herb before Jun and Chen herbs")

        herb_blacklist = self._get_herb_blacklist()
        name, context, reason = self.zuo_agent.find_zuo_herb(
            patient_obj=self.current_analysis.patient,
            jun_herb_name=self.current_analysis.herbs['jun']['preferred_name'],
            chen_herb_name=self.current_analysis.herbs['chen']['preferred_name'],
            blacklist_herbs=herb_blacklist
        )

        if name == "NA":
            return False

        for _ in range(self.max_retries):

            herb_dict = context['results'][0]['init']
            is_safe, safety_reason = self.check_herb_safety(name)

            if is_safe:
                self.current_analysis.add_herb('zuo', herb_dict, reason, (is_safe, safety_reason))
                print(f"==> Herb {name} added as Zuo")
                return True

            self.current_analysis.add_rejection('zuo', name, safety_reason)
            name, context, reason = self.jun_agent.choose_another_herb(name, safety_reason)

        return False

    def select_shi_herb(self) -> bool:
        """Select a safe Shi herb to guide the formula"""
        if not all(self.current_analysis.herbs[role] for role in ['jun', 'chen', 'zuo']):
            raise ValueError("Cannot select Shi herb before Jun, Chen, and Zuo herbs")

        herb_blacklist = self._get_herb_blacklist()
        name, context, reason = self.shi_agent.select_shi_herb(
            patient_obj=self.current_analysis.patient,
            herb_names={
                "Jun": self.current_analysis.herbs['jun']['preferred_name'],
                "Chen": self.current_analysis.herbs['chen']['preferred_name'],
                "Zuo": self.current_analysis.herbs['zuo']['preferred_name']
            },
            blacklist_herbs=herb_blacklist
        )

        if name == "NA":
            return False

        for _ in range(self.max_retries):

            herb_dict = context['results'][0]['init']
            is_safe, safety_reason = self.check_herb_safety(name)

            if is_safe:
                self.current_analysis.add_herb('shi', herb_dict, reason, (is_safe, safety_reason))
                print(f"==> Herb {name} added as Shi")
                return True

            self.current_analysis.add_rejection('shi', name, safety_reason)
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
        self.current_analysis = FormulaAnalysis(patient)
        logger = logging.getLogger(__name__)

        # Select herbs in sequence
        try:
            logger.info("Starting Jun herb selection...")
            if not self.select_jun_herb(gene_targets):
                return self.current_analysis
            logger.info(f"==> Herb {self.current_analysis.herbs['jun']['preferred_name']} added as Jun")

            logger.info("Starting Chen herb selection...")
            if not self.select_chen_herb(gene_targets):
                return self.current_analysis
            logger.info(f"==> Herb {self.current_analysis.herbs['chen']['preferred_name']} added as Chen")


            logger.info("Starting Zuo herb selection...")
            if not self.select_zuo_herb():
                return self.current_analysis
            logger.info(f"==> Herb {self.current_analysis.herbs['zuo']['preferred_name']} added as Zuo")

            logger.info("Starting Shi herb selection...")
            if not self.select_shi_herb():
                return self.current_analysis
            logger.info(f"==> Herb {self.current_analysis.herbs['shi']['preferred_name']} added as Shi")

        except Exception as e:
            print(f"[.design_formula] Formula design error: {str(e)}")

        return self.current_analysis
    

@dataclass
class FormulaReport:
    """Container for formula analysis results"""
    herb_properties: Dict[str, Dict[str, Any]]
    target_statistics: Dict[str, int]
    compound_activities: Dict[int, Dict[str, Any]]
    pathway_enrichment: Dict[str, Any]
    formula_properties: Dict[str, float]
    description: str
    errors: List[str]

    def to_dict(self) -> Dict:
        """Convert result object to dictionary format."""
        return asdict(self)

class FormulaAnalyzer(BaseAgent):
    """Analyzes TCM formulas combining molecular and traditional perspectives"""

    base_prompt: str = """You are a TCM expert analyzing herbal formulas.
    Your task is to provide comprehensive analysis of formula components,
    their molecular mechanisms, and traditional properties."""

    def __init__(self,
                 llm_options,
                 tools=None,
                 completion_max_tries: int = 3,
                 max_compounds: int = 50,
                 max_herbs: int = 10,
                 max_desc_chars: int = 2000):

        if tools is None:
            tools = [find_TCM_herb, find_TCM_compound]

        super().__init__(
            llm_options=llm_options,
            tools=tools,
            completion_max_tries=completion_max_tries
        )

        self.blacklist = {
            "compounds": {
                            5462328,  # Heroin
                            175,  # Acetate
                            241, # Benzene
                            702, # Ethanol
                            8857, # Ethyl acetate
                            243, # Benzoic acid
                            7501, # Styrene
                            931, # Naphatalene
                            44630435, # NA
                            887, # Mathanol
                            177, # Acetaldehyde
                        },
            "herbs": {"GOU SHEN", 'WU GONG','ZHI CAO WU',
                      'MENG CHONG', 'HA MA YOU', 'XIONG DAN',
                      'SHE XIANG', 'ZI HE CHE', 'DONG CHONG XIA CAO',
                      'LU RONG', 'JIU', 'REN NIAO',

                       'FU ZI', 'WU TOU', 'XI XIN', 'MA HUANG',
                       'YANG JIN HUA', 'LEI GONG TENG', 'WU GONG', 'QUAN XIE',
                       'BAI HUA SHE', 'MANG CHONG', 'ZHE CHONG', 'SHAN DOU GEN',
                       'BAN XIA', 'TIAN NAN XING', 'BAI FU ZI', 'WEI LING XIAN',
                       'XIAN MAO', 'WU ZHU YU', 'HUA JIAO', 'YUAN ZHI',
                       'KU LIAN PI', 'HE SHI', 'GUA DI', 'LI LU',
                       'CHANG SHAN', 'GAN SUI', 'DA JI', 'YUAN HUA',
                       'SHANG LU', 'QIAN NIU ZI', 'BA DOU', 'ZHU SHA'}

        }

        self.max_compounds = max_compounds
        self.max_herbs = max_herbs
        self.max_desc_chars = max_desc_chars

        self.initialize()

    def initialize(self):
        """Initialize analysis tools"""
        self.instruct(self.base_prompt)
        self.enricher = EnrichrAnalysis
        self.pubchem = PubChemAPI()
        self.chembl = ChemblBulkAPI()
        self.property_scorer = TCMPropertyScorer()

    def get_formula_tcm_properties(self,
                                   herb_names: List[str]) -> Dict[str, float]:
        """Calculate aggregate TCM properties for the formula"""
        properties = {
            'temperature_score': 0.0,
            'taste_balance': 0.0,
            'meridian_coverage': 0.0
        }

        try:
            herb_details = {}
            for herb in herb_names:
                result = find_TCM_herb(herb)
                if result['results']:
                    herb_details[herb] = result['results'][0]['init'].get('properties', {})

            if not herb_details:
                return properties

            # Temperature scoring
            temp_values = {'Hot': 2, 'Warm': 1, 'Neutral': 0, 'Cool': -1, 'Cold': -2}
            temps = []
            for herb_props in herb_details.values():
                if 'temperature' in herb_props:
                    primary_temp = herb_props['temperature'].get('primary', ['Neutral'])[0]
                    temps.append(temp_values.get(primary_temp, 0))
            properties['temperature_score'] = sum(temps) / len(temps) if temps else 0

            # Taste balance (diversity of tastes)
            all_tastes = []
            for herb_props in herb_details.values():
                if 'taste' in herb_props:
                    all_tastes.extend(herb_props['taste'].get('primary', []))
            taste_counts = Counter(all_tastes)
            max_taste_count = max(taste_counts.values()) if taste_counts else 1
            properties['taste_balance'] = len(taste_counts) / max_taste_count if max_taste_count > 0 else 0

            # Meridian coverage
            all_meridians = set()
            for herb_props in herb_details.values():
                if 'meridians' in herb_props:
                    all_meridians.update(herb_props['meridians'].get('primary', []))
            properties['meridian_coverage'] = len(all_meridians) / 12  # 12 regular meridians

        except Exception as e:
            print(f"Error calculating TCM properties: {str(e)}")

        return properties

    def analyze_compounds(self,
                          herb_names: List[str],
                          gene_list: List[str],
                          max_compounds: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
        """Analyze compounds in herbs and their biological activities"""
        max_compounds = max_compounds or self.max_compounds
        compound_info = {}

        try:
            # Get compounds from herbs
            fla_cpds = get_herbal_compounds(herb_names)
            fla_cpds = set(fla_cpds) - self.blacklist['compounds'] # set - dict?

            # Count target hits
            hit_tgs_per_cpd = count_targets_in_compounds(
                list(fla_cpds),
                gene_list,
                N_top=max_compounds
            )

            if not hit_tgs_per_cpd:
                return compound_info

            # Get compound metadata
            cid_list = list(hit_tgs_per_cpd.keys())
            chembl_ids = self.pubchem.get_chembl_ids_batch(cid_list)
            pubchem_descs = self.pubchem.get_descriptions_batch(cid_list)
            names = self.pubchem.get_names_batch(cid_list)

            # Get CHEMBL annotations
            if chembl_ids:
                chembl_annots = self.chembl.create_annotations(list(chembl_ids.values()))
                for cid, anno in zip(cid_list, chembl_annots):
                    if cid in pubchem_descs:
                        anno.add_desc(
                            name=names.get(cid, ""),
                            desc=pubchem_descs[cid][0] if pubchem_descs[cid] else ""
                        )
                    compound_info[cid] = {
                        'name': names.get(cid, "Unknown"),
                        'description': pubchem_descs.get(cid, []),
                        'targets_hit': hit_tgs_per_cpd[cid],
                        'activities': anno.activities if anno else [],
                        'indications': anno.indications if anno else []
                    }

        except Exception as e:
            print(f"Error analyzing compounds: {str(e)}")

        return compound_info

    def analyze_pathways(self,
                         gene_list: List[str],
                         p_value_threshold: float = 0.05,
                         min_overlap: int = 2,
                         max_terms: int = 30) -> Dict[str, Any]:
        """
        Analyze pathway enrichment for a list of genes

        Args:
            gene_list: List of genes to analyze
            p_value_threshold: P-value cutoff for significant pathways
            min_overlap: Minimum number of genes that must overlap with a pathway
            max_terms: Maximum number of enriched terms to return

        Returns:
            Dictionary containing:
                - legend: Column descriptions for pathway results
                - significant_terms: List of significant pathways
                - summary_stats: Basic statistics about enrichment
                - error: Error message if analysis failed
        """
        pathway_info = {
            'legend': self.enricher.legend,
            'significant_terms': [],
            'summary_stats': {
                'total_genes_analyzed': len(gene_list),
                'genes_mapped_to_pathways': 0,
                'significant_pathways': 0
            },
            'error': None
        }

        try:
            # Initialize and run analysis
            analysis = self.enricher(gene_list)
            analysis.analyze()

            if not analysis.results:
                pathway_info['error'] = "No enrichment results returned"
                return pathway_info

            # Process results for the first library (e.g. GO or KEGG)
            library_name = list(analysis.results.keys())[0]
            all_results = analysis.results[library_name]

            # Filter and format significant pathways
            significant = []
            genes_in_pathways = set()

            for result in all_results:
                if len(result) < 6:  # Skip malformed results
                    continue

                # Extract data
                term_name = result[1]
                p_value = float(result[2])
                odds_ratio = float(result[3])
                genes = result[5] if isinstance(result[5], list) else [result[5]]

                # Apply filters
                if p_value > p_value_threshold or len(genes) < min_overlap:
                    continue

                significant.append({
                    'term': term_name,
                    'p_value': p_value,
                    'odds_ratio': odds_ratio,
                    'genes': genes
                })
                genes_in_pathways.update(genes)

            # Sort by p-value and limit number of terms
            significant.sort(key=lambda x: x['p_value'])
            pathway_info['significant_terms'] = significant[:max_terms]

            # Update summary stats
            pathway_info['summary_stats'].update({
                'genes_mapped_to_pathways': len(genes_in_pathways),
                'significant_pathways': len(significant)
            })

        except Exception as e:
            pathway_info['error'] = str(e)

        return pathway_info

    def generate_report(self,
                        formula_data: Dict[str, Any]) -> str:
        """Generate natural language report from analysis data"""

        """
           Generate comprehensive TCM formula report with molecular and traditional perspectives
    
           Args:
               formula_data: Dictionary containing formula analysis results
               patient_info: Optional dictionary with patient details and conditions
    
           Returns:
               Markdown-formatted report string
           """

        # Format patient context if provided

        report_prompt = f"""
        You are a TCM practitioner preparing a detailed yet accessible explanation of a herbal formula for your patient.
        Generate a comprehensive report that helps them understand and properly use their medication.

        Base your analysis on this information aggregate from statistical analysis, TCM reference books, and patient 
        anamnesis:
       {json.dumps(formula_data, indent=4)}

        Structure your report in a patient-friendly format with these sections:

        1. Your Formula Overview
           1.1 Quick Summary
               - Purpose: Main benefits and treatment goals for your specific condition
               - Format: How your medicine is prepared (decoction/pill/powder)
               - Duration: How long you should take this formula
               - Expected outcomes: What improvements you may notice and when

           1.2 Components (Your Herbs)
               - For each herb: Pinyin name (Chinese name, common name)
               - Specific amounts and forms for your case
               - Role of each herb in addressing your conditions

           1.3 Usage Instructions
               - Daily dosage schedule (when and how much to take)
               - Best timing (before/after meals, morning/evening)
               - Maximum daily amount
               - How to prepare and store your medicine
               - Course duration and what to expect

        2. How Your Formula Works
           2.1 Traditional Understanding
               - Which energy channels (meridians) are being treated
               - How this formula balances your body's elements
               - Why these specific herbs were chosen for you (add any notes on their taste / temperature)
               - What conditions and symptoms it helps with

           2.2 Modern Scientific Perspective (where applicable, add information on particular protein targets and affected 
           biological pathways)
               - Key beneficial compounds in your formula (specify 3-4 key compounds and their sources)
               - How these ingredients work together (rely on the presented knwoledge of pathways and genes)
               - Major health-promoting effects observed in research (based on reported activities and pathways)
               - Anti-aging and geroprotective benefits (based on your knowledge of aging-related processes and the formula's molecular context)

           2.3 Your Personalized Benefits
               - How this formula addresses your specific conditions
               - Long-term health benefits
               - Signs that the formula is working

        3. Important Safety Information
           - Specific precautions for your case
           - Possible side effects to watch for
           - When to contact your practitioner
           - Dietary Guidelines:
             * Foods that enhance your treatment
             * Foods to moderate or avoid
             * Best times for meals with your medicine

        4. Support Notes
           - Tips for consistent usage
           - How to track your progress
           - When to schedule follow-up
           - Additional lifestyle recommendations

        Important guidelines:
        1. Write in clear, friendly language while maintaining professionalism
        2. Make all instructions specific and actionable
        3. Explain technical concepts through practical examples
        4. Focus on the patient's particular conditions and goals
        5. Provide concrete ways to measure progress
        6. Include supportive lifestyle recommendations
        7. Format numbers clearly and consistently
        8. Use bullet points and headers for easy reference, in accordance with Discord markdown parser
        9. Highlight crucial safety information
        10. Include positive reinforcement and encouragement
        11. Add practical tips for treatment compliance
        12. Note what feedback to watch for and report
        """

        return self.query(report_prompt)

    def describe_formula(self,
                         herb_names: List[str],
                         gene_list: List[str],
                         patient_obj: 'PatientProfile' = None) -> FormulaReport:
        """Generate comprehensive formula analysis"""

        # Initialize report container
        report = FormulaReport(
            herb_properties={},
            target_statistics={},
            compound_activities={},
            pathway_enrichment={},
            formula_properties={},
            description="",
            errors=[]
        )

        try:
            # Get individual herb properties
            for herb in herb_names:
                result = find_TCM_herb(herb)
                if result['results']:
                    report.herb_properties[herb] = result['results'][0]['init']

            # Get formula-level TCM properties
            report.formula_properties = self.get_formula_tcm_properties(herb_names)

            # Analyze molecular targets
            report.target_statistics = get_herbal_targets(herb_names, top_N=100)

            # Analyze compounds and their activities

            report.compound_activities = self.analyze_compounds(
                herb_names,
                gene_list,
                max_compounds=self.max_compounds
            )

            # Analyze pathway enrichment
            herb_targets = get_herbal_targets(herb_names, top_N=100)
            report.pathway_enrichment['herb_pathways'] = self.analyze_pathways(
                list(herb_targets.keys()),
                p_value_threshold=0.1,  # More permissive
                min_overlap=2
            )

            sleep(0.87)

            # Analyze pathways for input signature (stricter threshold)
            report.pathway_enrichment['signature_pathways'] = self.analyze_pathways(
                gene_list,
                p_value_threshold=0.05,  # Stricter
                min_overlap=2
            )

            # Generate natural language report
            formula_data = {
                'herb_properties': report.herb_properties,
                'formula_properties': report.formula_properties,
                'target_statistics': report.target_statistics,
                'compound_activities': report.compound_activities,
                'pathway_enrichment': report.pathway_enrichment,
                'patient_info': patient_obj.to_json()
            }

            report.description = self.generate_report(formula_data)

        except Exception as e:
            report.errors.append(f"Error in formula analysis: {str(e)}")

        return report
