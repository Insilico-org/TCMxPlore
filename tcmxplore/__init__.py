# Main package imports
from .connector.connector import BatmanDragonConnector, DBConnector

# Agent system
from .agents.tcm_crew import (
    TCMTools,
    find_TCM_herb,
    find_TCM_condition,
    find_TCM_formula,
    find_TCM_compound,
    pick_cpds_by_targets,
    pick_herbs_by_targets,
    pick_herbs_by_cids,
    get_herbal_compounds,
    get_herbal_targets,
    count_targets_in_compounds
)

from .agents.jun_chen_zuo_shi import (
    FormulaDesignAgent,
    PatientProfile,
    TCMPropertyScorer,
    FormulaAnalyzer,
    FormulaAnalysis
)

# Database models
from .batman_db.batman import (
    TCMDB,
    TCMEntity,
    Ingredient,
    Herb,
    Formula
)

from .dragon_db.annots import (
    TCMAnnotationDB,
    TCMAnnotation,
    HerbAnnotation,
    ConditionAnnotation,
    FormulaAnnotation
)

# API utilities
from .agents.api_callers import (
    EnrichrAnalysis,
    PubChemAPI,
    ChemblBulkAPI,
    CHEMBL_Annotation
)

# Version info
__version__ = "0.1.0"

# Define public API
__all__ = [
    # Main connectors
    "BatmanDragonConnector",
    "DBConnector",
    
    # Core TCM tools
    "TCMTools",
    "find_TCM_herb",
    "find_TCM_condition", 
    "find_TCM_formula",
    "find_TCM_compound",
    
    # Selection functions
    "pick_cpds_by_targets",
    "pick_herbs_by_targets",
    "pick_herbs_by_cids",
    "get_herbal_compounds",
    "get_herbal_targets",
    "count_targets_in_compounds",
    
    # Formula design system
    "FormulaDesignAgent",
    "PatientProfile",
    "TCMPropertyScorer",
    "FormulaAnalyzer",
    "FormulaAnalysis",
    
    # Database models
    "TCMDB",
    "TCMEntity",
    "Ingredient", 
    "Herb",
    "Formula",
    "TCMAnnotationDB",
    "TCMAnnotation",
    "HerbAnnotation",
    "ConditionAnnotation",
    "FormulaAnnotation",
    
    # API utilities
    "EnrichrAnalysis",
    "PubChemAPI", 
    "ChemblBulkAPI",
    "CHEMBL_Annotation"
]