from string import Template

PREFIXES = """
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    
    PREFIX purl:<http://purl.org/dc/terms/>
    PREFIX dcat:<http://www.w3.org/ns/dcat#>
    PREFIX prov:<http://www.w3.org/ns/prov#>
    PREFIX mls:<http://www.w3.org/ns/mls#>
    PREFIX mlso:<http://w3id.org/mlso/>
    PREFIX mlso_tt:<http://w3id.org/mlso/vocab/ml_task_type#>
    PREFIX sd:<https://w3id.org/okn/o/sd#>
    
    PREFIX mlsea_openml_dataset:<http://w3id.org/mlsea/openml/dataset/>
    PREFIX mlsea_openml_task:<http://w3id.org/mlsea/openml/task/>
    PREFIX mlsea_openml_flow:<http://w3id.org/mlsea/openml/flow/>
    PREFIX mlsea_openml_run:<http://w3id.org/mlsea/openml/run/>
    
"""

RETRIEVE_TASK_ID_FOR_RUN_ID = Template("""
    $PREFIXES
    SELECT
        (?id AS ?task_id)
    WHERE {
        BIND(mlsea_openml_run:$runId AS ?r)
        ?r a mls:Run .

        ?r mls:achieves ?t .
        ?t a mls:Task .
        ?t purl:identifier ?id .
    } LIMIT 1
    """)

RETRIEVE_DATASET_ID_FOR_TASK_ID = Template("""
    $PREFIXES
    SELECT
        (?id AS ?dataset_id)
    WHERE {
        BIND(mlsea_openml_task:$taskId AS ?t)
        ?t a mls:Task .
        
        ?t mls:definedOn ?ds .
        ?ds a dcat:Dataset .
        ?ds purl:identifier ?id .
    } LIMIT 1
    """)

RETRIEVE_BATCHED_DATASETS_FROM_OPENML = Template("""
    $PREFIXES
    SELECT
        (?ds AS ?mlsea_dataset_uri)
        (?id AS ?openml_dataset_id)
        (?d_url AS ?openml_dataset_url)
        ?title
        (?dtf_title AS ?default_target_feature_label)
    WHERE {
        ?ds a dcat:Dataset .
        FILTER CONTAINS(str(?ds), "openml") .
        
        ?ds purl:identifier ?id .
        ?ds purl:title ?title .

        ?ds dcat:distribution ?d .
        ?d a dcat:Distribution .
        ?d dcat:downloadURL ?d_url .
        ?d mlso:hasDefaultTargetFeature ?dtf .
        ?dtf purl:title ?dtf_title .
        
        FILTER (?id > $offsetId)
    } ORDER BY ?id
    LIMIT $limit
    """)

RETRIEVE_ALL_DATASETS_FROM_OPENML = Template("""
    $PREFIXES    
    SELECT
        (?ds AS ?mlsea_dataset_uri)
        (?id AS ?openml_dataset_id)
        (?d_url AS ?openml_dataset_url)
        ?title
        (?dtf_title AS ?default_target_feature_label)
    WHERE {
        ?ds a dcat:Dataset .
        FILTER CONTAINS(str(?ds), "openml") .
        
        ?ds purl:identifier ?id .
        ?ds purl:title ?title .

        ?ds dcat:distribution ?d .
        ?d a dcat:Distribution .
        ?d dcat:downloadURL ?d_url .
        ?d mlso:hasDefaultTargetFeature ?dtf .
        ?dtf purl:title ?dtf_title .
    } ORDER BY ?id
    """)

RETRIEVE_DATASET_FROM_OPENML = Template("""
    $PREFIXES
    SELECT
        (?ds AS ?mlsea_dataset_uri)
        (?id AS ?openml_dataset_id)
        (?d_url AS ?openml_dataset_url)
        ?title
        (?dtf_title AS ?default_target_feature_label) 
    WHERE {
        BIND(mlsea_openml_dataset:$datasetId AS ?ds)

        ?ds a dcat:Dataset .
        FILTER CONTAINS(str(?ds), "openml") .

        ?ds purl:identifier ?id .
        ?ds purl:title ?title .

        ?ds dcat:distribution ?d .
        ?d a dcat:Distribution .
        ?d dcat:downloadURL ?d_url .
        ?d mlso:hasDefaultTargetFeature ?dtf .
        ?dtf purl:title ?dtf_title .
    } ORDER BY ?id
    """)

RETRIEVE_DATASETS_FROM_OPENML = Template("""
    $PREFIXES
    SELECT
        (?ds AS ?mlsea_dataset_uri)
        (?id AS ?openml_dataset_id)
        (?d_url AS ?openml_dataset_url)
        ?title
        (?dtf_title AS ?default_target_feature_label) 
    WHERE {
        VALUE ?ds { $datasetUris }

        ?ds a dcat:Dataset .
        FILTER CONTAINS(str(?ds), "openml") .

        ?ds purl:identifier ?id .
        ?ds purl:title ?title .

        ?ds dcat:distribution ?d .
        ?d a dcat:Distribution .
        ?d dcat:downloadURL ?d_url .
        ?d mlso:hasDefaultTargetFeature ?dtf .
        ?dtf purl:title ?dtf_title .
    } ORDER BY ?id
    """)

RETRIEVE_ALL_TASKS_FROM_OPENML_FOR_DATASET = Template("""
    $PREFIXES
    SELECT 
        (?t AS ?mlsea_task_uri)
        (?id AS ?openml_task_id)
        (?t_url AS ?openml_task_url)
        ?title
        (?tt AS ?task_type)
        (?ept AS ?evaluation_procedure_type)
    WHERE {
        BIND(mlsea_openml_dataset:$datasetId AS ?ds)

        ?t a mls:Task .
        ?t mls:definedOn ?ds .

        ?t purl:identifier ?id .
        ?t mlso:hasTaskType ?tt . 
        ?t purl:title ?title .
        ?t prov:atLocation ?t_url .

        OPTIONAL {
            ?es a mls:EvaluationSpecification .
            ?es mls:defines ?t .
            ?es mls:hasPart ?ep .
            ?ep mlso:hasEvaluationProcedureType ?ept .
        }
    } ORDER BY ?id
    """)

RETRIEVE_BATCHED_TASKS_FROM_OPENML_FOR_DATASET = Template("""
    $PREFIXES
    SELECT 
        (?t AS ?mlsea_task_uri)
        (?id AS ?openml_task_id)
        (?t_url AS ?openml_task_url)
        ?title
        (?tt AS ?task_type)
        (?ept AS ?evaluation_procedure_type)
    WHERE {
        BIND(mlsea_openml_dataset:$datasetId AS ?ds)

        ?t a mls:Task .
        ?t mls:definedOn ?ds .

        ?t purl:identifier ?id .
        ?t mlso:hasTaskType ?tt . 
        ?t purl:title ?title .
        ?t prov:atLocation ?t_url .

        OPTIONAL {
            ?es a mls:EvaluationSpecification .
            ?es mls:defines ?t .
            ?es mls:hasPart ?ep .
            ?ep mlso:hasEvaluationProcedureType ?ept .
        }
        
        FILTER (?id > $offsetId)
    } ORDER BY ?id
    LIMIT $limit
    """)

RETRIEVE_ALL_TASKS_WITH_TYPE_FROM_OPENML_FOR_DATASET = Template("""
    $PREFIXES
    SELECT 
        (?t AS ?mlsea_task_uri)
        (?id AS ?openml_task_id)
        (?t_url AS ?openml_task_url)
        ?title
        (?tt AS ?task_type)
        (?ept AS ?evaluation_procedure_type)
    WHERE {
        BIND(mlsea_openml_dataset:$datasetId AS ?ds)
        BIND(mlso_tt:$taskTypeConcept AS ?tt)

        ?t a mls:Task .
        ?t mls:definedOn ?ds .

        ?t purl:identifier ?id .
        ?t mlso:hasTaskType ?tt . 
        ?t purl:title ?title .
        ?t prov:atLocation ?t_url .

        OPTIONAL {
            ?es a mls:EvaluationSpecification .
            ?es mls:defines ?t .
            ?es mls:hasPart ?ep .
            ?ep mlso:hasEvaluationProcedureType ?ept .
        }
    } ORDER BY ?id
    """)

RETRIEVE_BATCHED_TASKS_WITH_TYPE_FROM_OPENML_FOR_DATASET = Template(
    """
    $PREFIXES
    SELECT 
        (?t AS ?mlsea_task_uri)
        (?id AS ?openml_task_id)
        (?t_url AS ?openml_task_url)
        ?title
        (?tt AS ?task_type)
        (?ept AS ?evaluation_procedure_type)
    WHERE {
        BIND(mlsea_openml_dataset:$datasetId AS ?ds)
        BIND(mlso_tt:$taskTypeConcept AS ?tt)

        ?t a mls:Task .
        ?t mls:definedOn ?ds .

        ?t purl:identifier ?id .
        ?t mlso:hasTaskType ?tt . 
        ?t purl:title ?title .
        ?t prov:atLocation ?t_url .

        OPTIONAL {
            ?es a mls:EvaluationSpecification .
            ?es mls:defines ?t .
            ?es mls:hasPart ?ep .
            ?ep mlso:hasEvaluationProcedureType ?ept .
        }

        FILTER (?id > $offsetId)
    } ORDER BY ?id
    LIMIT $limit
    """
    )

RETRIEVE_ALL_EVALUATION_PROCEDURE_TYPES_FROM_OPENML_FOR_TASK = Template("""
    $PREFIXES
    SELECT
        (?ept AS ?evaluation_procedure_type)
    WHERE {
        BIND(mlsea_openml_task:$taskId AS ?t)
        ?t a mls:Task .

        ?es a mls:EvaluationSpecification .
        ?es mls:defines ?t .
        ?es mls:hasPart ?ep .
        ?ep mlso:hasEvaluationProcedureType ?ept .
    } ORDER BY ?id
    """)

RETRIEVE_ALL_IMPLEMENTATIONS_FROM_OPENML_FOR_TASK = Template("""
    $PREFIXES
    SELECT
        (?i AS ?mlsea_implementation_uri)
        (?id AS ?openml_flow_id)
        (?i_url AS ?openml_flow_url)
        (?i_title AS ?implementation_title)
    WHERE {
        BIND(mlsea_openml_task:$taskId AS ?t)
        ?t a mls:Task .

        ?t mlso:hasRelatedImplementation ?i .
        ?i a mls:Implementation .
        ?i purl:identifier ?id .
        ?i purl:title ?i_title .
        ?i prov:atLocation ?i_url .
    } ORDER BY ?id
    """)

RETRIEVE_BATCHED_IMPLEMENTATIONS_FROM_OPENML_FOR_TASK = Template("""
    $PREFIXES
    SELECT
        (?i AS ?mlsea_implementation_uri)
        (?id AS ?openml_flow_id)
        (?i_url AS ?openml_flow_url)
        (?i_title AS ?implementation_title)
    WHERE {
        BIND(mlsea_openml_task:$taskId AS ?t)
        ?t a mls:Task .

        ?t mlso:hasRelatedImplementation ?i .
        ?i a mls:Implementation .
        ?i purl:identifier ?id .
        ?i purl:title ?i_title .
        ?i prov:atLocation ?i_url .
        
        FILTER (?id > $offsetId)
    } ORDER BY ?id
    LIMIT $limit
    """)

RETRIEVE_IMPLEMENTATION_FROM_OPENML = Template("""
    $PREFIXES
    SELECT
        (?i AS ?mlsea_implementation_uri)
        (?id AS ?openml_flow_id)
        (?i_url AS ?openml_flow_url)
        (?i_title AS ?implementation_title)
    WHERE {
        BIND(mlsea_openml_flow:$implementationId AS ?i)
        ?i a mls:Implementation .
        ?i purl:identifier ?id .
        ?i purl:title ?i_title .
        ?i prov:atLocation ?i_url .
    } ORDER BY ?id
    """)

RETRIEVE_ALL_DEPENDENCIES_FROM_OPENML_FOR_IMPLEMENTATION = Template("""
    $PREFIXES
    SELECT
        (?s AS ?mlsea_software_uri)
        (?sr AS ?software_requirement)
    WHERE {
        BIND(mlsea_openml_flow:$implementationId AS ?i)
        ?i a mls:Implementation .

        ?i mls:hasPart ?s .
        ?s a sd:Software .
        ?s sd:softwareRequirements ?sr .
    } ORDER BY ?s
    """)

RETRIEVE_ALL_RUNS_FROM_OPENML_FOR_TASK = Template("""
    $PREFIXES
    SELECT
        (?r AS ?mlsea_run_uri)
        (?id AS ?openml_run_id)
        (?r_url AS ?openml_run_url)
        (?i AS ?mlsea_implementation_uri)
    WHERE {
        BIND(mlsea_openml_task:$taskId AS ?t)

        ?t a mls:Task .

        ?r mls:achieves ?t .
        ?r a mls:Run .
        ?r prov:atLocation ?r_url .
        
        ?r mls:executes ?i .
        ?i a mls:Implementation .
        
        BIND(xsd:integer(REPLACE(STR(?r), ".*[/#]", "")) AS ?id)
    } ORDER BY ?id
    """)

RETRIEVE_BATCHED_RUNS_FROM_OPENML_FOR_TASK = Template("""
    $PREFIXES
    SELECT
        (?r AS ?mlsea_run_uri)
        (?id AS ?openml_run_id)
        (?r_url AS ?openml_run_url)
        (?i AS ?mlsea_implementation_uri)
    WHERE {
        BIND(mlsea_openml_task:$taskId AS ?t)

        ?t a mls:Task .

        ?r mls:achieves ?t .
        ?r a mls:Run .
        ?r prov:atLocation ?r_url .

        ?r mls:executes ?i .
        ?i a mls:Implementation .
        
        BIND(xsd:integer(REPLACE(STR(?r), ".*[/#]", "")) AS ?id)
        FILTER (?id > $offsetId)
    } ORDER BY ?id
    LIMIT $limit
    """)

RETRIEVE_ALL_METRICS_FROM_OPENML_FOR_RUN = Template("""
    $PREFIXES
    SELECT
        (?emt AS ?measure_type)
        (?v AS ?value)
    WHERE {
        BIND(mlsea_openml_run:$runId AS ?r)

        ?r a mls:Run .

        ?r mls:hasOutput ?me .
        ?me a mls:ModelEvaluation . 
        ?me mls:hasValue ?v .
        ?me mls:specifiedBy ?em .
        ?em mlso:hasEvaluationMeasureType ?emt .
    } ORDER BY ?id
    """)

RETRIEVE_ALL_SOFTWARE_REQUIREMENTS_FROM_OPENML_FOR_IMPLEMENTATION = Template("""
    $PREFIXES
    SELECT
        (?sr AS ?software_requirement)
    WHERE {
        BIND(mlsea_openml_implementation:$implementationId AS ?r)

        ?r a mls:Run .

        ?r mls:executes ?i .
        ?i a mls:Implementation .
        ?i mls:hasSoftwareRequirement ?sr .
    } ORDER BY ?id
    """)
