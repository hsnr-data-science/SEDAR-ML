import re
from collections import OrderedDict
from typing import Optional, List

import openml
from beanie import WriteRules, Link

from common.data import Task, Implementation
from common.data.implementation import Parameter, Software, Platform
from mlsea import mlsea_repository as mlsea
from mlsea.dtos import ImplementationDto, SoftwareDto
from processing.types import ProcessingOptions

IMPLEMENTATION_BASE_URI = "http://w3id.org/mlsea/openml/flow/"

async def process_all_implementations(task: Task, options: ProcessingOptions = ProcessingOptions()):
    openml_task_id = int(task.mlsea_uri.split('/')[-1])
    count = 0
    offset_id = options.offset.pop('implementation', 0) if options.offset is not None else 0
    while True:
        task_implementations_df = mlsea.retrieve_all_implementations_from_openml_for_task(openml_task_id, batch_size=100, offset_id=offset_id)
        if task_implementations_df.empty:
            break

        if options.head is not None:
            task_implementations_df = task_implementations_df.head(options.head)

        for implementation_dto in task_implementations_df.itertuples(index=False):
            try:
                implementation_dto = ImplementationDto(*implementation_dto)

                print(f"Processing implementation {implementation_dto.openml_flow_id}")

                software_df = mlsea.retrieve_dependencies_from_openml_for_implementation(
                    implementation_dto.openml_flow_id)
                software_dtos = [SoftwareDto(*software_dto) for software_dto in software_df.itertuples(index=False)]

                implementation: Implementation = await _ensure_implementation_exists(implementation_dto, software_dtos)
                if task.related_implementations is None:
                    task.related_implementations = []

                if implementation.id not in [impl.to_ref().id for impl in task.related_implementations]:
                    task.related_implementations.append(Link(implementation.to_ref(), Implementation))

            except Exception as e:
                print(f"Error processing implementation {implementation_dto.openml_flow_id}: {e}")
                with open("error_messages.txt", "a") as f:
                    f.write(f"implementation {implementation_dto.openml_flow_id}: {e}\n")
                with open("error_implementations.txt", "a") as f:
                    f.write(f"{implementation_dto.openml_flow_id}\n")

            finally:
                count += 1
                offset_id = implementation_dto.openml_flow_id

        await task.save(link_rule=WriteRules.DO_NOTHING)

        if options.head is not None and count >= options.head:
            break

async def find_or_create_implementation(openml_implementation_id: int) -> Implementation:
    implementation = await Implementation.find_one(
        #Implementation.mlsea_uri == f"{IMPLEMENTATION_BASE_URI}{openml_implementation_id}"
        { "mlseaUri": f"{IMPLEMENTATION_BASE_URI}{openml_implementation_id}" }
    )
    if implementation is not None:
        return implementation

    implementation_df = mlsea.retrieve_implementation_from_openml(openml_implementation_id)
    implementation_dto = ImplementationDto(*implementation_df.iloc[0])
    software_df = mlsea.retrieve_dependencies_from_openml_for_implementation(openml_implementation_id)
    software_dtos = [SoftwareDto(*software_dto) for software_dto in software_df.itertuples(index=False)]

    implementation: Implementation = await _ensure_implementation_exists(implementation_dto, software_dtos)
    return implementation

async def _ensure_implementation_exists(implementation_dto: ImplementationDto, software_dtos: List[SoftwareDto]):
    implementation: Optional[Implementation] = await Implementation.find_one(
        #Implementation.mlsea_uri == implementation_dto.mlsea_implementation_uri
        { "mlseaUri": implementation_dto.mlsea_implementation_uri }
    )

    if implementation is not None:
        return implementation

    dependencies = [
        parsed_dependency
        for software_dto in software_dtos
        for parsed_dependency in _transform_software_dto(software_dto)
    ]

    platform = _identify_platform(implementation_dto, dependencies)

    openml_flow = openml.flows.get_flow(implementation_dto.openml_flow_id)
    parameters_meta_info = openml_flow.parameters_meta_info
    parameters = OrderedDict[str, Parameter]()
    for param, default_value in openml_flow.parameters.items():
        parameters[param] = Parameter(
            default_value=default_value,
            type=parameters_meta_info[param]['data_type'],
            description=parameters_meta_info[param]['description']
        )

    class_name = openml_flow.class_name

    components = {}
    for component_name, component in openml_flow.components.items():
        implementation = await find_or_create_implementation(component.flow_id)
        if implementation is None:
            raise ValueError(f"Could not find or create implementation for component: {component.flow_id}")
        components[component_name] = Link(implementation.to_ref(), Implementation)


    implementation = Implementation(
        mlsea_uri=implementation_dto.mlsea_implementation_uri,
        title=openml_flow.name,
        parameters=parameters,
        components=components if len(components.items()) > 0 else None,
        description=openml_flow.description,
        dependencies=dependencies,
        platform=platform,
        class_name=class_name
    )
    await implementation.insert(link_rule=WriteRules.DO_NOTHING)
    return implementation

def _transform_software_dto(software_dto: SoftwareDto) -> List[Software]:
    dependency_exceptions = [
        {
            'original': 'Shark machine learning library',
            'transformed': {'name': 'Shark', 'version': None}
        },
        {
            'original': 'Build on top of Weka API (Jar version 3.?.?)',
            'transformed': {'name': 'Weka', 'version': '3.?.?'}
        },
        {
            'original': 'MLR 2.4',
            'transformed': {'name': 'MLR', 'version': '2.4'}
        }
    ]

    # handle exceptions
    for exception in dependency_exceptions:
        if software_dto.software_requirement == exception['original']:
            return [Software(mlsea_uri=software_dto.mlsea_software_uri, **exception['transformed'])]

    requirements = software_dto.software_requirement.split(' ')
    pattern = r"([a-zA-Z\d]+)([_<>=!]*)([a-zA-Z\d.+-]*)"

    parsed_dependencies = []
    for requirement in requirements:
        match = re.match(pattern, requirement)
        if match:
            name, operator, version = match.groups()
            parsed_dependencies.append(
                Software(mlsea_uri=software_dto.mlsea_software_uri, name=name, version=version))

        else:
            raise ValueError(
                f"Could not parse software requirement: {requirement} for software: {software_dto.mlsea_software_uri}")

    return parsed_dependencies

def _identify_platform(implementation_dto: ImplementationDto, dependencies: List[Software]) -> Platform:
    supported_platforms = [p for p in Platform if p != Platform.UNKNOWN]

    # First check if the implementation title starts with a platform short form and if the platform is a dependency
    for platform in supported_platforms:
        if any(implementation_dto.title.lower().startswith(f"{prefix}.") for prefix in platform.short_forms):
            if any(dependency.name.lower() in platform.short_forms for dependency in dependencies):
                return platform

    # If not, check if any of the dependencies are a platform
    for platform in supported_platforms:
        if any(dependency.name.lower() in platform.short_forms for dependency in dependencies):
            return platform

    # Even if the platform is contained in the dependencies, it might start with a short form
    for platform in supported_platforms:
        if any(implementation_dto.title.lower().startswith(f"{prefix}.") for prefix in platform.short_forms):
            return platform

    return Platform.UNKNOWN
