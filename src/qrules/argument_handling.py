"""Handles argument handling for rules.

Responsibilities are the check of requirements for rules and the creation of the
arguments from general graph property maps. The information is extracted from the type
annotations of the rules.
"""

from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import attrs

from qrules.conservation_rules import (
    ConservationRule,
    EdgeQNConservationRule,
    GraphElementRule,
)
from qrules.quantum_numbers import EdgeQuantumNumber, NodeQuantumNumber, Parity

Scalar = Union[int, float]

Rule = Union[GraphElementRule, EdgeQNConservationRule, ConservationRule]

_ElementType = TypeVar("_ElementType")

GraphElementPropertyMap = Dict[Type[_ElementType], Scalar]
GraphEdgePropertyMap = GraphElementPropertyMap[EdgeQuantumNumber]
"""Type alias for a graph edge property map."""
GraphNodePropertyMap = GraphElementPropertyMap[NodeQuantumNumber]
"""Type alias for a graph node property map."""


def _is_optional(field_type: type | None) -> bool:
    return (
        getattr(field_type, "__origin__", None) is Union
        and type(None) in field_type.__args__  # type: ignore[union-attr]
    )


def _is_sequence_type(input_type: type) -> bool:
    origin = getattr(input_type, "__origin__", None)
    return origin in {list, tuple, List, Tuple}


def _is_edge_quantum_number(qn_type: Any) -> bool:
    return qn_type in EdgeQuantumNumber.__args__  # type: ignore[attr-defined]


def _is_node_quantum_number(qn_type: Any) -> bool:
    return qn_type in NodeQuantumNumber.__args__  # type: ignore[attr-defined]


class _CompositeArgumentCheck:
    def __init__(
        self,
        class_field_types: list[EdgeQuantumNumber] | list[NodeQuantumNumber],
    ) -> None:
        self.__class_field_types = class_field_types

    def __call__(
        self,
        props: GraphElementPropertyMap,
    ) -> bool:
        return all(
            class_field_type in props for class_field_type in self.__class_field_types
        )


def _direct_qn_check(
    qn_type: type[EdgeQuantumNumber] | type[NodeQuantumNumber],
) -> Callable[[GraphElementPropertyMap], bool]:
    def wrapper(props: GraphElementPropertyMap) -> bool:
        return qn_type in props

    return wrapper


def _sequence_input_check(func: Callable) -> Callable[[Sequence], bool]:
    def wrapper(states_list: Sequence[Any]) -> bool:
        if not isinstance(states_list, (list, tuple)):
            msg = "Rule evaluated with invalid argument type..."
            raise TypeError(msg)

        return all(func(x) for x in states_list)

    return wrapper


def _check_all_arguments(checks: list[Callable]) -> Callable[..., bool]:
    def wrapper(*args: Any) -> bool:
        return all(check(arg) for check, arg in zip(checks, args))

    return wrapper


class _ValueExtractor(Generic[_ElementType]):
    def __init__(self, obj_type: type[_ElementType] | None) -> None:
        self.__obj_type: type[_ElementType] = obj_type  # type: ignore[assignment]
        self.__function = self.__extract

        if _is_optional(obj_type):
            self.__obj_type = obj_type.__args__[0]  # type: ignore[union-attr]
            self.__function = self.__optional_extract  # type: ignore[assignment]

    def __call__(
        self, props: GraphElementPropertyMap[_ElementType]
    ) -> _ElementType | None:
        return self.__function(props)

    def __optional_extract(
        self, props: GraphElementPropertyMap[_ElementType]
    ) -> _ElementType | None:
        if self.__obj_type in props:
            return self.__extract(props)

        return None

    def __extract(
        self, props: GraphElementPropertyMap[_ElementType]
    ) -> _ElementType | None:
        value = props[self.__obj_type]
        if value is None:
            return None
        if (
            "__supertype__" in self.__obj_type.__dict__
            and self.__obj_type.__supertype__ == Parity  # type: ignore[attr-defined]
        ):
            return self.__obj_type.__supertype__(value)  # type: ignore[attr-defined]
        return self.__obj_type(value)  # type: ignore[call-arg]


class _CompositeArgumentCreator:
    def __init__(self, class_type: type) -> None:
        self.__class_type = class_type
        self.__extractors = {
            class_field.name: (
                _ValueExtractor[EdgeQuantumNumber](class_field.type)
                if _is_edge_quantum_number(class_field.type)
                else _ValueExtractor[NodeQuantumNumber](class_field.type)
            )
            for class_field in attrs.fields(class_type)  # type: ignore[misc]
        }

    def __call__(
        self,
        props: GraphElementPropertyMap,
    ) -> Any:
        return self.__class_type(**{
            arg_name: extractor(props)  # type: ignore[operator]
            for arg_name, extractor in self.__extractors.items()
        })


def _sequence_arg_builder(func: Callable) -> Callable[[Sequence], list[Any]]:
    def wrapper(states_list: Sequence[Any]) -> list[Any]:
        if not isinstance(states_list, (list, tuple)):
            msg = "Rule evaluated with invalid argument type..."
            raise TypeError(msg)

        return [func(x) for x in states_list if x]

    return wrapper


def _build_all_arguments(checks: list[Callable]) -> Callable:
    def wrapper(*args: Any) -> list[Any]:
        return [check(arg) for check, arg in zip(checks, args) if arg]

    return wrapper


class RuleArgumentHandler:
    def __init__(self) -> None:
        self.__rule_to_requirements_check: dict[Rule, Callable] = {}
        self.__rule_to_argument_builder: dict[Rule, Callable] = {}

    def __verify(self, rule_annotations: list) -> None:
        pass

    @staticmethod
    def __create_requirements_check(
        argument_types: list[type],
    ) -> Callable:
        individual_argument_checkers = []
        for input_type in argument_types:
            is_list = False
            qn_type = input_type
            if _is_sequence_type(input_type):
                qn_type = input_type.__args__[0]  # type: ignore[attr-defined]
                is_list = True

            if attrs.has(qn_type):
                class_field_types = [
                    class_field.type
                    for class_field in attrs.fields(qn_type)  # type: ignore[misc]
                    if not _is_optional(class_field.type)
                ]
                qn_check_function: Callable[..., bool] = _CompositeArgumentCheck(
                    class_field_types  # type: ignore[arg-type]
                )
            else:
                qn_check_function = _direct_qn_check(qn_type)

            if is_list:
                qn_check_function = _sequence_input_check(qn_check_function)

            individual_argument_checkers.append(qn_check_function)

        return _check_all_arguments(individual_argument_checkers)

    @staticmethod
    def __create_argument_builder(
        argument_types: list[type],
    ) -> Callable:
        individual_argument_builders = []
        for input_type in argument_types:
            is_list = False
            qn_type = input_type
            if _is_sequence_type(input_type):
                qn_type = input_type.__args__[0]  # type: ignore[attr-defined]
                is_list = True

            if attrs.has(qn_type):
                arg_builder: Callable[..., Any] = _CompositeArgumentCreator(qn_type)
            else:
                if _is_edge_quantum_number(qn_type):
                    arg_builder = _ValueExtractor[EdgeQuantumNumber](qn_type)
                elif _is_node_quantum_number(qn_type):
                    arg_builder = _ValueExtractor[NodeQuantumNumber](qn_type)
                else:
                    msg = (
                        f"Quantum number type {qn_type} is not supported. Has to be of"
                        " type Edge/NodeQuantumNumber."
                    )
                    raise TypeError(msg)

            if is_list:
                arg_builder = _sequence_arg_builder(arg_builder)

            individual_argument_builders.append(arg_builder)

        return _build_all_arguments(individual_argument_builders)

    def register_rule(self, rule: Rule) -> tuple[Callable, Callable]:
        if (
            rule not in self.__rule_to_requirements_check
            or rule not in self.__rule_to_argument_builder
        ):
            rule_annotations = _resolve_argument_type_hints(rule)

            # check type annotations are legal
            try:
                self.__verify(rule_annotations)
            except TypeError as exception:
                msg = f"rule {rule!s}: {exception!s}"
                raise TypeError(msg) from exception

            # then create requirements check function and add to dict
            self.__rule_to_requirements_check[rule] = self.__create_requirements_check(
                rule_annotations
            )

            # then create arguments builder function and add to dict
            self.__rule_to_argument_builder[rule] = self.__create_argument_builder(
                rule_annotations
            )

        return (
            self.__rule_to_requirements_check[rule],
            self.__rule_to_argument_builder[rule],
        )


def _resolve_argument_type_hints(rule: Rule) -> list:
    """Get the signature of a rule, with resolved type hints.

    >>> from qrules.conservation_rules import gellmann_nishijima, MassConservation
    >>> _resolve_argument_type_hints(gellmann_nishijima)
    [<class 'qrules.conservation_rules.GellMannNishijimaInput'>]
    >>> _resolve_argument_type_hints(MassConservation(width_factor=1.0))
    [typing.List[qrules.conservation_rules.MassEdgeInput], typing.List[qrules.conservation_rules.MassEdgeInput]]
    """
    func_signature = inspect.signature(rule)
    if not func_signature.return_annotation:
        msg = f"Missing return type annotation for rule {rule!s}"
        raise TypeError(msg)
    rule_annotations = []
    for par in func_signature.parameters.values():
        if not par.annotation:
            msg = f"missing type annotations for argument {par.name} of rule {rule!s}"
            raise TypeError(msg)
        rule_annotations.append(par.annotation)
    return rule_annotations


def get_required_qns(
    rule: Rule,
) -> tuple[set[type[EdgeQuantumNumber]], set[type[NodeQuantumNumber]]]:
    rule_annotations = []
    for par in inspect.signature(rule).parameters.values():
        if not par.annotation:
            msg = f"missing type annotations for rule {rule!s}"
            raise TypeError(msg)
        rule_annotations.append(par.annotation)

    required_edge_qns: set[type[EdgeQuantumNumber]] = set()
    required_node_qns: set[type[NodeQuantumNumber]] = set()

    for input_type in rule_annotations:
        class_type = input_type
        if _is_sequence_type(input_type):
            class_type = input_type.__args__[0]

        if attrs.has(class_type):
            for class_field in attrs.fields(class_type):  # type: ignore[misc]
                field_type = (
                    class_field.type.__args__[0]  # type: ignore[union-attr]
                    if _is_optional(class_field.type)
                    else class_field.type
                )
                if _is_edge_quantum_number(field_type):
                    required_edge_qns.add(field_type)
                else:
                    required_node_qns.add(field_type)
        else:
            if _is_edge_quantum_number(class_type):
                required_edge_qns.add(class_type)
            else:
                required_node_qns.add(class_type)

    return required_edge_qns, required_node_qns
