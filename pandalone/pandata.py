#! python
# -*- coding: utf-8 -*-
#
# Copyright 2013-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
A :dfn:`pandas-model` is a tree of strings, numbers, sequences, dicts, pandas instances and resolvable
URI-references, implemented by :class:`Pandel`.
"""

import abc
import binascii
import collections.abc as cabc
import functools as fnt
import numbers
import pickle
import re
from collections import OrderedDict, namedtuple
from json.decoder import JSONDecoder
from json.encoder import JSONEncoder
from typing import Union
from unittest.mock import MagicMock
from urllib.parse import urljoin

import jsonschema
import numpy as np
import pandas as pd
from jsonschema import ValidationError
from jsonschema.exceptions import RefResolutionError, SchemaError
from pandas.core.generic import NDFrame

__commit__ = ""

_value_with_units_regex = re.compile(
    r"""^\s*
                                        (?P<name>[^[<]*?)   # column-name
                                        \s*
                                        (?P<units>          # start parenthesized-units optional-group
                                            \[              # units enclosed in []
                                                [^\]]*
                                            \]
                                            |
                                            <              # units enclosed in <>
                                                [^)]*
                                            >
                                        )?                  # end parenthesized-units
                                        \s*$""",
    re.X,
)
_units_cleaner_regex = re.compile(r"^[<[]|[\]>]$")


"""An item-descriptor with units, i.e. used as a table-column header."""
_U = namedtuple("United", ("name", "units"))


def parse_value_with_units(arg):
    """
    Parses *name-units* pairs (i.e. used as a table-column header).

    :return:    a United(name, units) named-tuple, or `None` if bad syntax;
                note that ``name=''`` but ``units=None`` when missing.

    Examples::

        >>> parse_value_with_units('value [units]')
        United(name='value', units='units')

        >>> parse_value_with_units('foo   bar  <bar/krow>')
        United(name='foo   bar', units='bar/krow')

        >>> parse_value_with_units('no units')
        United(name='no units', units=None)

        >>> parse_value_with_units('')
        United(name='', units=None)

    But notice::

        >>> assert parse_value_with_units('ok but [bad units') is None

        >>> parse_value_with_units('<only units>')
        United(name='', units='only units')

        >>> parse_value_with_units(None)  # doctest:+ELLIPSIS
        Traceback (most recent call last):
        TypeError: expected string or ...

    """

    m = _value_with_units_regex.match(arg)
    if m:
        res = m.groupdict()
        units = res["units"]
        if units:
            res["units"] = _units_cleaner_regex.sub("", units)
        return _U(**res)


class ModelOperations(namedtuple("ModelOperations", "inp out conv")):

    """
    Customization functions for traversing, I/O, and converting self-or-descendant branch (sub)model values.
    """

    def __new__(cls, inp=None, out=None, conv=None):
        """

        :param list inp:    the `args-list` to :meth:`Pandel._read_branch()`

        :param out:         The args to :meth:`Pandel._write_branch()`, that may be specified either as:

                            * an `args-list`, that will apply for all model data-types (lists, dicts & pandas),
                            * a map of ``type`` --> ``args-list``, where the ``None`` key is the *catch-all* case,
                            * a function returning the `args-list` for some branch-value,
                              with signature: ``def get_write_branch_args(branch)``.

        :param conv:        The conversion-functions (:dfn:`convertors`) for the various model's data-types.
                            The convertors have signature ``def convert(branch)``, and they may be
                            specified either as:

                            * a map of ``(from_type, to_type)`` --> ``conversion_func()``, where the ``None`` key
                              is the *catch-all* case,
                            * a "master-switch" function returning the appropriate convertor
                              depending on the requested conversion.
                              The master-function's signature is ``def get_convertor(from_branch, to_branch)``.

                            The minimum convertors demanded by :class:`Pandel` are (at least, check the code for more):

                            * DataFrame  <--> dict
                            * Series     <--> dict
                            * ndarray    <--> list
        """

        return super(ModelOperations, cls).__new__(cls, inp, out, conv)

    def choose_out_args(self, branch):
        pass

    def choose_convertor(self, from_type, to_type):
        pass


cabc.Sequence.register(pd.Series)
cabc.Mapping.register(pd.DataFrame)
cabc.Mapping.register(pd.Series)


def _is_array(checker, instance):
    return not isinstance(instance, str) and isinstance(
        instance, (cabc.Sequence, np.ndarray)
    )


def _is_object(checker, instance):
    return isinstance(instance, cabc.Mapping)


def _is_bool(checker, instance):
    return isinstance(instance, (bool, np.bool_))


def _is_integer(checker, instance):
    # bool inherits from int, so ensure bools aren't reported as ints
    if isinstance(instance, bool):
        return False
    return isinstance(instance, (int, np.integer))


def _is_null(checker, instance):
    try:
        return instance is None or bool(np.isnan(instance))
    except (TypeError, ValueError):  # value-error when np-array.
        return False


def _find_additional_properties(instance, schema):
    """
    Return the set of additional properties for the given ``instance``.

    Weeds out properties that should have been validated by ``properties`` and
    / or ``patternProperties``.

    Assumes ``instance`` is dict-like already.

    """

    properties = schema.get("properties", {})
    patterns = "|".join(schema.get("patternProperties", {}))
    for property in instance.keys():  # keys() wer missin, PATCHING JUST FOR THIS!!!!
        if property not in properties:
            if patterns and re.search(patterns, property):
                continue
            yield property


def _rule_additionalProperties(validator, aP, instance, schema):
    if not validator.is_type(instance, "object"):
        return

    extras = set(_find_additional_properties(instance, schema))

    if validator.is_type(aP, "object"):
        for extra in extras:
            for error in validator.descend(instance[extra], aP, path=extra):
                yield error
    elif not aP and extras:
        if "patternProperties" in schema:
            patterns = sorted(schema["patternProperties"])
            if len(extras) == 1:
                verb = "does"
            else:
                verb = "do"
            error = "%s %s not match any of the regexes: %s" % (
                ", ".join(map(repr, sorted(extras))),
                verb,
                ", ".join(map(repr, patterns)),
            )
            yield ValidationError(error)
        else:
            from jsonschema import _utils

            error = "Additional properties are not allowed (%s %s unexpected)"
            yield ValidationError(error % _utils.extras_msg(extras))


def _rule_propertyNames(validator, propertyNames, instance, schema):
    if not validator.is_type(instance, "object"):
        return

    for property in instance.keys():
        for error in validator.descend(instance=property, schema=propertyNames):
            yield error


def _is_null_in_type(typ):
    if not typ:
        return False
    if isinstance(typ, str):
        return typ == "null"
    return "null" in typ


def first_defined(*var, default=None):
    """Return the 1st non-none `var`, or `default`."""
    for v in var:
        if v is not None:
            return v
    return default


def _rule_auto_defaults_properties(
    validator,
    properties,
    instance,
    schema,
    original_props_rule,
    auto_default,
    auto_default_nulls,
    auto_remove_nulls,
):
    """
    Adapted from: https://python-jsonschema.readthedocs.io/en/stable/faq/#frequently-asked-questions
    """
    if not _is_object(None, instance):
        return

    for property, subschema in properties.items():
        prop_given = property in instance
        if "default" in subschema and (
            (
                first_defined(subschema.get("autoDefault"), auto_default)
                and not prop_given
            )
            or (
                first_defined(subschema.get("autoDefaultNull"), auto_default_nulls)
                and not _is_null_in_type(subschema.get("type"))
                and prop_given
                and _is_null(None, instance[property])
            )
        ):
            instance[property] = subschema["default"]
        elif (
            first_defined(subschema.get("autoRemoveNull"), auto_remove_nulls)
            and not _is_null_in_type(subschema.get("type"))
            and prop_given
            and _is_null(None, instance[property])
        ):
            del instance[property]

    for error in original_props_rule(validator, properties, instance, schema):
        yield error


def rule_enum(validator, enums, instance, schema):
    """Overridden to evade pandas-equals after Julian/jsonschema#575 fixed bool != 0,1 (v3.0.2)."""
    unbool = jsonschema._utils.unbool
    if instance is 0 or instance is 1:
        unbooled = unbool(instance)
        if all(unbooled != unbool(each) for each in enums):
            yield ValidationError("%r is not one of %r" % (instance, enums))
    elif instance not in enums:
        yield ValidationError("%r is not one of %r" % (instance, enums))


def PandelVisitor(
    schema,
    resolver=None,
    format_checker=None,
    auto_default: Union[bool, None] = True,
    auto_default_nulls: Union[bool, None] = False,
    auto_remove_nulls: Union[bool, None] = False,
):
    """
    A customized jsonschema-validator suporting instance-trees with pandas and numpy objects, natively.

    :param auto_default:
        When the tri-state bool ``autoDefault`` in schema or this param are enabled,
        it applies any schema's ``default`` value if a property is missing and
        schema's ``type`` does not support `nulls`.

        - Independent of `auto_default_nulls` (you may enable both).
        - See meth:`_rule_auto_defaults_properties`.

    :param auto_default_nulls:
        When the tri-state bool ``autoDefaultNull`` in schema or this param are
        it applies any schema's ``default`` value if the property is `null` and
        schema's ``type`` does not support `nulls`.

        - Independent of `auto_default` (you may enable both).
        - Take precedence over `auto_remove_nulls`.
        - See meth:`_rule_auto_defaults_properties`.

    :param auto_remove_nulls:
        When the tri-state bool ``autoRemoveNull`` in schema or this param are
        it removes a `null` property value if the schema's ``type`` does not accept `nulls`.

        - See meth:`_rule_auto_defaults_properties`.

        .. ATTENTION::
            If this is enabled, any `required` properties rule must FOLLOW
            the `properties` rule.

    Any pandas or numpy instance (for example ``obj``) is treated like that:

    +----------------------------+------------------------------------------+
    |        Python Type         |     JSON Equivalence                     |
    +----------------------------+------------------------------------------+
    | :class:`pandas.DataFrame`  | as ``object`` *json-type*, with:         |
    |                            | keys: ``obj.columns`` (MUST be strings)  |
    |                            | values: ``obj[col].values``              |
    |                            |                                          |
    |                            | NOTE: len(df) on rows(!), not columns.   |
    +----------------------------+------------------------------------------+
    | :class:`pandas.Series`     | - as ``object`` *json-type*, with:       |
    |                            |   keys: ``obj.index`` (MUST be strings)  |
    |                            |   values: ``obj.values``                 |
    |                            | - as ``array`` *json-type*               |
    +----------------------------+------------------------------------------+
    | :class:`np.ndarray`        | as ``array`` *json-type* IF ndim == 1    |
    +----------------------------+------------------------------------------+
    | :class:`cabc.Sequence`     | as ``array`` IF not string               |
    |                            | (like lists, tuples)                     |
    +----------------------------+------------------------------------------+

    Note that the value of each dataFrame column is a :``ndarray`` instances.

    The simplest validations of an object or a pandas-instance is like this:

        >>> import pandas as pd

        >>> schema = {
        ...     'type': 'object',
        ... }
        >>> pv = PandelVisitor(schema)

        >>> pv.validate({'foo': 'bar'})
        >>> pv.validate(pd.Series({'foo': 1}))
        >>> pv.validate([1,2])                                       ## A sequence is invalid here.
        Traceback (most recent call last):
        ...
        jsonschema.exceptions.ValidationError: [1, 2] is not of type 'object'
        <BLANKLINE>
        Failed validating 'type' in schema:
            {'type': 'object'}
        <BLANKLINE>
        On instance:
            [1, 2]


    Or demanding specific properties with ``required`` and no ``additionalProperties``:

        >>> schema = {
        ...     'type':     'object',
        ...     'properties': {
        ...         'foo': {}
        ...     },
        ...     'required': ['foo'],
        ...     'additionalProperties': False,
        ... }
        >>> pv = PandelVisitor(schema)

        >>> pv.validate(pd.Series({'foo': 1}))
        >>> pv.validate(pd.Series({'foo': 1, 'bar': 2}))             ## Additional 'bar' is present!
        Traceback (most recent call last):
        ...
        jsonschema.exceptions.ValidationError: Additional properties are not allowed ('bar' was unexpected)
        <BLANKLINE>
        Failed validating 'additionalProperties' in schema:
            {'additionalProperties': False,
             'properties': {'foo': {}},
             'required': ['foo'],
             'type': 'object'}
        <BLANKLINE>
        On instance:
            foo    1
            bar    2
            dtype: int64

        >>> pv.validate(pd.Series({}))                               ## Required 'foo' missing!
        Traceback (most recent call last):
        ...
        jsonschema.exceptions.ValidationError: 'foo' is a required property
        <BLANKLINE>
        Failed validating 'required' in schema:
            {'additionalProperties': False,
             'properties': {'foo': {}},
             'required': ['foo'],
             'type': 'object'}
        <BLANKLINE>
        On instance:
            Series([], dtype: float64)

    """
    validator = jsonschema.validators.validator_for(schema)
    props_rule = validator.VALIDATORS["properties"]

    rules = {"additionalProperties": _rule_additionalProperties}
    if any((auto_default, auto_default_nulls, auto_remove_nulls)):
        rules["properties"] = fnt.partial(
            _rule_auto_defaults_properties,
            original_props_rule=props_rule,
            auto_default=auto_default,
            auto_default_nulls=auto_default_nulls,
            auto_remove_nulls=auto_remove_nulls,
        )
    if "propertyNames" in validator.VALIDATORS:
        rules["propertyNames"] = _rule_propertyNames
    if hasattr(jsonschema._utils, "unbool"):
        rules["enum"] = rule_enum  # fix pandas after jsonschema-3.0.2

    ValidatorClass = jsonschema.validators.extend(
        validator,
        type_checker=validator.TYPE_CHECKER.redefine_many(
            {
                "array": _is_array,
                "object": _is_object,
                "integer": _is_integer,
                "boolean": _is_bool,
                "null": _is_null,
            }
        ),
        validators=rules,
    )

    return ValidatorClass(schema, resolver=resolver, format_checker=format_checker)


class Pandel(object):

    """
    Builds, validates and stores a *pandas-model*, a mergeable stack of JSON-schema abiding trees of
    strings and numbers, assembled with

    * sequences,
    * dictionaries,
    * :class:`pandas.DataFrame`,
    * :class:`pandas.Series`, and
    * URI-references to other model-trees.



    .. _pandel-overview:

    **Overview**

    The **making of a model** involves, among others, schema-validating, reading :dfn:`subtree-branches`
    from URIs, cloning, converting and merging multiple :dfn:`sub-models` in a single :dfn:`unified-model` tree,
    without side-effecting given input.
    All these happen in 4+1 steps::

                       ....................... Model Construction .................
          ------------ :  _______    ___________                                  :
         / top_model /==>|Resolve|->|PreValidate|-+                               :
         -----------'  : |___0___|  |_____1_____| |                               :
          ------------ :  _______    ___________  |   _____    ________    ______ :   --------
         / base-model/==>|Resolve|->|PreValidate|-+->|Merge|->|Validate|->|Curate|==>/ model /
         -----------'  : |___0___|  |_____1_____|    |_ 2__|  |___3____|  |__4+__|:  -------'
                       ............................................................

    All steps are executed "lazily" using generators (with :keyword:`yield`).
    Before proceeding to the next step, the previous one must have completed successfully.
    That way, any ad-hoc code in building-step-5(*curation*), for instance, will not suffer a horrible death
    due to badly-formed data.

    [TODO] The **storing of a model** simply involves distributing model parts into different files and/or formats,
    again without side-effecting the unified-model.



    .. _pandel-building-model:

    **Building model**

    Here is a detailed description of each building-step:

    1.  :meth:`_resolve` and substitute any `json-references <http://tools.ietf.org/html/draft-pbryan-zyp-json-ref-03>`_
        present in the submodels with content-fragments fetched from the referred URIs.
        The submodels are **cloned** first, to avoid side-effecting them.

        Although by default a combination of *JSON* and *CSV* files is expected, this can be customized,
        either by the content in the json-ref, within the model (see below), or
        as :ref:`explained  <pandel-customization>` below.

        The **extended json-refs syntax** supported provides for passing arguments into :meth:`_read_branch()`
        and :meth:`_write_branch()` methods.  The syntax is easier to explain by showing what
        the default :attr:`_global_cntxt` corresponds to, for a ``DataFrame``::

            {
              "$ref": "http://example.com/example.json#/foo/bar",
              "$inp": ["AUTO"],
              "$out": ["CSV", "encoding=UTF-8"]
            }

        And here what is required to read and (later) store into a HDF5 local file with a predefined name::

            {
              "$ref": "file://./filename.hdf5",
              "$inp": ["AUTO"],
              "$out": ["HDF5"]
            }

        .. Warning:: Step NOT IMPLEMENTED YET!


    2.  Loosely :meth:`_prevalidate` each sub-model separately with :term:`json-schema`,
        where any pandas-instances (DataFrames and Series) are left as is.
        It is the duty of the developer to ensure that the prevalidation-schema is *loose enough* that
        it allows for various submodel-forms, prior to merging, to pass.


    3.  Recursively **clone**  and :meth:`_merge` sub-models in a single unified-model tree.
        Branches from sub-models higher in the stack override the respective ones from the sub-models below,
        recursively.  Different object types need to be **converted** appropriately (ie. merging
        a ``dict`` with a ``DataFrame`` results into a ``DataFrame``, so the dictionary has to convert
        to dataframe).

        The required **conversions** into pandas classes can be customized as :ref:`explained  <pandel-customization>`
        below.  Series and DataFrames cannot merge together, and Sequences do not merge
        with any other object-type (themselfs included), they just "overwrite".

        The default convertor-functions defined both for submodels and models are listed in the following table:

        ============ ========== =========================================
            From:       To:                  Method:
        ============ ========== =========================================
         dict        DataFrame  ``pd.DataFrame``  (the constructor)
         DataFrame   dict       ``lambda df: df.to_dict('list')``
         dict        Series     ``pd.Series``     (the constructor)
         Series      dict       :meth:`lambda sr: sr.to_dict()`
        ============ ========== =========================================


    4.  Strictly json-:meth:`_validate` the unified-model (ie enforcing ``required`` schema-rules).

        The required **conversions** from pandas classes can be customized as :ref:`explained  <pandel-customization>`
        below.

        The default convertor-functions are the same as above.


    5.  (Optionally) Apply the :meth:`_curate` functions on the the model to enforce dependencies and/or any
        ad-hoc generation-rules among the data.  You can think of bash-like expansion patterns,
        like ``${/some/path:=$HOME}`` or expressions like ``%len(../other/path)``.



    .. _pandel-storing:

    **Storing model**

    When storing model-parts, if unspecified, the filenames to write into will be deduced from the jsonpointer-path
    of the ``$out``'s parent, by substituting "strange" chars with undescores(``_``).

    .. Warning:: Functionality NOT IMPLEMENTED YET!



    .. _pandel-customization:

    **Customization**

    Some operations within steps (namely *conversion* and *IO*) can be customized by the following means
    (from lower to higher precedance):

    a.  The global-default :class:`ModelOperations` instance on the :attr:`_global_cntxt`,
        applied on both submodels and unified-model.

        For example to channel the whole reading/writing of models through
        `HDF5 <http://pandas.pydata.org/pandas-docs/stable/io.html#io-hdf5>`_ data-format, it would suffice
        to modify the :attr:`_global_cntxt` like that::

            pm = FooPandelModel()                        ## some concrete model-maker
            io_args = ["HDF5"]
            pm.mod_global_operations(inp=io_args, out=io_args)

    b.  [TODO] Extra-properties on the json-schema applied on both submodels and unified-model for the specific path defined.
        The supported properties are the non-functional properties of :class:`ModelOperations`.

    d.  Specific-properties regarding *IO* operations within each submodel - see the *resolve* building-step,
        above.

    c.  Context-maps of ``json_paths`` --> :class:`ModelOperations` instances, installed by :meth:`add_submodel()` and
        :attr:`unified_contexts` on the model-maker.  They apply to self-or-descedant subtree of each model.

        The `json_path` is a strings obeying a simplified :term:`json-pointer` syntax (no char-normalizations yet),
        ie ``/some/foo/1/pointer``.  An empty-string(``''``) matches all model.

        When multiple convertors match for a model-value, the selected convertor to be used is the most specific one
        (the one with longest prefix).  For instance, on the model::

            [ { "foo": { "bar": 0 } } ]


        all of the following would match the ``0`` value:

        - the global-default :attr:`_global_cntxt`,
        - ``/``, and
        - ``/0/foo``

        but only the last's context-props will be applied.



    .. _Attributes:

    **Atributes**

    .. Attribute:: model

        The model-tree that will receive the merged submodels after :meth:`build()` has been invoked.
        Depending on the submodels, the top-value can be any of the supported model data-types.


    .. Attribute:: _submodel_tuples

        The stack of (``submodel``, ``path_ops``) tuples. The list's 1st element is the :dfn:`base-model`,
        the last one, the :dfn:`top-model`.  Use the :meth:`add_submodel()` to build this list.


    .. Attribute:: _global_cntxt

        A :class:`ModelOperations` instance acting as the global-default context for the unified-model and all submodels.
        Use :meth:`mod_global_operations()` to modify it.


    .. Attribute:: _curate_funcs

        The sequence of *curate* functions to be executed as the final step by :meth:`_curate()`.
        They are "normal" functions (not generators) with signature::

            def curate_func(model_maker):
                pass      ## ie: modify ``model_maker.model``.

        Better specify this list of functions on construction time.


    .. Attribute:: _errored

            An internal boolean flag that becomes ``True`` if any build-step has failed,
            to halt proceeding to the next one.  It is ``None`` if build has not started yet.


    .. _pandel-examples:

    **Examples**

    The basic usage requires to subclass your own model-maker, just so that a *json-schema*
    is provided for both validation-steps, 2 & 4:

        >>> from collections import OrderedDict as od                           ## Json is better with stable keys-order

        >>> class MyModel(Pandel):
        ...     def _get_json_schema(self, is_prevalidation):
        ...         return {                                                    ## Define the json-schema.
        ...             '$schema': 'http://json-schema.org/draft-04/schema#',
        ...             'required': [] if is_prevalidation else ['a', 'b'],     ## Prevalidation is more loose.
        ...             'properties': {
        ...                 'a': {'type': 'string'},
        ...                 'b': {'type': 'number'},
        ...                 'c': {'type': 'number'},
        ...             }
        ...         }


    Then you can instanciate it and add your submodels:

        >>> mm = MyModel()
        >>> mm.add_submodel(od(a='foo', b=1))                                   ## submodel-1 (base)
        >>> mm.add_submodel(pd.Series(od(a='bar', c=2)))                        ## submodel-2 (top-model)


    You then have to build the final unified-model (any validation errors would be reported at this point):

        >>> mdl = mm.build()

    Note that you can also access the unified-model in the :attr:`model` attribute.
    You can now interogate it:

        >>> mdl['a'] == 'bar'                       ## Value overridden by top-model
        True
        >>> mdl['b'] == 1                           ## Value left intact from base-model
        True
        >>> mdl['c'] == 2                           ## New value from top-model
        True


    Lets try to build with invalid submodels:

        >>> mm = MyModel()
        >>> mm.add_submodel({'a': 1})               ## According to the schema, this should have been a string,
        >>> mm.add_submodel({'b': 'string'})        ## and this one, a number.

        >>> sorted(mm.build_iter(), key=lambda ex: ex.message)    ## Fetch a list with all validation errors. # doctest: +NORMALIZE_WHITESPACE
        [<ValidationError: "'string' is not of type 'number'">,
         <ValidationError: "1 is not of type 'string'">,
         <ValidationError: 'Gave-up building model after step 1.prevalidate (out of 4).'>]

        >>> mdl = mm.model
        >>> mdl is None                                     ## No model constructed, failed before merging.
        True


    And lets try to build with valid submodels but invalid merged-one:

        >>> mm = MyModel()
        >>> mm.add_submodel({'a': 'a str'})
        >>> mm.add_submodel({'c': 1})

        >>> sorted(mm.build_iter(), key=lambda ex: ex.message)  # doctest: +NORMALIZE_WHITESPACE
        [<ValidationError: "'b' is a required property">,
         <ValidationError: 'Gave-up building model after step 3.validate (out of 4).'>]

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, curate_funcs=()):
        """

        :param sequence curate_funcs:   See :attr:`_curate_funcs`.
        """

        self.model = None
        self._errored = None
        self._submodel_tuples = []
        self._curate_funcs = curate_funcs
        self._global_cntxt = []
        self._unified_contexts = None

    def mod_global_operations(self, operations=None, **cntxt_kwargs):
        """

        Since it is the fall-back operation for *conversions* and *IO* operation, it must exist and have
        all its props well-defined for the class to work correctly.

        :param ModelOperations operations:  Replaces values of the installed context with
                                            non-empty values from this one.
        :param cntxt_kwargs:                Replaces the keyworded-values on the existing `operations`.
                                            See :class:`ModelOperations` for supported keywords.
        """
        if operations:
            assert isinstance(operations, ModelOperations), (
                type(operations),
                operations,
            )
            self._global_cntxt = operations
        self._global_cntxt._replace(**cntxt_kwargs)

    @property
    def unified_contexts(self):
        """
        A map of ``json_paths`` --> :class:`ModelOperations` instances acting on the unified-model.
        """
        return self._unified_contexts

    @unified_contexts.setter
    def unified_contexts(self, path_ops):
        assert isinstance(path_ops, cabc.Mapping), (type(path_ops), path_ops)
        self._unified_contexts = path_ops

    def _select_context(self, path, branch):
        """
        Finds which context to use while visiting model-nodes, by enforcing the precedance-rules described
        in the :ref:`Customizations  <pandel-customization>`.

        :param str path:    the branch's jsonpointer-path
        :param str branch:  the actual branch's node
        :return:            the selected :class:`ModelOperations`
        """
        pass

    def _read_branch(self):
        """
        Reads model-branches during *resolve* step.
        """
        pass  # TODO: impl read_branch()

    def _write_branch(self):
        """
        Writes model-branches during *distribute* step.
        """
        pass  # TODO: impl write_branch()

    def _get_json_schema(self, is_prevalidation):
        """
        :return: a json schema, more loose when `prevalidation` for each case
        :rtype: dictionary
        """
        # TODO: Make it a factory o
        pass

    def _rule_AdditionalProperties(self, validator, aP, required, instance, schema):
        properties = schema.get("properties", {})
        patterns = "|".join(schema.get("patternProperties", {}))
        extras = set()
        for prop in instance:
            if prop not in properties:
                if patterns and re.search(patterns, prop):
                    continue
                extras.add(prop)

        if validator.is_type(aP, "object"):
            for extra in extras:
                for error in validator.descend(instance[extra], aP, path=extra):
                    yield error
        elif not aP and extras:
            error = "Additional properties are not allowed (%s %s unexpected)"
            yield ValidationError(error % jsonschema._utils.extras_msg(extras))

    def _rule_Required(self, validator, required, instance, schema):
        if (
            validator.is_type(instance, "object")
            or validator.is_type(instance, "DataFrame")
            or validator.is_type(instance, "Series")
        ):
            for prop in required:
                if prop not in instance:
                    yield ValidationError("%r is a required property" % prop)

    def _get_model_validator(self, schema):
        return PandelVisitor(schema)

    def _validate_json_model(self, schema, mdl):
        validator = self._get_model_validator(schema)
        for err in validator.iter_errors(mdl):
            self._errored = True
            yield err

    def _clone_and_merge_submodels(self, a, b, path=""):
        """' Recursively merge b into a, cloning both. """

        if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
            a = pd.DataFrame() if a is None else pd.DataFrame(a)
            b = pd.DataFrame() if b is None else pd.DataFrame(b)

            a.update(b)  # , 'outer') NOT IMPL YET
            extra_b_items = list(set(b.columns) - set(a.columns))
            a[extra_b_items] = b[extra_b_items]

        elif isinstance(a, pd.Series) or isinstance(b, pd.Series):
            a = pd.Series() if a is None else pd.Series(a)
            b = pd.Series() if b is None else pd.Series(b)
            # a.update(b) # DOES NOT append extra keys!
            a = b.combine_first(a)

        elif isinstance(a, cabc.Mapping) or isinstance(b, cabc.Mapping):
            a = OrderedDict() if a is None else OrderedDict(a)
            b = OrderedDict() if b is None else OrderedDict(b)

            for key in b:
                b_val = b[key]
                if key in a:
                    val = self._clone_and_merge_submodels(
                        a[key], b_val, "%s/%s" % (path, key)
                    )
                else:
                    val = b_val
                a[key] = val

        elif (isinstance(a, cabc.Sequence) and not isinstance(a, str)) or (
            isinstance(b, cabc.Sequence) and not isinstance(b, str)
        ):
            if b is not None:
                val = b
            else:
                val = a

            l = list()
            for (i, item) in enumerate(val):
                l.append(
                    self._clone_and_merge_submodels(item, None, "%s[%i]" % (path, i))
                )
            a = l

        elif a is None and b is None:
            raise ValidationError("Cannot merge Nones at path(%s)!" % path)

        else:
            if b is not None:
                a = b

        return a

    def _resolve(self):
        "Step-1"
        if False:
            yield

    def _prevalidate(self):
        "Step-1"
        for (mdl, path_ops) in self._submodel_tuples:
            schema = self._get_json_schema(is_prevalidation=True)
            for err in self._validate_json_model(schema, mdl):
                yield err

    def _merge(self):
        "Step-2"
        for (mdl, path_ops) in self._submodel_tuples:
            self.model = self._clone_and_merge_submodels(self.model, mdl)
        if False:
            yield  # Just mark method as generator.

    def _validate(self):
        "Step-3"
        schema = self._get_json_schema(is_prevalidation=False)
        for err in self._validate_json_model(schema, self.model):
            yield err

    def _curate(self):
        "Step-4:  Invokes any curate-functions found in :attr:`_curate_funcs`."
        if False:
            yield  # To be overriden, just mark method as generator.
        for curfunc in self._curate_funcs:
            curfunc(self)

    def add_submodel(self, model, path_ops=None):
        """
        Pushes on top a submodel, along with its context-map.

        :param model:               the model-tree (sequence, mapping, pandas-types)
        :param dict path_ops:       A map of ``json_paths`` --> :class:`ModelOperations` instances acting on the
                                    unified-model.  The `path_ops` may often be empty.

        **Examples**

        To change the default DataFrame --> dictionary convertor for a submodel, use the following:

            >>> mdl = {'foo': 'bar'}
            >>> submdl = ModelOperations(mdl, conv={(pd.DataFrame, dict): lambda df: df.to_dict('record')})

        """

        if path_ops:
            assert isinstance(path_ops, cabc.Mapping), (type(path_ops), path_ops)

        return self._submodel_tuples.append((model, path_ops))

    def build_iter(self):
        """
        Iteratively build model, yielding any problems as :class:`ValidationError` instances.

        For debugging, the unified model at :attr:`model` my contain intermediate results at any time,
        even if construction has failed.  Check the :attr:`_errored` flag if neccessary.
        """

        steps = [
            (self._prevalidate, "prevalidate"),
            (self._merge, "merge"),
            (self._validate, "validate"),
            (self._curate, "curate"),
        ]
        self._errored = False
        self.model = None

        for (i, (step, step_name)) in enumerate(steps, start=1):
            try:
                for err in step():
                    yield err
            except ValidationError as ex:
                self._errored = True
                yield ex

            except Exception as ex:
                self._errored = True

                nex = ValidationError(
                    "Model step-%i(%s) failed due to: %s" % (i, step_name, ex)
                )
                nex.cause = ex

                yield nex

            if self._errored:
                yield ValidationError(
                    "Gave-up building model after step %i.%s (out of %i)."
                    % (i, step_name, len(steps))
                )
                break

    def build(self):
        """
        Attempts to build the model by exhausting :meth:`build_iter()`, or raises its 1st error.

        Use this method when you do not want to waste time getting the full list of errors.
        """

        err = next(self.build_iter(), None)
        if err:
            raise err

        return self.model

    def get(self, path, **kws):
        resolve_jsonpointer(self.model, path, **kws)


def escape_jsonpointer_part(part):
    return part.replace("~", "~0").replace("/", "~1")


def unescape_jsonpointer_part(part):
    return part.replace("~1", "/").replace("~0", "~")


def iter_jsonpointer_parts(jsonpath):
    """
    Generates the ``jsonpath`` parts according to jsonpointer spec.

    :param str jsonpath:  a jsonpath to resolve within document
    :return:              The parts of the path as generator), without
                          converting any step to int, and None if None.

    :author: Julian Berman, ankostis

    Examples::

        >>> list(iter_jsonpointer_parts('/a/b'))
        ['a', 'b']

        >>> list(iter_jsonpointer_parts('/a//b'))
        ['a', '', 'b']

        >>> list(iter_jsonpointer_parts('/'))
        ['']

        >>> list(iter_jsonpointer_parts(''))
        []


    But paths are strings begining (NOT_MPL: but not ending) with slash('/')::

        >>> list(iter_jsonpointer_parts(None))
        Traceback (most recent call last):
        AttributeError: 'NoneType' object has no attribute 'split'

        >>> list(iter_jsonpointer_parts('a'))
        Traceback (most recent call last):
        jsonschema.exceptions.RefResolutionError: Jsonpointer-path(a) must start with '/'!

        #>>> list(iter_jsonpointer_parts('/a/'))
        #Traceback (most recent call last):
        #jsonschema.exceptions.RefResolutionError: Jsonpointer-path(a) must NOT ends with '/'!

    """

    #     if jsonpath.endswith('/'):
    #         msg = "Jsonpointer-path({}) must NOT finish with '/'!"
    #         raise RefResolutionError(msg.format(jsonpath))
    parts = jsonpath.split("/")
    if parts.pop(0) != "":
        msg = "Jsonpointer-path({}) must start with '/'!"
        raise RefResolutionError(msg.format(jsonpath))

    for part in parts:
        part = unescape_jsonpointer_part(part)

        yield part


def iter_jsonpointer_parts_relaxed(jsonpointer):
    """
    Like :func:`iter_jsonpointer_parts()` but accepting also non-absolute paths.

    The 1st step of absolute-paths is always ''.

    Examples::

        >>> list(iter_jsonpointer_parts_relaxed('a'))
        ['a']
        >>> list(iter_jsonpointer_parts_relaxed('a/'))
        ['a', '']
        >>> list(iter_jsonpointer_parts_relaxed('a/b'))
        ['a', 'b']

        >>> list(iter_jsonpointer_parts_relaxed('/a'))
        ['', 'a']
        >>> list(iter_jsonpointer_parts_relaxed('/a/'))
        ['', 'a', '']

        >>> list(iter_jsonpointer_parts_relaxed('/'))
        ['', '']

        >>> list(iter_jsonpointer_parts_relaxed(''))
        ['']

    """
    for part in jsonpointer.split("/"):
        yield unescape_jsonpointer_part(part)


_scream = object()


def resolve_jsonpointer(doc, jsonpointer, default=_scream):
    """
    Resolve a ``jsonpointer`` within the referenced ``doc``.

    :param doc:      the referrant document
    :param str path: a jsonpointer to resolve within document
    :param default:  A value to return if path does not resolve.
    :return:         the resolved doc-item or raises :class:`RefResolutionError`
    :raises:     RefResolutionError (if cannot resolve path and no `default`)

    Examples:

        >>> dt = {
        ...     'pi':3.14,
        ...     'foo':'bar',
        ...     'df': pd.DataFrame(np.ones((3,2)), columns=list('VN')),
        ...     'sub': {
        ...         'sr': pd.Series({'abc':'def'}),
        ...     }
        ... }
        >>> resolve_jsonpointer(dt, '/pi', default=_scream)
        3.14

        >>> resolve_jsonpointer(dt, '/pi/BAD')
        Traceback (most recent call last):
        jsonschema.exceptions.RefResolutionError: Unresolvable JSON pointer('/pi/BAD')@(BAD)

        >>> resolve_jsonpointer(dt, '/pi/BAD', 'Hi!')
        'Hi!'

    :author: Julian Berman, ankostis
    """
    for part in iter_jsonpointer_parts(jsonpointer):
        if isinstance(doc, cabc.Sequence):
            # Array indexes should be turned into integers
            try:
                part = int(part)
            except ValueError:
                pass
        try:
            doc = doc[part]
        except (TypeError, LookupError):
            if default is _scream:
                raise RefResolutionError(
                    "Unresolvable JSON pointer(%r)@(%s)" % (jsonpointer, part)
                )
            else:
                return default

    return doc


def resolve_path(doc, path, default=_scream, root=None):
    """
    Like :func:`resolve_jsonpointer` also for relative-paths & attribute-branches.

    :param doc:      the referrant document
    :param str path: An abdolute or relative path to resolve within document.
    :param default:  A value to return if path does not resolve.
    :param root:     Document for absolute paths, assumed `doc` if missing.
    :return:         the resolved doc-item or raises :class:`RefResolutionError`
    :raises:     RefResolutionError (if cannot resolve path and no `default`)

    Examples:

        >>> dt = {
        ...     'pi':3.14,
        ...     'foo':'bar',
        ...     'df': pd.DataFrame(np.ones((3,2)), columns=list('VN')),
        ...     'sub': {
        ...         'sr': pd.Series({'abc':'def'}),
        ...     }
        ... }
        >>> resolve_path(dt, '/pi', default=_scream)
        3.14

        >>> resolve_path(dt, 'df/V')
        0    1.0
        1    1.0
        2    1.0
        Name: V, dtype: float64

        >>> resolve_path(dt, '/pi/BAD', 'Hi!')
        'Hi!'

    :author: Julian Berman, ankostis
    """

    def resolve_root(d, p):
        if not p:
            return root or doc
        raise ValueError()

    part_resolvers = [
        lambda d, p: d[int(p)],
        lambda d, p: d[p],
        lambda d, p: getattr(d, p),
        resolve_root,
    ]
    for part in iter_jsonpointer_parts_relaxed(path):
        start_i = 0 if isinstance(doc, cabc.Sequence) else 1
        for resolver in part_resolvers[start_i:]:
            try:
                doc = resolver(doc, part)
                break
            except (ValueError, TypeError, LookupError, AttributeError):
                pass
        else:
            if default is _scream:
                raise RefResolutionError("Unresolvable path(%r)@(%s)" % (path, part))
            return default

    return doc


def set_jsonpointer(doc, jsonpointer, value, object_factory=OrderedDict):
    """
    Resolve a ``jsonpointer`` within the referenced ``doc``.

    :param doc: the referrant document
    :param str jsonpointer: a jsonpointer to the node to modify
    :raises: RefResolutionError (if jsonpointer empty, missing, invalid-contet)
    """

    parts = list(iter_jsonpointer_parts(jsonpointer))

    # Will scream if used on 1st iteration.
    #
    pdoc = None
    ppart = None
    for i, part in enumerate(parts):
        if isinstance(doc, cabc.Sequence) and not isinstance(doc, str):
            # Array indexes should be turned into integers
            #
            doclen = len(doc)
            if part == "-":
                part = doclen
            else:
                try:
                    part = int(part)
                except ValueError:
                    raise RefResolutionError(
                        "Expected numeric index(%s) for sequence at (%r)[%i]"
                        % (part, jsonpointer, i)
                    )
                else:
                    if part > doclen:
                        raise RefResolutionError(
                            "Index(%s) out of bounds(%i) of (%r)[%i]"
                            % (part, doclen, jsonpointer, i)
                        )
        try:
            ndoc = doc[part]
        except (LookupError):
            break  # Branch-extension needed.
        except (TypeError):  # Maybe indexing a string...
            ndoc = object_factory()
            pdoc[ppart] = ndoc
            doc = ndoc
            break  # Branch-extension needed.

        doc, pdoc, ppart = ndoc, doc, part
    else:
        doc = pdoc  # If loop exhausted, cancel last assignment.

    # Build branch with value-leaf.
    #
    nbranch = value
    for part2 in reversed(parts[i + 1 :]):
        ndoc = object_factory()
        ndoc[part2] = nbranch
        nbranch = ndoc

    # Attach new-branch.
    try:
        doc[part] = nbranch
    # Inserting last sequence-element raises IndexError("list assignment index
    # out of range")
    except IndexError:
        doc.append(nbranch)


#    except (IndexError, TypeError) as ex:
# if isinstance(ex, IndexError) or 'list indices must be integers' in str(ex):
#        raise RefResolutionError("Incompatible content of JSON pointer(%r)@(%s)" % (jsonpointer, part))
#        else:
#            doc = {}
#            parent_doc[parent_part] = doc
#            doc[part] = value


def build_all_jsonpaths(schema):
    # Totally quick an dirty, TODO: Use json-validator to build all json-paths.
    forks = ["oneOf", "anyOf", "allOf"]

    def _visit(schema, path, paths):
        for f in forks:
            objlist = schema.get(f)
            if objlist:
                for obj in objlist:
                    _visit(obj, path, paths)

        props = schema.get("properties")
        if props:
            for p, obj in props.items():
                _visit(obj, path + "/" + p, paths)
        else:
            paths.append(path)

    paths = []
    _visit(schema, "", paths)

    return paths


_NONE = object()
"""Denotes non-existent json-schema attribute in :class:`JSchema`."""


class JSchema(object):

    """
    Facilitates the construction of json-schema-v4 nodes on :class:`PStep` code.

    It does just rudimentary args-name check.   Further validations
    should apply using a proper json-schema validator.

    :param type: if omitted, derived as 'object' if it has children
    :param kws:  for all the rest see http://json-schema.org/latest/json-schema-validation.html

    """

    type = (_NONE,)  # @ReservedAssignment
    items = (_NONE,)  # @ReservedAssignment
    required = (_NONE,)
    title = (_NONE,)
    description = (_NONE,)
    minimum = (_NONE,)
    exclusiveMinimum = (_NONE,)
    maximum = (_NONE,)
    exclusiveMaximum = (_NONE,)
    patternProperties = (_NONE,)
    pattern = (_NONE,)
    enum = (_NONE,)
    allOf = (_NONE,)
    oneOf = (_NONE,)
    anyOf = (_NONE,)

    def todict(self):
        return {k: v for k, v in vars(self).items() if v is not _NONE}


class JSONCodec:

    """
    Json coders/decoders capable for (almost) all python objects, by pickling them.

    Example::

        >>> import json
        >>> obj_list = [
        ...    3.14,
        ...    {
        ...         'aa': pd.DataFrame([]),
        ...         2: np.array([]),
        ...         33: {'foo': 'bar'},
        ...     },
        ...     pd.DataFrame(np.random.randn(10, 2)),
        ...     ('b', pd.Series({})),
        ... ]
        >>> for o in obj_list + [obj_list]:
        ...     s = json.dumps(o, cls=JSONCodec.Encoder)
        ...     oo = json.loads(s, cls=JSONCodec.Decoder)
        ...     #assert trees_equal(o, oo)
        ...

    .. seealso::
        For pickle-limitations: https://docs.python.org/3.7/library/pickle.html#pickle-picklable
    """

    _ver_key = "_ver"
    _ver = "0"
    _obj = "$qpickle"

    class Encoder(JSONEncoder):
        def encode(self, o):
            pickle_bytes = pickle.dumps(o)
            pickle_str = binascii.b2a_qp(pickle_bytes).decode(encoding="utf8")
            o = {JSONCodec._obj: pickle_str, JSONCodec._ver_key: JSONCodec._ver}
            return JSONEncoder.encode(self, o)

    class Decoder(JSONDecoder):
        def decode(self, s):
            o = JSONDecoder.decode(self, s)
            pickle_str = o.get(JSONCodec._obj, None)
            if pickle_str:
                # file_ver = o[JSONCodec._ver_key]
                # if file_ver != JSONCodec._ver:
                #     msg = 'Unsopported json-encoded version(%s != %s)!'
                #     raise ValueError(msg % (file_ver, JSONCodec._ver))
                pickle_bytes = binascii.a2b_qp(pickle_str.encode(encoding="utf8"))
                o = pickle.loads(pickle_bytes)
            return o


if __name__ == "__main__":  # pragma: no cover
    raise NotImplementedError
