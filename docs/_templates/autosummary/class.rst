{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% set own_attributes = attributes | reject("in", inherited_members) | list %}
{% set own_methods = methods | reject("in", inherited_members) | reject("equalto", "__init__") | list %}
{% if own_attributes %}
.. rubric:: Attributes

.. autosummary::
{% for item in own_attributes %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{% if own_methods %}
.. rubric:: Methods

.. autosummary::
{% for item in own_methods %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{% for item in own_attributes %}
.. autoattribute:: {{ objname }}.{{ item }}
{%- endfor %}
{% for item in own_methods %}
.. automethod:: {{ objname }}.{{ item }}
{%- endfor %}
