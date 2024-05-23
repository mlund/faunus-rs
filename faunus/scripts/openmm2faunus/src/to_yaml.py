"""
Copyright 2023-2024 Mikael Lund & Ladislav Bartos

Licensed under the Apache license, version 2.0 (the "license");
you may not use this file except in compliance with the license.
You may obtain a copy of the license at

    http://www.apache.org/licenses/license-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the license is distributed on an "as is" basis,
without warranties or conditions of any kind, either express or implied.
See the license for the specific language governing permissions and
limitations under the license.
"""

"""
Modifications of the default pyyaml tags.
"""

# ruff: noqa: E402
import yaml  # type: ignore


# Print no tag for the class.
def notag_representer(dumper, data):
    return dumper.represent_mapping(
        "tag:yaml.org,2002:map",
        dict((k, v) for (k, v) in data.__dict__.items() if v is not None),
    )


# Print no tag for a tuple
def notag_tuple_representer(self, data):
    return self.represent_sequence("tag:yaml.org,2002:seq", data)


yaml.add_representer(tuple, notag_tuple_representer)


# Decorator for adding YAML representers
def yaml_tag(tag):
    def decorator(cls):
        def representer(dumper, data):
            return dumper.represent_mapping(
                tag, dict((k, v) for (k, v) in data.__dict__.items() if v is not None)
            )

        yaml.add_representer(cls, representer)
        return cls

    return decorator


# Decorator for unit class instance.
def yaml_unit(tag):
    def decorator(cls):
        def representer(dumper, data):
            # this does not work properly as it prints '' after the tag
            # the work-around is to just remove all instances of '' from the yaml output but that's not ideal
            # todo: find a proper solution
            return dumper.represent_scalar(tag, "")

        yaml.add_representer(cls, representer)
        return cls

    return decorator


# Decorator for Faunus Default.
def yaml_default():
    def decorator(cls):
        def representer(dumper, obj):
            return dumper.represent_scalar("tag:yaml.org,2002:str", "default", style="")

        yaml.add_representer(cls, representer)
        return cls

    return decorator


# Classes without a YAML representer decorator should have no tags.
yaml.add_multi_representer(object, notag_representer)
