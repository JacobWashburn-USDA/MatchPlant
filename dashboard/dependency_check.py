"""Check installed distribution versions in a module's Python environment."""
import json
import re
import subprocess
from packaging.requirements import Requirement, InvalidRequirement
from packaging.markers import default_environment
from packaging.version import InvalidVersion


def _normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements(text):
    requirements = []
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line or line.startswith('-'):
            continue
        try:
            requirements.append(Requirement(line))
        except InvalidRequirement:
            continue
    return requirements


def _requirement_names(text):
    return [requirement.name for requirement in _requirements(text)]


def missing_packages(python_exe, requirements_text, timeout=10):
    requirements = _requirements(requirements_text)
    if not requirements:
        return []
    probe = """import importlib.metadata as m, json, platform, sys, os
print(json.dumps({'packages': {d.metadata['Name']:d.version for d in m.distributions() if d.metadata.get('Name')}, 'environment': {'python_version':'.'.join(map(str,sys.version_info[:2])), 'python_full_version':platform.python_version(), 'sys_platform':sys.platform, 'platform_system':platform.system(), 'platform_machine':platform.machine(), 'os_name':os.name}}))
"""
    try:
        result = subprocess.run([python_exe, '-c', probe], capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            return [str(requirement) for requirement in requirements]
        data = json.loads(result.stdout)
        installed = {_normalize(name): version for name, version in data['packages'].items()}
        environment = default_environment()
        environment.update(data['environment'])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return [str(requirement) for requirement in requirements]
    missing = []
    for requirement in requirements:
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        version = installed.get(_normalize(requirement.name))
        try:
            valid = version is not None and requirement.specifier.contains(version, prereleases=True)
        except InvalidVersion:
            valid = False
        if not valid:
            missing.append(str(requirement))
    return missing
