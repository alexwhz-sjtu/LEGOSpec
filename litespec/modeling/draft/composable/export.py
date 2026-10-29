"""Inference-only flat dependency bundle for Transformers dynamic loading."""

import ast
import importlib.util
from pathlib import Path

PREFIX = "litespec.modeling.draft."


def export_composable(model, save_dir, *, state_dict=None):
    source_root = Path(__file__).resolve().parents[1]
    modules = {}
    pending = [PREFIX + "composable.modeling", PREFIX + "composable.configuration"]

    def path_for(name):
        relative = name.removeprefix(PREFIX).replace(".", "/")
        path = source_root / (relative + ".py")
        return path if path.is_file() else source_root / relative / "__init__.py"

    def resolve(node, module, path):
        if node.level:
            package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
            return importlib.util.resolve_name(
                "." * node.level + (node.module or ""), package
            )
        return node.module

    while pending:
        module = pending.pop()
        if module in modules:
            continue
        path = path_for(module)
        if not path.is_file():
            raise FileNotFoundError(path)
        source = path.read_text()
        tree = ast.parse(source)
        modules[module] = (path, source, tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                name = resolve(node, module, path)
                if name and name.startswith(PREFIX):
                    pending.append(name)
                elif name and name.startswith("litespec."):
                    raise ValueError(
                        "Deployment dependency escapes inference package: " + name
                    )
    dependencies = {
        module: {resolve(node, module, path) for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)}
        for module, (path, _, tree) in modules.items()
    }
    names = {
        module: "cs_" + module.removeprefix(PREFIX).replace(".", "_")
        for module in modules
    }
    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.config.auto_map = {
        "AutoConfig": names[PREFIX + "composable.configuration"]
        + ".ComposableDraftConfig",
        "AutoModel": names[PREFIX + "composable.modeling"] + ".ComposableDraftModel",
    }
    model.save_pretrained(destination, state_dict=state_dict)
    for module, (path, source, tree) in modules.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                name = resolve(node, module, path)
                if name in names:
                    node.level, node.module = 1, names[name]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                node.decorator_list = [
                    d
                    for d in node.decorator_list
                    if not (isinstance(d, ast.Name) and d.id == "register_draft")
                ]
        ast.fix_missing_locations(tree)
        notices = "\n".join(
            line for line in source.splitlines()[:25] if line.startswith("#")
        )
        text = (
            "# Generated inference bundle from "
            + str(path.relative_to(source_root))
            + "\n"
            + notices
            + "\n"
            + ast.unparse(tree)
            + "\n"
        )
        if module == PREFIX + "composable.configuration":
            # AutoConfig loads before AutoModel. Transformers copies only direct
            # dependencies before hashing its recursive closure, so advertise
            # all transitive pure-config modules directly as well.
            closure, pending_config = set(), [module]
            while pending_config:
                dependency = pending_config.pop()
                for target in dependencies[dependency]:
                    if target in modules and target != module and target not in closure:
                        closure.add(target)
                        pending_config.append(target)
            text += "\n" + "\n".join(
                "from ." + names[name] + " import __name__ as _export_config_" + names[name]
                for name in sorted(closure)
            ) + "\n"
        if module == PREFIX + "composable.modeling":
            text += (
                "\n# Direct closure for Transformers 4.57 local-directory loading.\n"
            )
            text += (
                "\n".join(
                    "from ." + value + " import __name__ as _export_" + value
                    for key, value in names.items()
                    if key != module
                )
                + "\n"
            )
        (destination / (names[module] + ".py")).write_text(text)
    return sorted(names.values())
