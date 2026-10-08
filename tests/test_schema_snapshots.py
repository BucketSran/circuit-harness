from importlib.resources import files

from alphaapollo._schema_export import SCHEMA_OWNERS, check_snapshots


def test_component_json_schema_snapshots_are_current() -> None:
    assert check_snapshots() == []


def test_all_schema_snapshots_are_owned_package_resources() -> None:
    for owner in SCHEMA_OWNERS:
        expected = {f"{name}.json" for name in owner.models}
        resources = files(owner.package).joinpath("json")
        packaged = {item.name for item in resources.iterdir() if item.name.endswith(".json")}
        assert packaged == expected
