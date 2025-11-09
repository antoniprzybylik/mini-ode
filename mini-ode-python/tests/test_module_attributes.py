import mini_ode
import tomllib
import pathlib

def test_version():
    with pathlib.Path(__file__).parent.parent.parent.joinpath("Cargo.toml").open('rb') as fp:
        mini_ode_config = tomllib.load(fp)
    expected_version = mini_ode_config["workspace"]["package"]["version"]
    assert expected_version == mini_ode.__version__
