# :control_knobs: Use configuration files for the NARPS open pipelines project

## Purpose

The package `narps_open.utils.configuration` allows to load configuration files at runtime. This is useful, especially while running pipelines in several environments.

The class `Configuration` of module `narps_open.utils.configuration` acts as a parser for configuration files that are installed with the package `narps_open`. `Configuration` is a singleton (see `narps_open.utils.singleton`), meaning that there is only one instance of this object per runtime. This makes sure only one configuration type is used at a time (avoiding mixing parameters from several configurations).

For now, two configuration types are available:
* `testing` : used for testing purposes only (while running `pytest`)
* `default` : the default configuration

## Usage

```python
from narps_open.utils.configuration import Configuration

configuration = Configuration(config_type = 'testing')
dir_dataset = configuration['directories']['dataset'] # gets the directory where the dataset is
```

You can also set a custom configuration file that better suits your needs.

```python
from narps_open.utils.configuration import Configuration

configuration = Configuration(config_type = 'custom')
configuration.config_file = '/path/to/my/own/configuration/file.toml'
```

## Writing configuration files

Configuration files must conform with the [TOML](https://toml.io/en/) format. See this [article](https://realpython.com/python-toml/#use-toml-as-a-configuration-format) on Real Python to know more about configuration files with TOML.

For python versions below 3.11, we use [tomli](https://pypi.org/project/tomli/) as a dependency for parsing TOML. Starting from python 3.11, [tomllib](https://docs.python.org/3/library/tomllib.html) is included in the Standard Library and would replace tomli.
