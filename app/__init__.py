from click import CommandCollection

from .routes.training import training_group

cli = CommandCollection(sources=[training_group])
