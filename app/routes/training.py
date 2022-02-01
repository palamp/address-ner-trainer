import click

from ..controllers.training import train_model_controller


@click.group()
def training_group():
    pass


@training_group.command()
@click.option("--debug", is_flag=True)
@click.option("--early_stop", is_flag=True)
def train(debug, early_stop):
    train_model_controller(debug=debug, early_stop=early_stop)
