import click

from ..controllers.training import train_model_controller


@click.group()
def training_group():
    pass


@training_group.command()
@click.option("--padding", is_flag=True)
@click.option("--debug", is_flag=True)
def train(padding, debug):
    train_model_controller(is_padding=padding, debug=debug)
