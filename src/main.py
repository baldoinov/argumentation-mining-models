import lightning as L

from lightning.pytorch.cli import LightningCLI, ArgsType

from src.model import ArgumentationMiningTaskModel
from src.dataset import TweetsDataModule


def main(args: ArgsType = None):
    cli = LightningCLI(
        model_class=ArgumentationMiningTaskModel, datamodule_class=TweetsDataModule, args=args
    )


if __name__ == "__main__":
    main()
