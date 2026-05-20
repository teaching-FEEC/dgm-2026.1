from pathlib import Path
import torch


def save_checkpoint(
    trainer,
    epoch,
    checkpoint_dir
):

    checkpoint_dir = Path(
        checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "G_H2P":
                trainer.G_H2P.state_dict(),

            "G_P2H":
                trainer.G_P2H.state_dict(),

            "D_H":
                trainer.D_H.state_dict(),

            "D_P":
                trainer.D_P.state_dict(),

            "optimizer_G":
                trainer.optimizer_G.state_dict(),

            "optimizer_D_H":
                trainer.optimizer_D_H.state_dict(),

            "optimizer_D_P":
                trainer.optimizer_D_P.state_dict(),

            "epoch": epoch
        },

        checkpoint_dir / f"epoch_{epoch}.pt"
    )