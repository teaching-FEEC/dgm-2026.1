import torch


@torch.no_grad()
def validate(trainer, dataloader):

    trainer.G_H2P.eval()
    trainer.G_P2H.eval()

    losses = {
        "cycle": 0.0
    }

    for batch in dataloader:

        real_H = batch["healthy"].to(
            trainer.device
        )

        real_P = batch["pneumonia"].to(
            trainer.device
        )

        fake_P = trainer.G_H2P(real_H)
        rec_H = trainer.G_P2H(fake_P)

        fake_H = trainer.G_P2H(real_P)
        rec_P = trainer.G_H2P(fake_H)

        loss_cycle_H = trainer.cycle_loss(
            rec_H,
            real_H
        )

        loss_cycle_P = trainer.cycle_loss(
            rec_P,
            real_P
        )

        loss = (
            loss_cycle_H +
            loss_cycle_P
        )

        losses["cycle"] += loss.item()

    losses["cycle"] /= len(dataloader)

    return losses