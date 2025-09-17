from pathlib import Path
from cluey import Cluey, main, tool, method


class WiDiTApp(Cluey):
    @method
    def model(
        self,
        model: str = "WiDiT3D-B/2",
        input_size: int = 100,
        in_channels: int = 1,
        use_diffusion: bool = True,   
        verbose:bool=False,     
        **kwargs,
    ):
        from widit import PRESETS
        assert model in PRESETS, f"Model '{model}' not in PRESETS: {list(PRESETS.keys())}"

        model_name = model

        model = PRESETS[model_name](
            input_size=input_size,
            in_channels=in_channels,
            learn_sigma=use_diffusion, # if using diffusion, learn sigma
            use_conditioning=use_diffusion,
        )

        if verbose:
            total = sum(p.numel() for p in model.parameters())
            print(f"Model: {model_name}")
            print(f"Model Summary:\n{model}")
            print(f"Model '{model_name}' has {total:,} parameters")
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model '{model_name}' has {trainable:,} trainable parameters")

        return model

    @method
    def datasets(
        self,
        **kwargs,
    ) -> tuple:
        """ Returns training and validation datasets """
        raise NotImplementedError("Datasets method not yet implemented.")

    @method("datasets")
    def dataloaders(
        self,
        num_workers:int = 4,
        batch_size:int = 1,
        **kwargs,
    ):
        from torch.utils.data import DataLoader

        training_dataset, validation_dataset = self.datasets(**kwargs)
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        validation_dataloader = None
        if validation_dataset is not None:
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        
        return training_dataloader, validation_dataloader

    @tool("model", "dataloaders")
    def train(
        self,
        epochs: int = 40,
        log_every: int = 100,
        results_dir: Path = "./results",
        use_diffusion: bool = True,
        wandb: bool = False,
        wandb_project: str = "",
        run_name: str = "",
        **kwargs,
    ):
        """ Train the model """

        from .training import train

        model = self.model(use_diffusion=use_diffusion, **kwargs)

        wandb_project = wandb_project or str(self.__class__.__name__)

        training_dataloader, validation_dataloader = self.dataloaders(**kwargs)
        train(
            model=model,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            results_dir=results_dir,
            use_diffusion=use_diffusion,
            epochs=epochs,
            log_every=log_every,
            run_name=run_name,
            wandb_logging=wandb,
            wandb_project=wandb_project,
        )

    @main
    def predict(
        self,
        **kwargs,
    ):
        """ Makes predictions """
        raise NotImplementedError("Prediction not yet implemented.")